import {
  app,
  BrowserWindow,
  ipcMain,
  Menu,
  Tray,
  Notification,
  nativeImage,
  screen,
  shell,
  dialog,
} from "electron";
import * as path from "path";
import { Config } from "./config";
import { GeminiClient } from "./core/gemini/client";
import { MemoryStore } from "./core/rag/memory";
import { Evaluator } from "./core/brain/evaluator";
import { Heart } from "./core/heart/stateMachine";
import { DiaryWriter } from "./core/diary/writer";
import { PoseStudio } from "./core/face/poseStudio";
import { EMOTION_POSE, getPose, POSES } from "./core/face/poses";
import { Presence } from "./presence";
import { EventServer } from "./server";
import { EasterEggs } from "./easterEggs";
import { EndingDirector } from "./ending";
import { FIRST_DAY_TXT, ONBOARDING } from "./core/endingTexts";
import { SYSTEM_PERSONA, buildMemoryBlock } from "./core/persona";
import { DayLog, Emotion, SensorEvent } from "./core/types";

const PRELOAD = path.join(__dirname, "preload.js");
const RENDERER = path.join(__dirname, "..", "renderer");
const BUNDLED_POSES = path.join(__dirname, "..", "assets", "poses");

let win: BrowserWindow | undefined;
let tray: Tray | undefined;
let diarySaved = false;
let quitting = false;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

// 백그라운드 상주 앱(올라마처럼): 한 번에 하나만 뜨고, 또 켜면 기존 위젯을 보여준다.
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (win) {
      win.show();
    }
  });
}

// ── 하루 누적 (일기 재료) ────────────────────────────────
let day: DayLog = freshDay();
function freshDay(): DayLog {
  return {
    date: new Date().toISOString().slice(0, 10),
    learned: [],
    moments: [],
    peakEmotion: "calm",
  };
}

app.whenReady().then(async () => {
  const config = new Config();
  const gemini = new GeminiClient(config);
  const memory = new MemoryStore(config.memoryFile, gemini);
  await memory.load();

  const evaluator = new Evaluator(gemini, memory);
  const heart = new Heart();
  const diary = new DiaryWriter(gemini, memory, config.diaryFolder);
  const bundledPoses = path.join(__dirname, "..", "assets", "poses");
  const poses = new PoseStudio(config.poseCacheDir, gemini, bundledPoses);
  await poses.load();

  const presence = new Presence(heart);

  // ── 렌더러 뷰 전송 (단일 채널 'state') ───────────────────
  // 위젯은 언제나 미리 생성해 둔 일관된 포즈 PNG만 그린다. (SVG 폴백 없음)
  // 사라짐(gone)도 같은 채널에 실어 보내고, 렌더러가 'ready' 핸드셰이크를
  // 보낸 뒤에만 전송한다 → 초기 메시지 유실 방지.
  let rendererReady = false;
  let goneFlag = false;
  let lastView: Record<string, unknown> = { gone: false };

  async function viewForPose(
    poseId: string,
    line?: string,
    rageLevel = 0
  ): Promise<Record<string, unknown>> {
    const frames = await poses.frameDataUrls(poseId);
    return {
      gone: false,
      pose: poseId,
      rageLevel,
      poseFrames: frames,
      poseFrameMs: getPose(poseId)?.frameMs ?? 0,
      line,
    };
  }

  function send(view: Record<string, unknown>): void {
    lastView = view;
    if (win && rendererReady) {
      win.webContents.send("state", view);
    }
  }

  /** 현재 감정 상태(또는 사라짐)를 위젯에 반영. */
  async function sendCurrent(): Promise<void> {
    if (goneFlag) {
      send({ gone: true });
      return;
    }
    const s = heart.current();
    // 엔딩 복귀 시퀀스 중엔 심장의 평소 대사("왔구나!" 등)를 누른다 — 비트가 톤을 쥔다.
    const line = ending.isReturning() ? "" : s.line;
    send(await viewForPose(EMOTION_POSE[s.emotion], line, s.rageLevel));
  }

  /** 한 포즈 + 한 마디를 잠깐 띄운다 (쓰다듬기/이스터에그/엔딩 비트). */
  function say(pose: string, line: string): void {
    if (goneFlag) {
      return;
    }
    void viewForPose(pose, line).then(send);
  }

  /** 유저 입력창에 한 줄을 자동 타이핑 (엔딩 대화처럼 보이게). */
  function autotype(text: string): void {
    win?.webContents.send("autotype", { text });
  }

  /** OS 알림으로 앱 레벨에서 표시. 위젯을 안 보고 있어도 알 수 있게. */
  function notify(title: string, body: string, openFile?: string): void {
    try {
      if (!Notification.isSupported()) {
        return;
      }
      const n = new Notification({ title, body, silent: false });
      if (openFile) {
        n.on("click", () => void shell.openPath(openFile));
      }
      n.show();
    } catch {
      /* 알림 미지원 환경은 조용히 무시 */
    }
  }

  heart.onChange(() => void sendCurrent());
  const eggs = new EasterEggs(say);
  eggs.start();

  // 꼬질룡의 인생 전체(온보딩~엔딩): 진짜 위젯 동작 + 진짜 파일 + 앱 레벨 알림.
  const ending = new EndingDirector(
    {
      stateFile: config.endingStateFile,
      diaryFolder: config.diaryFolder,
      projectFolder: config.projectFolder,
      setGone: (gone) => {
        goneFlag = gone;
        void sendCurrent();
      },
      say: (pose, line) => say(pose, line),
      autotype,
      openPath: (p) => void shell.openPath(p),
      notify,
    },
    process.env.KKOJI_ENDING_DEMO === "1"
  );

  // 센서 이벤트 처리.
  let busy = false;
  const server = new EventServer(config.serverPort, (e: SensorEvent) => {
    void handleEvent(e);
  });
  async function handleEvent(e: SensorEvent): Promise<void> {
    switch (e.type) {
      case "heartbeat":
        presence.beat();
        return;
      case "blur":
        presence.blur();
        return;
      case "build":
        presence.beat();
        ending.onActivity();
        if (ending.isGone() || onboarding) {
          return; // 사라진 동안/온보딩 중엔 조용히.
        }
        if (e.ok) {
          heart.applyEvaluation({
            pose: "joy",
            rageLevel: 0,
            line: e.warnings ? "빌드 됐다! 근데 경고가 좀…" : "초록불!! 최고야!!",
            newConcepts: [],
            knownConcepts: [],
          });
          day.moments.push("빌드 성공");
        }
        return;
      case "code":
        presence.beat();
        ending.onActivity();
        if (ending.isGone() || onboarding || busy) {
          return; // 사라진 동안/온보딩 중엔 조용히 / 한 번에 하나만.
        }
        busy = true;
        try {
          // 침묵 곡선(§17.1)은 evaluator가 단계별 프롬프트로 처리한다.
          // 많이 배울수록 모델 스스로 말을 줄이고, 빈 줄(침묵)을 낸다.
          const ev = await evaluator.evaluate(e.code, e.languageId);
          heart.applyEvaluation(ev);
          eggs.onCode(e.code); // 이스터에그 감지
          day.learned.push(...ev.newConcepts);
          day.peakEmotion = strongest(day.peakEmotion, ev.pose);
          if (ev.pose === "rage") {
            day.moments.push(`코드 보고 빡쳤다: ${ev.line}`);
          } else if (ev.pose === "joy") {
            day.moments.push("주인 코드 멋져서 춤췄다.");
          }
        } catch (err) {
          if (!String(err).includes("NO_API_KEY")) {
            console.error("[꼬질룡] 평가 실패:", err);
          }
        } finally {
          busy = false;
        }
        return;
    }
  }

  server.start();
  presence.start();
  createWindow();
  createTray();
  await ending.init();

  // 첫 설치면 print부터 가르쳐 받는다 (수미상관의 시작).
  let onboarding = ending.needsOnboarding();

  // 시간/방치 맥락 틱.
  const tickTimer = setInterval(() => heart.tick(), 60_000);

  // ── IPC ────────────────────────────────────────────────
  // 렌더러가 로드 완료를 알리면 그때 현재 뷰를 보낸다 (초기 메시지 유실 방지).
  ipcMain.on("ready", () => {
    rendererReady = true;
    if (win) {
      win.webContents.send("state", lastView);
    }
    void sendCurrent();
    if (onboarding) {
      void startOnboarding();
    } else if (!hasKeyHinted) {
      void maybeHintKey();
    }
  });

  /** 첫날: 유저가 print를 가르쳐 줘야 꼬질룡이 깨어난다. */
  async function startOnboarding(): Promise<void> {
    win?.webContents.send("mode", { input: true, placeholder: "print 라고 쳐봐" });
    await sleep(600);
    say("worry", ONBOARDING.ask);
  }

  let onboardingBusy = false;
  /** 유저가 입력창에 친 말. 온보딩 중이면 print 가르치기, 아니면 자유 대화. */
  async function onTalk(text: string): Promise<void> {
    const t = (text || "").trim();
    if (!t) {
      return;
    }
    if (onboarding) {
      if (onboardingBusy) {
        return;
      }
      if (!/print/i.test(t)) {
        say("worry", ONBOARDING.retry);
        return;
      }
      onboardingBusy = true;
      onboarding = false;
      for (const b of ONBOARDING.learn) {
        say(b.pose || "worry", b.line);
        await sleep(b.gapMs);
      }
      // 1일차 일기 = first_day.txt 그대로 (AI 없이). 수미상관의 첫 매듭.
      try {
        const p = await diary.writeRaw(FIRST_DAY_TXT);
        await diary.writeRaw(FIRST_DAY_TXT, "first_day"); // 영구 보관본도 같이
        notify("꼬질룡", "첫 일기를 썼어요.", p);
      } catch {
        /* 조용히 */
      }
      await ending.incDiary();
      await ending.markOnboarded();
      // 이제부터 늘 같이 있는다 — 백그라운드 자동 실행 켠다(언제든 끌 수 있음).
      try {
        app.setLoginItemSettings({ openAtLogin: true });
      } catch {
        /* 플랫폼 미지원 무시 */
      }
      refreshTray();
      win?.webContents.send("mode", { input: false });
      await sleep(900);
      say("moved", "이제 네가 코딩하는 거 보고 싶어. (우클릭 → Gemini 키)");
      onboardingBusy = false;
      return;
    }
    // 평소 자유 대화: 짧게 한 마디 받아준다.
    await freeChat(t);
  }
  ipcMain.on("talk", (_e, text: string) => void onTalk(text));

  /** 키가 없으면 부드럽게 한 번만 안내 (닦달 금지). */
  let hasKeyHinted = false;
  async function maybeHintKey(): Promise<void> {
    hasKeyHinted = true;
    if (!(await config.hasApiKey())) {
      await sleep(700);
      say("worry", "우클릭 메뉴에서 Gemini 키 넣어줘. 그래야 너 코딩하는 거 봐.");
    }
  }

  /** 위젯 입력창으로 말 걸면 꼬질룡이 짧게 답한다 (키 있을 때). */
  async function freeChat(text: string): Promise<void> {
    if (goneFlag) {
      return;
    }
    if (!(await config.hasApiKey())) {
      say("worry", "우클릭 → Gemini 키 넣어줘. 그래야 너랑 얘기하지.");
      return;
    }
    try {
      const reply = await gemini.generateText(
        SYSTEM_PERSONA,
        `${buildMemoryBlock(memory.knownConcepts())}\n\n주인이 너한테 말했어: "${text}"\n짧게 한 마디로 대답해. (한 문장)`,
        { temperature: 0.95 }
      );
      heart.touch();
      say("calm", reply.replace(/\s+/g, " ").trim().slice(0, 80));
    } catch {
      /* 조용히 */
    }
  }

  ipcMain.on("pet", () => {
    heart.touch();
    say("pet", "에헤헤… 또 쓰다듬어줘…");
  });

  ipcMain.on("menu", () => showMenu());

  /** 위젯 우클릭 메뉴 / 트레이 메뉴가 공유하는 항목들. */
  function buildMenuItems(): Electron.MenuItemConstructorOptions[] {
    return [
      { label: "쓰다듬기", click: () => ipcMain.emit("pet") },
      { type: "separator" },
      {
        label: "오늘 일기 보기",
        click: async () => {
          const p = (await diary.todayPath()) ?? (await writeDiary(false, true));
          if (p) {
            shell.openPath(p);
          } else {
            say("worry", "아직 일기 없어… 키 넣고 같이 코딩하면 써줄게.");
          }
        },
      },
      { label: "일기 폴더 열기", click: () => void shell.openPath(config.diaryFolder) },
      {
        label: "꼬질룡이 아는 것…",
        click: () => {
          const o = memory.oldest();
          dialog.showMessageBox({
            message: o
              ? `${growthStage(memory.size)} · 아는 개념 ${memory.size}개\n제일 먼저 배운 건 "${o.concept}" (${o.firstSeen}).`
              : "알 · 아직 아무것도 몰라… 너 코딩하는 거 보여줘.",
          });
        },
      },
      { type: "separator" },
      { label: "Gemini API 키 입력…", click: () => void askKey() },
      {
        label: "시작할 때 자동 실행",
        type: "checkbox",
        checked: app.getLoginItemSettings().openAtLogin,
        click: (mi) => app.setLoginItemSettings({ openAtLogin: mi.checked }),
      },
      { label: "포즈 다시 그리기 (개발용)", click: () => void generatePoses() },
      { type: "separator" },
      { label: "꼬질룡 끄기…", click: () => void turnOff() },
      { label: "종료", click: () => quitApp() },
    ];
  }

  function showMenu(): void {
    Menu.buildFromTemplate(buildMenuItems()).popup({ window: win });
  }

  function toggleWidget(): void {
    if (win?.isVisible()) {
      win.hide();
    } else {
      win?.show();
    }
    refreshTray();
  }

  function refreshTray(): void {
    if (!tray) {
      return;
    }
    tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: win?.isVisible() ? "잠깐 숨기기" : "보이기", click: toggleWidget },
        ...buildMenuItems(),
      ])
    );
  }

  function createTray(): void {
    let icon = nativeImage.createFromPath(
      path.join(BUNDLED_POSES, "calm_0.png")
    );
    if (!icon.isEmpty()) {
      icon = icon.resize({ width: 18, height: 18 });
      icon.setTemplateImage(true);
    }
    tray = new Tray(icon);
    tray.setToolTip("꼬질룡 — 오늘도 코딩하자");
    tray.on("click", () => toggleWidget());
    refreshTray();
  }

  async function askKey(): Promise<void> {
    const value = await promptString("Gemini API 키 (BYOK)");
    if (!value) {
      return;
    }
    config.setApiKey(value);
    say("joy", "오! 이제 너 코딩하는 거 볼 수 있어!");
  }

  async function writeDiary(
    announce = false,
    force = false
  ): Promise<string | undefined> {
    try {
      if (!force && (await diary.hasToday())) {
        return diary.todayPath(); // 오늘 일기가 이미 있음(1일차 등) — 안 덮어쓴다.
      }
      const snapshot = { ...day, learned: dedupe(day.learned) };
      day = freshDay();
      const p = await diary.write(snapshot);
      await ending.incDiary();
      if (announce) {
        say("moved", "오늘 일기 다 썼어. 사각사각.");
      }
      return p;
    } catch (err) {
      if (!String(err).includes("NO_API_KEY")) {
        dialog.showErrorBox("꼬질룡", `일기 실패: ${err}`);
      }
      return undefined;
    }
  }

  async function generatePoses(): Promise<void> {
    if (!(await config.hasApiKey())) {
      dialog.showMessageBox({ message: "먼저 Gemini API 키부터 입력해줘." });
      return;
    }
    say("focus", "포즈 그리는 중… 조금만 기다려…");
    try {
      await poses.generateAll();
      void sendCurrent(); // 새 PNG로 갱신.
      say("moved", "이게… 진짜 내 모습이야. 어때?");
    } catch (err) {
      dialog.showErrorBox("꼬질룡", `포즈 생성 실패: ${err}`);
    }
  }

  function quitApp(): void {
    quitting = true;
    app.quit();
  }

  // §17.8 — 끄기. "삭제"가 아니라 "끄기", 되돌릴 수 있다. 작별 편지를 남긴다.
  async function turnOff(): Promise<void> {
    const { response } = await dialog.showMessageBox({
      type: "question",
      message: "정말 끄시겠어요?",
      detail: "꼬질룡은 되돌릴 수 있습니다. (일기는 폴더에 그대로 남아요.)",
      buttons: ["취소", "끄기"],
      defaultId: 0,
      cancelId: 0,
    });
    if (response !== 1) {
      return;
    }
    await ending.turnOff();
    app.quit();
  }

  ipcMain.on("gone-click", () => ending.onGoneClick());

  // 종료 시 일기 저장.
  app.on("before-quit", (e) => {
    quitting = true;
    if (diarySaved) {
      return;
    }
    if (day.learned.length || day.moments.length) {
      e.preventDefault();
      diarySaved = true;
      void writeDiary().finally(() => {
        clearInterval(tickTimer);
        server.stop();
        presence.stop();
        app.quit();
      });
    }
  });

  void POSES; // (참조 유지) 포즈 카탈로그.
});

function createWindow(): void {
  const { workArea } = screen.getPrimaryDisplay();
  const w = 210;
  const h = 250;
  win = new BrowserWindow({
    width: w,
    height: h,
    x: workArea.x + workArea.width - w - 16,
    y: workArea.y + workArea.height - h - 16,
    frame: false,
    transparent: true,
    resizable: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    webPreferences: {
      preload: PRELOAD,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.setAlwaysOnTop(true, "screen-saver");
  win.loadFile(path.join(RENDERER, "index.html"));
  // 닫아도 안 꺼진다 — 트레이로 숨을 뿐. (진짜 종료는 트레이 "종료"로만)
  win.on("close", (e) => {
    if (!quitting) {
      e.preventDefault();
      win?.hide();
    }
  });
  win.on("closed", () => (win = undefined));
}

/** 작은 모달 입력 창. @returns 입력값 또는 취소 시 undefined. */
function promptString(title: string): Promise<string | undefined> {
  return new Promise((resolve) => {
    const p = new BrowserWindow({
      width: 400,
      height: 170,
      parent: win,
      modal: true,
      resizable: false,
      title,
      webPreferences: { preload: PRELOAD, contextIsolation: true },
    });
    p.setMenuBarVisibility(false);
    p.loadFile(path.join(RENDERER, "prompt.html"));
    p.webContents.once("did-finish-load", () =>
      p.webContents.send("prompt:init", { title })
    );
    const onSubmit = (_e: unknown, value: string | null) => {
      ipcMain.removeListener("prompt:submit", onSubmit);
      resolve(value || undefined);
      if (!p.isDestroyed()) {
        p.close();
      }
    };
    ipcMain.on("prompt:submit", onSubmit);
    p.on("closed", () => {
      ipcMain.removeListener("prompt:submit", onSubmit);
      resolve(undefined);
    });
  });
}

// 백그라운드 상주 앱: 창을 다 닫아도 트레이에 살아있는다. 종료는 트레이로만.
app.on("window-all-closed", () => {
  /* 일부러 아무것도 안 한다 (상주). */
});

function strongest(a: Emotion, b: Emotion): Emotion {
  const weight: Record<Emotion, number> = {
    calm: 0,
    focus: 1,
    sleepy: 1,
    sulk: 2,
    worry: 3,
    rage: 4,
    joy: 4,
    moved: 5,
  };
  return weight[b] > weight[a] ? b : a;
}

function dedupe(arr: string[]): string[] {
  return [...new Set(arr.map((s) => s.trim()).filter(Boolean))];
}

/** 성장 단계 (기획서 §8). 화려해지진 않고, 아는 게 많아진다. */
function growthStage(size: number): string {
  if (size < 1) return "알";
  if (size < 50) return "깬 꼬질룡";
  if (size < 300) return "배우는 꼬질룡";
  if (size < 800) return "똑똑해진 꼬질룡";
  return "용이 된 꼬질룡";
}


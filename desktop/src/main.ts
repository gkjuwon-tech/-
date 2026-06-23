import {
  app,
  BrowserWindow,
  ipcMain,
  Menu,
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
import { DayLog, Emotion, SensorEvent } from "./core/types";

const PRELOAD = path.join(__dirname, "preload.js");
const RENDERER = path.join(__dirname, "..", "renderer");

let win: BrowserWindow | undefined;
let diarySaved = false;

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
    send(await viewForPose(EMOTION_POSE[s.emotion], s.line, s.rageLevel));
  }

  /** 한 포즈 + 한 마디를 잠깐 띄운다 (쓰다듬기/이스터에그/엔딩 비트). */
  function say(pose: string, line: string): void {
    if (goneFlag) {
      return;
    }
    void viewForPose(pose, line).then(send);
  }

  heart.onChange(() => void sendCurrent());
  const eggs = new EasterEggs(say);
  eggs.start();

  // 엔딩 아크 (§17): 진짜 사라졌다 며칠 뒤 돌아오고, 진짜 파일을 남긴다.
  const ending = new EndingDirector(
    {
      stateFile: config.endingStateFile,
      diaryFolder: config.diaryFolder,
      projectFolder: config.projectFolder,
      setGone: (gone) => {
        goneFlag = gone;
        void sendCurrent();
      },
      say: (pose, line) => say(pose as Emotion, line),
      openPath: (p) => void shell.openPath(p),
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
        if (ending.isGone()) {
          return; // 사라진 동안엔 조용히.
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
        if (ending.isGone() || busy) {
          return; // 사라진 동안엔 조용히 / 한 번에 하나만.
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
  await ending.init();

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
  });

  ipcMain.on("pet", () => {
    heart.touch();
    say("pet", "에헤헤… 또 쓰다듬어줘…");
  });

  ipcMain.on("menu", () => showMenu());

  function showMenu(): void {
    const menu = Menu.buildFromTemplate([
      { label: "쓰다듬기", click: () => ipcMain.emit("pet") },
      { type: "separator" },
      {
        label: "오늘 일기 보기",
        click: async () => {
          const p = (await diary.todayPath()) ?? (await writeDiary());
          if (p) {
            shell.openPath(p);
          }
        },
      },
      { label: "지금 일기 쓰기", click: () => void writeDiary(true) },
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
      { label: "포즈 다시 그리기 (개발용)", click: () => void generatePoses() },
      {
        label: "설정 폴더 열기",
        click: () => shell.showItemInFolder(config.settingsFile),
      },
      { type: "separator" },
      { label: "꼬질룡 끄기…", click: () => void turnOff() },
      { label: "종료", click: () => app.quit() },
    ]);
    menu.popup({ window: win });
  }

  async function askKey(): Promise<void> {
    const value = await promptString("Gemini API 키 (BYOK)");
    if (!value) {
      return;
    }
    config.setApiKey(value);
    win?.webContents.send("say", {
      emotion: "joy",
      line: "오! 이제 더 잘 할 수 있어!",
    });
  }

  async function writeDiary(announce = false): Promise<string | undefined> {
    try {
      const snapshot = { ...day, learned: dedupe(day.learned) };
      day = freshDay();
      const p = await diary.write(snapshot);
      await ending.incDiary();
      if (announce) {
        win?.webContents.send("say", {
          emotion: "moved",
          line: "오늘 일기 다 썼어. 사각사각.",
        });
      }
      return p;
    } catch (err) {
      dialog.showErrorBox("꼬질룡", `일기 실패: ${err}`);
      return undefined;
    }
  }

  async function generatePoses(): Promise<void> {
    if (!(await config.hasApiKey())) {
      dialog.showMessageBox({ message: "먼저 Gemini API 키부터 입력해줘." });
      return;
    }
    win?.webContents.send("say", {
      emotion: "focus",
      line: "포즈 그리는 중… 조금만 기다려…",
    });
    try {
      await poses.generateAll();
      void sendCurrent(); // 새 PNG로 갱신.
      win?.webContents.send("say", {
        emotion: "moved",
        line: "이게… 진짜 내 모습이야. 어때?",
      });
    } catch (err) {
      dialog.showErrorBox("꼬질룡", `포즈 생성 실패: ${err}`);
    }
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
  const w = 180;
  const h = 200;
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

app.on("window-all-closed", () => {
  // 캐릭터 창이 닫히면 종료 (모달 프롬프트만 닫힌 경우는 win이 살아있음).
  if (!win) {
    app.quit();
  }
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


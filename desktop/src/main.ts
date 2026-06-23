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
import { Heart, HeartState } from "./core/heart/stateMachine";
import { DiaryWriter } from "./core/diary/writer";
import { PoseStudio } from "./core/face/poseStudio";
import { EMOTION_POSE, getPose, POSES } from "./core/face/poses";
import { Presence } from "./presence";
import { EventServer } from "./server";
import { EasterEggs } from "./easterEggs";
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

  // 렌더러로 한 마디 시키기 (이스터에그/엔딩/메뉴 공용).
  function say(pose: Emotion, line: string): void {
    win?.webContents.send("say", { emotion: pose, line });
  }
  const eggs = new EasterEggs(say);
  eggs.start();

  // 감정 변화를 렌더러로 (포즈 PNG가 있으면 그걸로, 없으면 SVG 폴백).
  async function pushState(state: HeartState): Promise<void> {
    if (!win) {
      return;
    }
    const poseId = EMOTION_POSE[state.emotion];
    const frames = poses.hasPose(poseId)
      ? await poses.frameDataUrls(poseId)
      : [];
    win.webContents.send("state", {
      ...state,
      poseFrames: frames,
      poseFrameMs: getPose(poseId)?.frameMs ?? 0,
    });
  }
  heart.onChange((s) => void pushState(s));

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
        if (busy) {
          return; // 한 번에 하나만.
        }
        busy = true;
        try {
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
  if (process.env.KKOJI_FORCE_ENDING !== "1") {
    createWindow(); // 데모 엔딩 녹화 땐 캐릭터 창 생략(깔끔한 단일 화면).
  }

  // 시간/방치 맥락 틱.
  const tickTimer = setInterval(() => heart.tick(), 60_000);

  // ── IPC ────────────────────────────────────────────────
  ipcMain.on("pet", () => {
    heart.touch();
    win?.webContents.send("say", {
      emotion: "joy",
      line: "에헤헤… 또 쓰다듬어줘…",
    });
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
              ? `아는 개념 ${memory.size}개. 제일 먼저 배운 건 "${o.concept}" (${o.firstSeen}).`
              : "아직 아무것도 몰라… 너 코딩하는 거 보여줘.",
          });
        },
      },
      { type: "separator" },
      { label: "엔딩 미리보기 (개발용)", click: () => void playEnding() },
      { label: "포즈 다시 그리기 (개발용)", click: () => void generatePoses() },
      { label: "Gemini API 키 입력…", click: () => void askKey() },
      {
        label: "설정 폴더 열기",
        click: () => shell.showItemInFolder(config.settingsFile),
      },
      { type: "separator" },
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
      void pushState(heart.current()); // 새 PNG로 갱신.
      win?.webContents.send("say", {
        emotion: "moved",
        line: "이게… 진짜 내 모습이야. 어때?",
      });
    } catch (err) {
      dialog.showErrorBox("꼬질룡", `포즈 생성 실패: ${err}`);
    }
  }

  // ── 엔딩 시퀀스 (기획서 §17) ─────────────────────────────
  async function playEnding(): Promise<void> {
    const ids: Emotion[] = ["sleepy", "moved", "calm", "joy"];
    const images: Record<string, string> = {};
    for (const id of ids) {
      const frames = await poses.frameDataUrls(id);
      if (frames[0]) {
        images[id] = frames[0];
      }
    }
    const petFrames = await poses.frameDataUrls("pet").catch(() => []);
    if (petFrames[0]) {
      images.pet = petFrames[0];
    }

    const { workArea } = screen.getPrimaryDisplay();
    const W = 760;
    const H = 520;
    const e = new BrowserWindow({
      width: W,
      height: H,
      x: workArea.x + Math.round((workArea.width - W) / 2),
      y: workArea.y + Math.round((workArea.height - H) / 2),
      frame: false,
      resizable: false,
      backgroundColor: "#0c0c0d",
      title: "꼬질룡",
      webPreferences: {
        preload: PRELOAD,
        contextIsolation: true,
        nodeIntegration: false,
      },
    });
    e.setMenuBarVisibility(false);
    e.loadFile(path.join(RENDERER, "ending.html"));
    e.webContents.once("did-finish-load", () =>
      e.webContents.send("ending:init", { images })
    );

    await new Promise<void>((resolve) => {
      const onDone = () => {
        ipcMain.removeListener("ending:done", onDone);
        resolve();
      };
      ipcMain.on("ending:done", onDone);
      e.on("closed", () => {
        ipcMain.removeListener("ending:done", onDone);
        resolve();
      });
    });
    if (!e.isDestroyed()) {
      e.close();
    }
  }

  // 데모/검증용: 환경변수로 부팅하자마자 엔딩 재생.
  if (process.env.KKOJI_FORCE_ENDING === "1") {
    void playEnding().then(() => {
      if (process.env.KKOJI_E2E === "1") {
        app.exit(0);
      }
    });
  }

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

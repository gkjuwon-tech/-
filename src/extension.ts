import * as vscode from "vscode";
import { Config } from "./config";
import { GeminiClient } from "./gemini/client";
import { MemoryStore } from "./rag/memory";
import { Evaluator } from "./brain/evaluator";
import { Heart } from "./heart/stateMachine";
import { Eye } from "./eye/observer";
import { DiaryWriter } from "./diary/writer";
import { CharacterView } from "./face/characterView";
import { EasterEggs } from "./easterEggs";

/**
 * 꼬질룡 부화. 모든 부품을 여기서 한 번 조립한다.
 *
 *   The Eye(관찰) → The Brain(평가+RAG) → The Heart(감정) → 얼굴(webview)
 *                                         The Pen(일기) ↘ The Memory(RAG)
 */
let eye: Eye | undefined;
let diary: DiaryWriter | undefined;
let tickTimer: NodeJS.Timeout | undefined;

export async function activate(context: vscode.ExtensionContext) {
  const config = new Config(context);
  const gemini = new GeminiClient(config);
  const memory = new MemoryStore(context, gemini);
  await memory.load();

  const heart = new Heart();
  const evaluator = new Evaluator(gemini, memory);
  eye = new Eye(evaluator, heart, config);
  diary = new DiaryWriter(gemini, memory, config);

  const view = new CharacterView(context.extensionUri, heart, config);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(CharacterView.viewId, view, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  // 재설치/복귀 인사 (기획서 §10 EE-18)
  if (memory.size > 0) {
    heart.moved("…일기 읽었어. 우리 예전에 같이 공부했더라. 돌아와서 다행이야.");
  }

  eye.start();

  const eggs = new EasterEggs(view);
  eggs.start();
  context.subscriptions.push(eggs);

  // 시간/방치 맥락 틱 (졸림/삐짐)
  tickTimer = setInterval(() => heart.tick(), 60_000);

  // ── 커맨드 ───────────────────────────────────────────────
  context.subscriptions.push(
    vscode.commands.registerCommand("kkojilryong.setApiKey", async () => {
      const key = await vscode.window.showInputBox({
        title: "Gemini API 키 (BYOK)",
        prompt: "꼬질룡은 네 키로만 움직여. 우리가 보관 안 해. https://aistudio.google.com/apikey",
        password: true,
        ignoreFocusOut: true,
      });
      if (key) {
        await config.setApiKey(key);
        view.say("기쁨", "오! 이제 너 코딩하는 거 볼 수 있어!!");
      }
    }),

    vscode.commands.registerCommand("kkojilryong.openDiary", async () => {
      const uri =
        (await diary!.todayUri()) ??
        (await writeAndGet(eye!, diary!, view));
      if (uri) {
        await vscode.window.showTextDocument(uri);
      } else {
        view.say("calm", "오늘은 아직 일기 안 썼어…");
      }
    }),

    vscode.commands.registerCommand("kkojilryong.writeDiaryNow", async () => {
      const uri = await writeAndGet(eye!, diary!, view);
      if (uri) {
        view.say("감동", "오늘 일기 다 썼어. 사각사각.");
        await vscode.window.showTextDocument(uri);
      }
    }),

    vscode.commands.registerCommand("kkojilryong.showMemory", () => {
      const oldest = memory.oldest();
      const msg = oldest
        ? `꼬질룡이 아는 개념: ${memory.size}개. 제일 먼저 배운 건 "${oldest.concept}" (${oldest.firstSeen}).`
        : "아직 아무것도 몰라… 너 코딩하는 거 보여줘.";
      vscode.window.showInformationMessage(`🦖 ${msg}`);
    }),

    vscode.commands.registerCommand("kkojilryong.pet", () => {
      heart.touch();
      view.say("기쁨", "에헤헤… 또 쓰다듬어줘…");
    })
  );

  // 키 없으면 부드럽게 안내 (닦달 금지)
  if (!(await config.hasApiKey())) {
    const pick = await vscode.window.showInformationMessage(
      "🦖 꼬질룡이 깨어났어! 같이 있으려면 Gemini API 키가 필요해 (네 키, 네 비용 — BYOK).",
      "키 등록하기"
    );
    if (pick) {
      await vscode.commands.executeCommand("kkojilryong.setApiKey");
    }
  }
}

/** 세션 종료 = 하루 끝. 그날 본 걸로 일기를 남기고 RAG에 새긴다. */
export async function deactivate() {
  if (tickTimer) {
    clearInterval(tickTimer);
  }
  if (eye && diary) {
    const day = eye.takeDay();
    if (day.learned.length || day.moments.length) {
      try {
        await diary.write(day);
      } catch {
        // 종료 중 실패는 조용히. 다음에 또 쓴다.
      }
    }
  }
}

async function writeAndGet(
  eye: Eye,
  diary: DiaryWriter,
  view: CharacterView
): Promise<vscode.Uri | undefined> {
  try {
    const day = eye.takeDay();
    return await diary.write(day);
  } catch (err) {
    if (String(err).includes("NO_API_KEY")) {
      view.say("calm", "일기 쓰려면 API 키부터 등록해줘…");
    } else {
      vscode.window.showErrorMessage(`꼬질룡 일기 실패: ${err}`);
    }
    return undefined;
  }
}

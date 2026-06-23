import * as vscode from "vscode";
import { Evaluator } from "../brain/evaluator";
import { Heart } from "../heart/stateMachine";
import { Config } from "../config";
import { DayLog, Emotion } from "../types";

/**
 * The Eye. 주인이 무엇을 치는지 어깨너머로 본다. (기획서 §6 기능1)
 *
 * 부담 안 주게 디바운스: 매 키 입력마다 평가하지 않고, 타이핑이 멈춘 뒤
 * 한 번 슥 본다. 평가는 비동기 백그라운드 — 메인 스레드 안 막는다.
 */
export class Eye implements vscode.Disposable {
  private timer?: NodeJS.Timeout;
  private disposables: vscode.Disposable[] = [];
  private busy = false;

  /** 오늘 하루 누적. 일기 작성기로 흘러간다. */
  private day: DayLog = freshDay();

  constructor(
    private readonly evaluator: Evaluator,
    private readonly heart: Heart,
    private readonly config: Config
  ) {}

  start(): void {
    // 타이핑 멈춤 → 디바운스 평가.
    this.disposables.push(
      vscode.workspace.onDidChangeTextDocument((e) => {
        if (!isCode(e.document)) {
          return;
        }
        this.heart.touch();
        this.schedule(e.document);
      })
    );

    // 저장 → 즉시 평가.
    this.disposables.push(
      vscode.workspace.onDidSaveTextDocument((doc) => {
        if (isCode(doc)) {
          this.evaluate(doc);
        }
      })
    );
  }

  private schedule(doc: vscode.TextDocument): void {
    if (this.timer) {
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(
      () => this.evaluate(doc),
      this.config.debounceMs
    );
  }

  private async evaluate(doc: vscode.TextDocument): Promise<void> {
    if (this.busy) {
      return; // 한 번에 하나만. 큐 안 쌓는다.
    }
    this.busy = true;
    try {
      const ev = await this.evaluator.evaluate(doc.getText(), doc.languageId);
      this.heart.applyEvaluation(ev);

      // 하루 누적 갱신.
      this.day.learned.push(...ev.newConcepts);
      this.day.peakEmotion = strongest(this.day.peakEmotion, emotionOf(ev.verdict));
      if (ev.verdict === "rage") {
        this.day.moments.push(`코드 보고 빡쳤다: ${ev.line}`);
      } else if (ev.verdict === "dance") {
        this.day.moments.push(`주인 코드 멋져서 춤췄다.`);
      }
    } catch (err) {
      // 키 없음 등은 조용히 넘긴다. 꼬질룡은 주인을 닦달하지 않는다.
      if (String(err).includes("NO_API_KEY")) {
        return;
      }
      console.error("[꼬질룡] 평가 실패:", err);
    } finally {
      this.busy = false;
    }
  }

  /** 일기 작성용으로 오늘 누적을 꺼내고, 하루를 새로 시작한다. */
  takeDay(): DayLog {
    const snapshot = { ...this.day, learned: dedupe(this.day.learned) };
    this.day = freshDay();
    return snapshot;
  }

  dispose(): void {
    if (this.timer) {
      clearTimeout(this.timer);
    }
    this.disposables.forEach((d) => d.dispose());
  }
}

function freshDay(): DayLog {
  return {
    date: new Date().toISOString().slice(0, 10),
    learned: [],
    moments: [],
    peakEmotion: "calm",
  };
}

function isCode(doc: vscode.TextDocument): boolean {
  return (
    doc.uri.scheme === "file" &&
    doc.languageId !== "plaintext" &&
    doc.languageId !== "log"
  );
}

function emotionOf(v: "dance" | "calm" | "rage"): Emotion {
  return v === "dance" ? "joy" : v === "rage" ? "rage" : "calm";
}

/** 더 "강한"(일기에 남길 가치 있는) 감정을 고른다. */
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

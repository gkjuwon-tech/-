import { Emotion, Evaluation, RageLevel } from "../types";

export interface HeartState {
  emotion: Emotion;
  rageLevel: RageLevel;
  line: string;
}

type Listener = (state: HeartState) => void;

/**
 * The Heart. 입력(평가/시간/방치/자리비움)을 감정으로 전이시킨다.
 *
 * 데스크탑 본체에서 가장 중요한 새 입력은 "present"다:
 *   VS Code(눈)의 하트비트가 끊기면 → 주인이 딴 거 하는 중 → 꼬질룡은 잔다.
 *   하트비트가 돌아오면 → 깬다.
 */
export class Heart {
  private state: HeartState = {
    emotion: "sleepy",
    rageLevel: 0,
    line: "쿨… 쿨…",
  };
  private listeners: Listener[] = [];
  private lastInteraction = Date.now();
  /** VS Code가 활성(코딩 중)인가. false면 잔다. */
  private present = false;

  onChange(fn: Listener): void {
    this.listeners.push(fn);
  }

  current(): HeartState {
    return this.state;
  }

  isPresent(): boolean {
    return this.present;
  }

  touch(): void {
    this.lastInteraction = Date.now();
  }

  /**
   * 주인이 VS Code에 있는지(코딩 중인지) 갱신. 자는 모션의 핵심.
   * 끊기면 잠들고, 돌아오면 깬다.
   */
  setPresent(present: boolean): void {
    if (present === this.present) {
      return;
    }
    this.present = present;
    if (present) {
      this.touch();
      this.set({ emotion: "calm", rageLevel: 0, line: "왔구나! 같이 하자." });
    } else {
      this.set({
        emotion: "sleepy",
        rageLevel: 0,
        line: "다른 거 하는구나… 나 잠깐 잘게… 쿨…",
      });
    }
  }

  applyEvaluation(ev: Evaluation): void {
    this.present = true;
    this.touch();
    // Gemini가 고른 포즈를 그대로 감정으로 쓴다. 단, 평온인데 처음 보는 게
    // 많으면 집중(노려보기)으로 살짝 보정.
    const emotion: Emotion =
      ev.pose === "calm" && ev.newConcepts.length > 0 ? "focus" : ev.pose;
    this.set({ emotion, rageLevel: ev.rageLevel, line: ev.line });
  }

  moved(line: string): void {
    this.touch();
    this.set({ emotion: "moved", rageLevel: 0, line });
  }

  worry(line: string): void {
    this.set({ emotion: "worry", rageLevel: 0, line });
  }

  /** 주기적 틱. 자는 중엔 깨우지 않는다. 깨어있을 때만 시간/방치 맥락 적용. */
  tick(now: number = Date.now()): void {
    if (!this.present) {
      return; // 자는 중. 냅둔다.
    }
    const idleMs = now - this.lastInteraction;
    const hour = new Date(now).getHours();

    if (idleMs > 1000 * 60 * 30) {
      if (this.state.emotion !== "sulk") {
        this.set({ emotion: "sulk", rageLevel: 0, line: "…" });
      }
      return;
    }
    if (
      hour >= 0 &&
      hour < 5 &&
      (this.state.emotion === "calm" || this.state.emotion === "focus")
    ) {
      this.set({
        emotion: "sleepy",
        rageLevel: 0,
        line: "하암… 너 코딩하니까 나도 깨어있을게…",
      });
    }
  }

  private set(next: HeartState): void {
    this.state = next;
    for (const fn of this.listeners) {
      fn(next);
    }
  }
}

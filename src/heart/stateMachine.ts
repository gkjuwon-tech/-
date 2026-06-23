import { Emotion, Evaluation, RageLevel } from "../types";

export interface HeartState {
  emotion: Emotion;
  rageLevel: RageLevel;
  /** 화면에 표시할 마지막 한 마디. */
  line: string;
}

type Listener = (state: HeartState) => void;

/**
 * The Heart. 입력(평가/시간/방치)을 감정 상태로 전이시킨다.
 *
 * 평가 결과만으로 감정이 정해지진 않는다. 새벽이면 졸리고, 오래 방치되면
 * 삐지고, 에러가 반복되면 걱정한다. 이 맥락 레이어가 캐릭터를 살아있게 한다.
 */
export class Heart {
  private state: HeartState = {
    emotion: "calm",
    rageLevel: 0,
    line: "...오늘도 코딩하네?",
  };
  private listeners: Listener[] = [];
  private lastInteraction = Date.now();

  onChange(fn: Listener): void {
    this.listeners.push(fn);
  }

  current(): HeartState {
    return this.state;
  }

  /** 사용자가 코드를 치거나 쓰다듬으면 호출. 방치 타이머 리셋. */
  touch(): void {
    this.lastInteraction = Date.now();
  }

  /** 평가 결과를 감정으로 변환. 단, 시간 맥락이 우선할 수 있다. */
  applyEvaluation(ev: Evaluation): void {
    this.touch();
    const emotion: Emotion =
      ev.verdict === "dance" ? "joy" : ev.verdict === "rage" ? "rage" : "calm";
    // 처음 보는 개념이 많으면 평온 대신 집중(노려보기)으로.
    const adjusted: Emotion =
      emotion === "calm" && ev.newConcepts.length > 0 ? "focus" : emotion;
    this.set({ emotion: adjusted, rageLevel: ev.rageLevel, line: ev.line });
  }

  /** 감동의 순간 (스트릭, 본명 공개, 큰 리팩토링 등)을 직접 주입. */
  moved(line: string): void {
    this.touch();
    this.set({ emotion: "moved", rageLevel: 0, line });
  }

  /** 걱정 (같은 에러 반복 등). */
  worry(line: string): void {
    this.set({ emotion: "worry", rageLevel: 0, line });
  }

  /**
   * 주기적 틱. 시간/방치 맥락으로 감정을 갱신한다.
   * @param now 테스트 주입용. 기본 Date.now().
   */
  tick(now: number = Date.now()): void {
    const idleMs = now - this.lastInteraction;
    const hour = new Date(now).getHours();

    // 오래 방치 → 삐짐. (기획서: 너무 오래 두면 삐진다)
    if (idleMs > 1000 * 60 * 30) {
      if (this.state.emotion !== "sulk") {
        this.set({
          emotion: "sulk",
          rageLevel: 0,
          line: "…",
        });
      }
      return;
    }

    // 새벽(0~5시)이고 평온/집중이면 졸림으로.
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

import { Emotion } from "./core/types";

type Say = (pose: Emotion, line: string) => void;

/**
 * 이스터에그 (기획서 §10). 웃긴 거 반, 감동 반.
 *
 * 데스크탑이 받는 코드 텍스트 + 시계로 감지 가능한 것들을 구현한다.
 * 같은 에그는 쿨다운을 둬서 도배하지 않는다.
 */
export class EasterEggs {
  private cooldown = new Map<string, number>();
  private clockTimer?: NodeJS.Timeout;

  constructor(private readonly say: Say) {}

  start(): void {
    // EE-04: 새벽 4시 44분
    this.clockTimer = setInterval(() => {
      const d = new Date();
      if (d.getHours() === 4 && d.getMinutes() === 44) {
        this.fire("0444", "worry", "…농담이야. 근데 너 진짜 안 자?", 60 * 60_000);
      }
    }, 30_000);
  }

  /** 코드 한 조각을 보고 숨은 신호를 찾는다. */
  onCode(code: string): void {
    // EE-01: TODO 너무 많음
    const todos = (code.match(/TODO/gi) || []).length;
    if (todos >= 8) {
      this.fire("todo", "worry", "이거… 우리 '나중에'가 너무 많아졌어…");
    }
    // EE-02: 디버그 막 찍는 로그
    if (/console\.log\((['"`])(여기|여기여기|\d{2,})\1\)/.test(code) ||
        /print\((['"`])(여기|디버그|\d{2,})\1\)/.test(code)) {
      this.fire("debuglog", "focus", "여기? 여기가 어디야?");
    }
    // EE-06: 사과 주석
    if (/(?:\/\/|#)\s*(미안|죄송|sorry)/i.test(code)) {
      this.fire("sorry", "moved", "괜찮아. 다들 그래.");
    }
    // EE-07: temp 형제들
    if (/\btemp2\b/.test(code) && /\btemp(3|Final|FinalReal)?\b/.test(code)) {
      this.fire("temp", "calm", "얘네 다 형제야?");
    }
    // EE-10: 무한루프
    if (/while\s*\(\s*(true|1)\s*\)/.test(code)) {
      this.fire("loop", "sleepy", "우리… 같이… 어지러…");
    }
  }

  stop(): void {
    if (this.clockTimer) {
      clearInterval(this.clockTimer);
    }
  }

  private fire(
    id: string,
    pose: Emotion,
    line: string,
    cooldownMs = 5 * 60_000
  ): void {
    const now = Date.now();
    if ((this.cooldown.get(id) ?? 0) > now) {
      return;
    }
    this.cooldown.set(id, now + cooldownMs);
    this.say(pose, line);
  }
}

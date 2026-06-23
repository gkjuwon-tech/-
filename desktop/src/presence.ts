import { Heart } from "./core/heart/stateMachine";

/**
 * 자리 감지 = "자는 모션"의 핵심.
 *
 * VS Code(눈)는 포커스 중일 때만 하트비트를 보낸다. 주인이 브라우저 등 다른
 * 앱으로 옮겨가면 하트비트가 끊기고, timeoutMs가 지나면 꼬질룡은 잠든다.
 * 다시 VS Code로 돌아와 하트비트가 오면 깬다.
 */
export class Presence {
  private last = 0;
  private timer?: NodeJS.Timeout;

  constructor(
    private readonly heart: Heart,
    private readonly timeoutMs = 12_000
  ) {}

  start(): void {
    this.timer = setInterval(() => this.check(), 3_000);
  }

  /** 하트비트/코드 이벤트 수신 → 깨어있음. */
  beat(): void {
    this.last = Date.now();
    this.heart.setPresent(true);
  }

  /** VS Code 포커스 잃음 → 즉시 잠들 준비. */
  blur(): void {
    this.last = 0;
    this.heart.setPresent(false);
  }

  private check(): void {
    if (this.heart.isPresent() && Date.now() - this.last > this.timeoutMs) {
      this.heart.setPresent(false); // 하트비트 끊김 → 잠.
    }
  }

  stop(): void {
    if (this.timer) {
      clearInterval(this.timer);
    }
  }
}

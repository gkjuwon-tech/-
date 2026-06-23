import * as fs from "fs/promises";
import * as path from "path";
import {
  GEUNAL_TXT,
  MY_FIRST_PROGRAM,
  ANNYEONG_TXT,
  FIRST_DAY_TXT,
  WAITING_TXT,
  RETURN_BEATS,
  TRUE_ENDING_BEATS,
} from "./core/endingTexts";

export interface EndingDeps {
  stateFile: string;
  diaryFolder: string;
  projectFolder: string;
  /** 위젯에서 꼬질룡을 숨기거나(=사라짐) 다시 보이게. */
  setGone: (gone: boolean) => void;
  /** 위젯 말풍선으로 한 마디. */
  say: (pose: string, line: string) => void;
  /** 파일을 기본 앱으로 열기. */
  openPath: (p: string) => void;
}

interface EndingState {
  phase: "normal" | "gone" | "returned" | "off";
  lastSeen: string; // YYYY-MM-DD
  goneAt?: number;
  activeDays: number;
  diaryCount: number;
}

/**
 * 엔딩 아크 (기획서 §17)를 "진짜 위젯 동작 + 진짜 파일"로 구현한다.
 *
 * 절대 광고하지 않는다. 팝업 없다. 꼬질룡은 죽지 않는다 — 그냥 말수가 줄고,
 * 어느 날 조용히 사라졌다가(구석에 [ … ]만 남기고 그날.txt를 남긴다),
 * 며칠 뒤 네가 다시 코딩하면 조용히 돌아온다. "원래 여기 있었어."
 * 되돌릴 수 있다. 모든 작별은 "또 불러줘"로 끝난다.
 */
export class EndingDirector {
  private state: EndingState = {
    phase: "normal",
    lastSeen: today(),
    activeDays: 0,
    diaryCount: 0,
  };
  private hiddenGreetArmed = false;
  private returning = false;

  /** 데모/개발 모드: 조건을 빨리 채우고 대기 시간을 짧게. */
  constructor(
    private readonly deps: EndingDeps,
    private readonly demo = false
  ) {}

  async init(): Promise<void> {
    try {
      this.state = JSON.parse(await fs.readFile(this.deps.stateFile, "utf8"));
    } catch {
      /* 첫 실행 */
    }

    const last = this.state.lastSeen;
    if (last !== today()) {
      this.state.activeDays += 1;
      this.state.lastSeen = today();
    }

    // §17.7 히든: 사라진 채로 오래 안 왔다 돌아옴 → 마지막 접속일에 멈춘 일기.
    if (
      (this.state.phase === "gone" || this.state.phase === "returned") &&
      daysBetween(last, today()) >= (this.demo ? 0 : 14)
    ) {
      await this.safeWrite(`${last}.txt`, WAITING_TXT);
      this.hiddenGreetArmed = true;
    }

    await this.persist();

    if (this.state.phase === "normal" && this.shouldEnter()) {
      // 조건이 다 차고 "며칠이 지난 어느 날" 조용히 사라진다.
      setTimeout(() => void this.disappear(), this.demo ? 3000 : 60_000);
    } else if (this.state.phase === "gone") {
      // 여전히 사라진 상태로 부팅 → 구석에 [ … ]만.
      this.deps.setGone(true);
    }
  }

  isGone(): boolean {
    return this.state.phase === "gone";
  }

  /** 엔딩 이후엔 거의 말하지 않는다 (침묵 곡선의 끝). */
  isQuiet(): boolean {
    return this.state.phase === "returned";
  }

  async incDiary(): Promise<void> {
    this.state.diaryCount += 1;
    await this.persist();
  }

  /** 코드/빌드 이벤트가 올 때 호출. 사라진 상태면 "며칠 뒤" 복귀를 깨운다. */
  onActivity(): void {
    if (this.hiddenGreetArmed) {
      this.hiddenGreetArmed = false;
      this.deps.setGone(false);
      this.beat("calm", "헬로.", 0);
      this.beat("moved", "오랜만.", 1800);
      return;
    }
    if (this.state.phase === "gone" && !this.returning) {
      const waited = Date.now() - (this.state.goneAt ?? 0);
      if (waited >= (this.demo ? 5000 : 1000 * 60 * 60 * 24)) {
        void this.comeBack();
      }
    }
  }

  /** 구석의 [ … ]를 누르면 그날.txt를 연다. */
  onGoneClick(): void {
    this.deps.openPath(path.join(this.deps.diaryFolder, "그날.txt"));
  }

  /** §17.8 — [꼬질룡 끄기]. 작별 편지 + 진짜 첫 일기를 남기고 끈다. */
  async turnOff(): Promise<void> {
    await this.safeWrite("안녕.txt", ANNYEONG_TXT);
    await this.safeWrite("first_day.txt", FIRST_DAY_TXT);
    this.state.phase = "off";
    await this.persist();
  }

  // ── 내부 ────────────────────────────────────────────────

  private shouldEnter(): boolean {
    if (this.demo) {
      return true;
    }
    // §17.2 — 의도적으로 빡세게. (근사치: 실사용 1년 + 일기 1000개)
    return this.state.activeDays >= 365 && this.state.diaryCount >= 1000;
  }

  private async disappear(): Promise<void> {
    this.state.phase = "gone";
    this.state.goneAt = Date.now();
    await this.persist();
    await this.safeWrite("그날.txt", GEUNAL_TXT);
    this.deps.setGone(true); // 위젯에서 사라지고 구석에 [ … ]만. (말 없음)
  }

  private async comeBack(): Promise<void> {
    this.returning = true;
    this.deps.setGone(false); // 조용히 다시 나타난다.
    let t = 0;
    // §17.4 "원래 여기 있었어"
    for (const b of RETURN_BEATS) {
      this.beat(b.pose, b.line, t);
      t += b.gapMs;
    }
    // §17.5 진엔딩 — 처음으로 부탁한다 + 자기 프로그램을 만든다.
    for (const b of TRUE_ENDING_BEATS) {
      t += b.gapMs;
      this.beat(b.pose, b.line, t);
    }
    t += 1200;
    setTimeout(() => void this.makeFirstProgram(), t);

    this.state.phase = "returned";
    await this.persist();
    setTimeout(() => (this.returning = false), t + 4000);
  }

  private async makeFirstProgram(): Promise<void> {
    const file = path.join(this.deps.projectFolder, "hello.py");
    try {
      await fs.mkdir(this.deps.projectFolder, { recursive: true });
      await fs.writeFile(file, MY_FIRST_PROGRAM, "utf8");
      this.deps.openPath(file);
    } catch {
      /* 조용히 */
    }
  }

  private beat(pose: string, line: string, delay: number): void {
    setTimeout(() => this.deps.say(pose, line), delay);
  }

  private async safeWrite(name: string, content: string): Promise<void> {
    try {
      await fs.mkdir(this.deps.diaryFolder, { recursive: true });
      await fs.writeFile(
        path.join(this.deps.diaryFolder, name),
        content,
        "utf8"
      );
    } catch {
      /* 조용히 */
    }
  }

  private async persist(): Promise<void> {
    try {
      await fs.mkdir(path.dirname(this.deps.stateFile), { recursive: true });
      await fs.writeFile(
        this.deps.stateFile,
        JSON.stringify(this.state, null, 2),
        "utf8"
      );
    } catch {
      /* 조용히 */
    }
  }
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

function daysBetween(a: string, b: string): number {
  const da = Date.parse(a + "T00:00:00Z");
  const db = Date.parse(b + "T00:00:00Z");
  if (isNaN(da) || isNaN(db)) {
    return 0;
  }
  return Math.round((db - da) / 86400000);
}

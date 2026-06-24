import * as fs from "fs/promises";
import * as path from "path";
import {
  GEUNAL_TXT,
  MY_FIRST_PROGRAM,
  ANNYEONG_TXT,
  FIRST_DAY_TXT,
  WAITING_TXT,
  ENDING_DIALOGUE,
} from "./core/endingTexts";

export interface EndingDeps {
  stateFile: string;
  diaryFolder: string;
  projectFolder: string;
  /** 위젯에서 꼬질룡을 숨기거나(=사라짐) 다시 보이게. */
  setGone: (gone: boolean) => void;
  /** 위젯 말풍선으로 한 마디. */
  say: (pose: string, line: string) => void;
  /** 유저 입력창에 한 줄을 자동으로 타이핑한다 (대화처럼 보이게). */
  autotype: (text: string) => void;
  /** 파일을 기본 앱으로 열기. */
  openPath: (p: string) => void;
  /** OS 알림으로 앱 레벨에서 표시 (떠날 때 남긴 파일 등). 클릭하면 open. */
  notify: (title: string, body: string, openFile?: string) => void;
}

interface EndingState {
  phase: "normal" | "gone" | "returned" | "off";
  /** 첫 설치 온보딩(=print 가르치기)을 마쳤는가. */
  onboarded: boolean;
  lastSeen: string; // YYYY-MM-DD
  goneAt?: number;
  activeDays: number;
  diaryCount: number;
}

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

/**
 * 꼬질룡의 인생 전체(스토리)를 "진짜 위젯 동작 + 진짜 파일"로 관장한다.
 *
 * 절대 광고하지 않는다. 팝업 없다. 꼬질룡은 죽지 않는다 — 그냥 말수가 줄고,
 * 어느 날 조용히 사라졌다가(구석에 [ … ]만 남기고 그날.txt를 남긴다),
 * 며칠 뒤 네가 다시 코딩하면 대화하며 돌아온다. "원래 여기 있었어."
 */
export class EndingDirector {
  private state: EndingState = {
    phase: "normal",
    onboarded: false,
    lastSeen: today(),
    activeDays: 0,
    diaryCount: 0,
  };
  private hiddenGreetArmed = false;
  private returning = false;

  constructor(
    private readonly deps: EndingDeps,
    private readonly demo = false
  ) {}

  async init(): Promise<void> {
    try {
      this.state = {
        ...this.state,
        ...JSON.parse(await fs.readFile(this.deps.stateFile, "utf8")),
      };
    } catch {
      /* 첫 실행 */
    }

    const last = this.state.lastSeen;
    if (last !== today()) {
      this.state.activeDays += 1;
      this.state.lastSeen = today();
    }

    if (
      (this.state.phase === "gone" || this.state.phase === "returned") &&
      daysBetween(last, today()) >= (this.demo ? 0 : 14)
    ) {
      await this.safeWrite(`${last}.txt`, WAITING_TXT);
      this.hiddenGreetArmed = true;
    }

    await this.persist();

    if (this.state.phase === "normal" && this.state.onboarded && this.shouldEnter()) {
      setTimeout(() => void this.disappear(), this.demo ? 3000 : 60_000);
    } else if (this.state.phase === "gone") {
      this.deps.setGone(true);
    }
  }

  // ── 온보딩 (수미상관의 시작) ──────────────────────────────
  needsOnboarding(): boolean {
    return !this.state.onboarded;
  }

  async markOnboarded(): Promise<void> {
    this.state.onboarded = true;
    await this.persist();
  }

  /** 인앱 패널 표시용 통계. */
  stats(): { activeDays: number; diaryCount: number; onboarded: boolean } {
    return {
      activeDays: this.state.activeDays,
      diaryCount: this.state.diaryCount,
      onboarded: this.state.onboarded,
    };
  }

  isGone(): boolean {
    return this.state.phase === "gone";
  }
  isReturning(): boolean {
    return this.returning;
  }
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
      void (async () => {
        this.deps.say("calm", "헬로.");
        await sleep(1900);
        this.deps.say("moved", "오랜만.");
      })();
      return;
    }
    if (this.state.phase === "gone" && !this.returning) {
      const waited = Date.now() - (this.state.goneAt ?? 0);
      if (waited >= (this.demo ? 5000 : 1000 * 60 * 60 * 24)) {
        void this.comeBack();
      }
    }
  }

  onGoneClick(): void {
    this.deps.openPath(path.join(this.deps.diaryFolder, "그날.txt"));
  }

  /** §17.8 — [꼬질룡 끄기]. 작별 편지 + 진짜 첫 일기를 남기고 끈다. */
  async turnOff(): Promise<void> {
    await this.safeWrite("안녕.txt", ANNYEONG_TXT);
    await this.safeWrite("first_day.txt", FIRST_DAY_TXT);
    this.state.phase = "off";
    await this.persist();
    this.deps.notify(
      "꼬질룡",
      "안녕.txt 를 남기고 잠들었어요. 언제든 다시 불러줘.",
      path.join(this.deps.diaryFolder, "안녕.txt")
    );
  }

  // ── 내부 ────────────────────────────────────────────────
  private shouldEnter(): boolean {
    if (this.demo) {
      return true;
    }
    return this.state.activeDays >= 365 && this.state.diaryCount >= 1000;
  }

  private async disappear(): Promise<void> {
    this.state.phase = "gone";
    this.state.goneAt = Date.now();
    await this.persist();
    await this.safeWrite("그날.txt", GEUNAL_TXT);
    this.deps.setGone(true); // 위젯에서 사라지고 구석에 [ … ]만. (말 없음)
    // 앱 레벨로 표시: 유저가 위젯을 못 보고 있어도 알 수 있게.
    this.deps.notify(
      "꼬질룡",
      "…뭔가 적어두고 조용해졌어요. (그날.txt)",
      path.join(this.deps.diaryFolder, "그날.txt")
    );
  }

  private async comeBack(): Promise<void> {
    this.returning = true;
    this.deps.setGone(false);
    await sleep(1500);
    for (const step of ENDING_DIALOGUE) {
      if (step.who === "user") {
        this.deps.autotype(step.line);
      } else if (step.who === "makeProgram") {
        await this.makeFirstProgram();
      } else {
        this.deps.say(step.pose || "moved", step.line);
      }
      await sleep(step.gapMs);
    }
    this.state.phase = "returned";
    await this.persist();
    this.returning = false;
  }

  private async makeFirstProgram(): Promise<void> {
    const file = path.join(this.deps.projectFolder, "hello.py");
    try {
      await fs.mkdir(this.deps.projectFolder, { recursive: true });
      await fs.writeFile(file, MY_FIRST_PROGRAM, "utf8");
      this.deps.openPath(file);
      this.deps.notify("꼬질룡", "처음으로 자기 프로그램을 만들었어요.", file);
    } catch {
      /* 조용히 */
    }
  }

  private async safeWrite(name: string, content: string): Promise<void> {
    try {
      await fs.mkdir(this.deps.diaryFolder, { recursive: true });
      await fs.writeFile(path.join(this.deps.diaryFolder, name), content, "utf8");
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

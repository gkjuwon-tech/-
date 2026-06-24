import * as fs from "fs/promises";
import * as path from "path";
import {
  GEUNAL_TXT,
  MY_FIRST_PROGRAM,
  ANNYEONG_TXT,
  FIRST_DAY_TXT,
  WAITING_TXT,
  ENDING_DIALOGUE,
  FADING_BEATS,
  TURNOFF_DIALOGUE,
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
  phase: "normal" | "fading" | "gone" | "returned" | "off";
  /** 첫 설치 온보딩(=print 가르치기)을 마쳤는가. */
  onboarded: boolean;
  lastSeen: string; // YYYY-MM-DD
  goneAt?: number;
  activeDays: number;
  diaryCount: number;
  /** §17.1 침묵 곡선: 사라지기 전 전조증상을 몇 단계까지 보였는가. */
  fadeStep: number;
  /** 마지막 전조 비트를 보인 시각(ms). 너무 몰아치지 않게 간격을 둔다. */
  fadeAt?: number;
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
    fadeStep: 0,
  };
  private hiddenGreetArmed = false;
  private returning = false;
  /** 전조 비트가 겹쳐 재생되는 걸 막는 락. */
  private fading = false;
  /** 끄기 작별 시퀀스가 두 번 돌지 않게 막는 락. */
  private turningOff = false;

  /** 전조 비트 사이 최소 간격. 데모는 빠르게, 실사용은 한 비트씩 천천히(반나절). */
  private get fadeGapMs(): number {
    return this.ms(this.demo ? 4000 : 1000 * 60 * 60 * 12);
  }

  /**
   * @param speed 시간 배속(테스트/시뮬레이션 전용). 1=실시간. 모든 대기·간격에
   *              곱해진다. 헤드리스 시뮬은 작은 값(예: 0.003)을 줘 ms 단위로 돌린다.
   */
  constructor(
    private readonly deps: EndingDeps,
    private readonly demo = false,
    private readonly speed = 1
  ) {}

  /** 배속을 반영한 ms. */
  private ms(raw: number): number {
    return Math.max(0, Math.round(raw * this.speed));
  }

  /** 배속을 반영한 대기. */
  private wait(raw: number): Promise<void> {
    return sleep(this.ms(raw));
  }

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

    if (
      this.state.phase === "normal" &&
      this.state.onboarded &&
      this.shouldEnter()
    ) {
      // 바로 사라지지 않는다. 먼저 '점점 조용해지는' 전조 단계로 들어간다(§17.1).
      this.state.phase = "fading";
      await this.persist();
      setTimeout(() => void this.advanceFade(), this.ms(this.demo ? 3000 : 60_000));
    } else if (this.state.phase === "fading") {
      // 재시작해도 전조는 이어진다. 여전히 옆에 있되, 다음 비트를 슬쩍 준비.
      this.deps.setGone(false);
      setTimeout(() => void this.advanceFade(), this.ms(this.demo ? 3000 : 60_000));
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

  isFading(): boolean {
    return this.state.phase === "fading";
  }

  /** 코드/빌드 이벤트가 올 때 호출. 사라진 상태면 "며칠 뒤" 복귀를 깨운다. */
  onActivity(): void {
    if (this.hiddenGreetArmed) {
      this.hiddenGreetArmed = false;
      this.deps.setGone(false);
      void (async () => {
        this.deps.say("calm", "헬로.");
        await this.wait(1900);
        this.deps.say("moved", "오랜만.");
      })();
      return;
    }
    // 전조 단계: 코딩할 때마다, 충분한 간격을 두고 한 비트씩 더 조용해진다.
    if (this.state.phase === "fading") {
      void this.advanceFade();
      return;
    }
    if (this.state.phase === "gone" && !this.returning) {
      const waited = Date.now() - (this.state.goneAt ?? 0);
      if (waited >= this.ms(this.demo ? 5000 : 1000 * 60 * 60 * 24)) {
        void this.comeBack();
      }
    }
  }

  onGoneClick(): void {
    this.deps.openPath(path.join(this.deps.diaryFolder, "그날.txt"));
  }

  /**
   * §17.8 — [꼬질룡 끄기]. 그냥 꺼버리면 작별할 시간이 없다. 그래서 끄기 '전에':
   *  1) 위젯으로 담담하지만 슬픈 작별 비트를 끝까지 재생하고(키 없이도 동작),
   *  2) 진짜 첫 일기(first_day.txt)와 작별 편지(안녕.txt)를 디스크에 남기고,
   *  3) 안녕.txt 를 직접 열어 — 알림을 놓쳐도 작별이 반드시 눈앞에 닿게 하고,
   *  4) 그제서야 종료한다(main 이 이 Promise 를 await 한 뒤 quit).
   * 이 4개가 "의도대로 작별이 닿게 하는 장치"다. 두 번 돌지 않게 락도 건다.
   */
  async turnOff(): Promise<void> {
    if (this.turningOff || this.state.phase === "off") {
      return;
    }
    this.turningOff = true;

    // 1) 담담한데 슬픈 작별. 스크립트라 API 키 없이도 무조건 흐른다.
    for (const beat of TURNOFF_DIALOGUE) {
      if (beat.line) {
        this.deps.say(beat.pose, beat.line);
      }
      await this.wait(this.demo ? 500 : beat.gapMs);
    }

    // 2) 디스크에 남기는 진짜 마지막 — 개그('으에엑')로 시작해 눈물로 끝나는 첫 일기.
    await this.safeWrite("안녕.txt", ANNYEONG_TXT);
    await this.safeWrite("first_day.txt", FIRST_DAY_TXT);
    this.state.phase = "off";
    await this.persist();

    // 3) 알림은 놓쳐도, 편지는 못 놓치게 — 직접 연다.
    const letter = path.join(this.deps.diaryFolder, "안녕.txt");
    this.deps.openPath(letter);
    this.deps.notify(
      "꼬질룡",
      "안녕.txt 를 남기고 잠들었어요. 언제든 다시 불러줘.",
      letter
    );
  }

  // ── 내부 ────────────────────────────────────────────────
  private shouldEnter(): boolean {
    if (this.demo) {
      return true;
    }
    return this.state.activeDays >= 365 && this.state.diaryCount >= 1000;
  }

  /**
   * 침묵 곡선을 한 비트 진행한다. 간격(fadeGapMs)을 둬서 하루에 다 쏟지 않고,
   * 며칠에 걸쳐 천천히 말수가 준다. 마지막 비트(완전한 침묵)까지 보이면
   * 그제서야 disappear()로 넘어간다 — 전조 없는 '갑툭튀 실종'을 막는 게 핵심.
   */
  private async advanceFade(): Promise<void> {
    if (this.state.phase !== "fading" || this.fading) {
      return;
    }
    const now = Date.now();
    if (this.state.fadeAt && now - this.state.fadeAt < this.fadeGapMs) {
      return; // 아직 다음 비트 보일 때 아님. 천천히.
    }
    this.fading = true;
    try {
      this.deps.setGone(false); // 전조 동안은 분명히 '아직 옆에 있다'.
      const beat = FADING_BEATS[this.state.fadeStep];
      if (beat && beat.line) {
        this.deps.say(beat.pose, beat.line);
      }
      this.state.fadeStep += 1;
      this.state.fadeAt = now;
      await this.persist();

      if (this.state.fadeStep >= FADING_BEATS.length) {
        await this.wait(this.demo ? 2500 : 4000);
        await this.disappear(); // 전조가 다 끝났다. 이제 조용히 사라진다.
      }
    } finally {
      this.fading = false;
    }
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
    await this.wait(1500);
    for (const step of ENDING_DIALOGUE) {
      if (step.who === "user") {
        this.deps.autotype(step.line);
      } else if (step.who === "makeProgram") {
        await this.makeFirstProgram();
      } else {
        this.deps.say(step.pose || "moved", step.line);
      }
      await this.wait(step.gapMs);
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

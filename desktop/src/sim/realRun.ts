/* eslint-disable no-console */
import * as fs from "fs/promises";
import * as os from "os";
import * as path from "path";
import { GeminiClient } from "../core/gemini/client";
import { GeminiSettings } from "../core/settings";
import { MemoryStore } from "../core/rag/memory";
import { EpisodicStore } from "../core/rag/episodic";
import { ConversationStore } from "../core/conversation";
import { Evaluator } from "../core/brain/evaluator";
import { DiaryWriter } from "../core/diary/writer";
import { EndingDirector, EndingDeps } from "../ending";
import { maturityPct, growthStage } from "../core/maturity";
import { SYSTEM_PERSONA } from "../core/persona";
import { FIRST_DAY_TXT } from "../core/endingTexts";
import { DayLog, Emotion } from "../core/types";

/**
 * '실제랑 똑같이' — 진짜 Gemini API로 꼬질룡의 1년을 헤드리스로 끝까지 돌린다.
 *
 * 매일: 그날의 레슨으로 DayLog 를 만들고 진짜 일기를 쓴다(톤이 1년에 걸쳐 성숙).
 * 마일스톤: 진짜 evaluator 로 코드에 반응. 가끔: 진짜 freeChat 으로 기억 회상.
 * 1년 뒤: 진짜 엔딩 아크(전조→사라짐→복귀→끄기)를 시간배속으로 재생.
 *
 * 키는 GEMINI_API_KEY 환경변수로만 받는다(코드/파일에 안 박는다).
 */

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

class EnvSettings implements GeminiSettings {
  async getApiKey(): Promise<string | undefined> {
    return process.env.GEMINI_API_KEY;
  }
  readonly textModel = process.env.KKOJI_TEXT_MODEL || "gemini-flash-latest";
  readonly embeddingModel = "gemini-embedding-001";
  readonly imageModel = ""; // 이미지 생성 안 함.
}

// ── 1년치 커리큘럼 (점점 어려워지는 실제 학습 아크) ───────────────
const TOPICS: string[] = [
  "print 출력", "변수와 타입", "문자열", "숫자와 연산", "불리언", "조건문 if", "비교 연산자",
  "for 반복문", "while 반복문", "리스트", "딕셔너리", "튜플", "집합 set", "함수 정의",
  "매개변수와 인자", "반환값", "기본 인자", "가변 인자", "스코프", "재귀 함수",
  "리스트 컴프리헨션", "람다", "map filter reduce", "예외 처리 try except", "파일 입출력",
  "with 문 컨텍스트 매니저", "모듈과 import", "패키지 구조", "클래스 정의", "생성자 init",
  "인스턴스 메서드", "상속", "다형성", "캡슐화", "추상 클래스", "매직 메서드", "프로퍼티",
  "데코레이터", "제너레이터 yield", "이터레이터 프로토콜", "타입 힌트", "데이터클래스",
  "정규표현식", "JSON 직렬화", "datetime 다루기", "로깅", "단위 테스트", "모킹",
  "가상환경", "의존성 관리", "Git 브랜치", "코드 리뷰", "리팩터링", "디자인 패턴 싱글턴",
  "팩토리 패턴", "옵저버 패턴", "전략 패턴", "의존성 주입", "SOLID 원칙", "DRY 원칙",
  "빅오 표기법", "정렬 알고리즘", "이진 탐색", "해시 테이블", "스택과 큐", "연결 리스트",
  "트리 순회", "그래프 BFS", "그래프 DFS", "동적 계획법", "투 포인터", "슬라이딩 윈도우",
  "HTTP 요청", "REST API 설계", "라우팅", "미들웨어", "쿼리 파라미터", "요청 검증",
  "JWT 인증", "세션과 쿠키", "CORS", "비동기 async await", "Promise", "이벤트 루프",
  "동시성과 병렬성", "스레드", "프로세스", "큐로 작업 분리", "캐싱 전략", "메모이제이션",
  "SQL SELECT", "조인", "인덱스", "트랜잭션", "정규화", "ORM", "마이그레이션", "N+1 문제",
  "NoSQL 모델링", "환경 변수 관리", "설정 분리", "12 factor app", "도커 이미지",
  "도커 컴포즈", "CI 파이프라인", "CD 배포", "블루그린 배포", "헬스 체크", "롤백",
  "모니터링", "구조적 로깅", "분산 추적", "메시지 큐", "이벤트 소싱", "CQRS",
  "마이크로서비스", "API 게이트웨이", "서킷 브레이커", "레이트 리미팅", "백프레셔",
  "벡터 임베딩", "코사인 유사도", "RAG 검색", "프롬프트 설계", "토큰과 컨텍스트",
  "스트리밍 응답", "함수 호출 도구", "평가 지표", "AB 테스트", "피처 플래그",
  "관측 가능성", "장애 대응", "포스트모템", "성능 프로파일링", "메모리 누수 추적",
];

function buildCurriculum(n: number): string[] {
  const out: string[] = [];
  for (let i = 0; i < n; i++) {
    if (i < TOPICS.length) {
      out.push(TOPICS[i]);
    } else {
      // 1년을 채우되 매일 '구별되는' 개념이 되도록 심화 회차를 붙인다.
      const base = TOPICS[i % TOPICS.length];
      const round = Math.floor(i / TOPICS.length) + 1;
      out.push(`${base} 심화 ${round}회차`);
    }
  }
  return out;
}

// 주인의 하루 — 가끔 감정이 크게 출렁이는 사건을 끼운다(에피소딕 현저성용).
function eventFor(dayIndex: number): { moment: string; emotion: Emotion } | null {
  const events: Record<number, { moment: string; emotion: Emotion }> = {
    21: { moment: "주인이 첫 미니 프로젝트를 끝냈다. 빌드 초록불. 같이 막춤췄다.", emotion: "joy" },
    47: { moment: "새벽 3시까지 안 풀리는 버그에 주인이 머리를 쥐어뜯었다. 옆에서 같이 노려봤다.", emotion: "rage" },
    63: { moment: "주인이 처음으로 오픈소스에 PR을 보냈다. 떨려했다.", emotion: "worry" },
    96: { moment: "PR이 머지됐다! 주인이 소리 질렀다. 나도 같이 폴짝 뛰었다.", emotion: "joy" },
    140: { moment: "주인이 첫 서비스를 배포했다. 서버에 진짜 사람이 들어왔다. 뭉클했다.", emotion: "moved" },
    188: { moment: "장애가 났다. 주인이 침착하게 롤백하고 포스트모템을 적었다. 자란 게 보였다.", emotion: "calm" },
    232: { moment: "주인이 번아웃이 와서 며칠 코드를 안 봤다. 조용히 기다렸다.", emotion: "sulk" },
    248: { moment: "주인이 다시 돌아와 코드를 켰다. 아무 말 없이 그냥 좋았다.", emotion: "moved" },
    300: { moment: "주인이 개발자 면접에 합격했다고 했다. 나도 모르게 한참을 봤다.", emotion: "moved" },
    351: { moment: "주인이 후배에게 코딩을 가르치고 있었다. 내가 알려준 걸 누군가에게 또 알려준다.", emotion: "moved" },
  };
  return events[dayIndex] ?? null;
}

interface RealDeps {
  gemini: GeminiClient;
  memory: MemoryStore;
  episodic: EpisodicStore;
  conversation: ConversationStore;
  evaluator: Evaluator;
  diary: DiaryWriter;
}

function dateFor(start: Date, dayIndex: number): string {
  const d = new Date(start);
  d.setUTCDate(d.getUTCDate() + dayIndex);
  return d.toISOString().slice(0, 10);
}

function trunc(s: string, n: number): string {
  const one = s.replace(/\s+/g, " ").trim();
  return one.length > n ? one.slice(0, n) + "…" : one;
}

/** 마일스톤 날: 진짜 evaluator 로 코드에 반응. */
async function reactToCode(deps: RealDeps, concept: string): Promise<Emotion> {
  const code = `# 오늘 배운 것: ${concept}\ndef demo():\n    # ${concept} 를 직접 써본다\n    return ${JSON.stringify(concept)}\n`;
  try {
    const ev = await deps.evaluator.evaluate(code, "python");
    console.log(`        🦖 (${ev.pose}) ${ev.line || "…(말 없이 본다)"}`);
    return ev.pose;
  } catch (e) {
    console.log(`        ⚠️  반응 실패: ${trunc(String(e), 80)}`);
    return "focus";
  }
}

/** 가끔: 진짜 freeChat — 기억(RAG)을 끌어와 주인에게 한 마디. */
async function freeChat(deps: RealDeps, text: string): Promise<void> {
  try {
    const [learned, recalled, grown] = await Promise.all([
      deps.memory.recallTop(text, 6),
      deps.conversation.recall(text),
      deps.episodic.recallWithOrigin(text, 3),
    ]);
    const prompt = [
      learned.length ? `[관련해서 배운 것]\n${learned.map((s) => "- " + s).join("\n")}` : "[관련해서 배운 것]\n(아직 안 배움)",
      recalled.length ? `[문득 기억나는 대화]\n${recalled.map((s) => "- " + s).join("\n")}` : "",
      grown.length ? `[같이 자라온 날들]\n${grown.map((s) => "- " + s).join("\n")}` : "",
      `[주인이 방금 한 말]\n"${text}"`,
      "위 기억을 바탕으로, 척척박사 말고 '같이 배우는 공룡'으로 한두 문장 담백하게 답해라. 한국어.",
    ].filter(Boolean).join("\n\n");
    const reply = await deps.gemini.generateText(SYSTEM_PERSONA, prompt, { temperature: 0.95 });
    const clean = trunc(reply, 120);
    console.log(`\n   🗨️  주인> ${text}`);
    console.log(`   🦖 ${clean}`);
    if (grown.length) {
      console.log(`      (떠올린 성장 기억: ${grown.map((g) => g.slice(0, 24) + "…").join(" / ")})`);
    }
    await deps.conversation.append("user", text);
    await deps.conversation.append("kkoji", clean);
    console.log("");
  } catch (e) {
    console.log(`   ⚠️  대화 실패: ${trunc(String(e), 80)}`);
  }
}

async function runYear(deps: RealDeps, days: number, start: Date): Promise<void> {
  const evalEvery = Number(process.env.KKOJI_EVAL_EVERY || 30);
  const chatDays = new Set([45, 150, 305]); // 옛 사건을 떠올리게 하는 질문 날
  const curriculum = buildCurriculum(days);

  // Day 1 — 온보딩. 첫 일기는 고정(first_day.txt), print 를 배운다.
  console.log("\n════════════ 🥚 Day 1 — 깨어남 (print 를 배운다) ════════════");
  await deps.diary.writeRaw(FIRST_DAY_TXT, dateFor(start, 0));
  await deps.diary.write({
    date: dateFor(start, 0),
    learned: ["print 출력"],
    moments: ["주인이 'print' 를 알려줬다. 따라 하다 으에엑 했다."],
    peakEmotion: "joy",
  });
  console.log(`        🦖 print…? 이게 뭐지… 나도 해봤어. 으에엑. 근데 신기해!`);
  console.log(`        📔 (first_day.txt) ${trunc(FIRST_DAY_TXT, 60)}`);

  for (let i = 1; i < days; i++) {
    const date = dateFor(start, i);
    const concept = curriculum[i];
    const ev = eventFor(i);
    const isEvalDay = i % evalEvery === 0;

    // 마일스톤이면 진짜 코드 반응으로 그날의 감정을 정한다.
    let emotion: Emotion = ev?.emotion ?? "focus";
    if (isEvalDay) {
      console.log(`\n   ── 마일스톤 Day ${i + 1} · ${concept} — 진짜 코드 반응 ──`);
      emotion = await reactToCode(deps, concept);
    }

    const moments = [
      ev ? ev.moment : `주인이 ${concept} 를 공부했다.`,
      `나도 ${concept} 를 오늘 처음 봤다.`,
    ];
    const day: DayLog = { date, learned: [concept], moments, peakEmotion: emotion };

    try {
      const file = await deps.diary.write(day);
      const body = await fs.readFile(file, "utf8");
      const pct = maturityPct(deps.memory.size);
      const flag = ev ? " ★" : isEvalDay ? " ◆" : "";
      console.log(
        `Day ${String(i + 1).padStart(3)} │ ${date} │ ${String(pct).padStart(3)}% (${String(deps.memory.size).padStart(3)}개) │ ${trunc(body, 64)}${flag}`
      );
    } catch (e) {
      const msg = String(e);
      console.log(`Day ${String(i + 1).padStart(3)} │ ${date} │ ⚠️  ${trunc(msg, 70)}`);
      if (/401|UNAUTHENTICATED|invalid auth|API key|NO_API_KEY|403/i.test(msg)) {
        console.log("\n⛔ 토큰이 만료됐거나 막혔다. 일기 루프를 멈추고 곧장 엔딩으로 간다.");
        break;
      }
    }

    if (chatDays.has(i)) {
      const q = i === 45 ? "야 우리 처음에 print 배웠던 거 기억나?"
        : i === 150 ? "나 이번에 또 배포하려는데 떨려"
          : "예전에 새벽까지 버그 잡던 거 생각난다";
      await freeChat(deps, q);
    }
  }

  const pct = maturityPct(deps.memory.size);
  console.log(`\n🌳 1년 후 — ${growthStage(deps.memory.size)} · 아는 개념 ${deps.memory.size}개 · 성숙도 ${pct}%`);
}

/** 1년 뒤 — 진짜 엔딩 아크. (스크립트라 API 불필요, 시간배속으로 재생) */
async function runEnding(rootDir: string): Promise<void> {
  const t0 = Date.now();
  const stamp = () => `[+${String(((Date.now() - t0) / 1000).toFixed(1)).padStart(5)}s]`;
  const dir = path.join(rootDir, "ending");
  const deps: EndingDeps = {
    stateFile: path.join(dir, "ending.json"),
    diaryFolder: path.join(rootDir, "일기"),
    projectFolder: path.join(dir, "my_first_program"),
    setGone: (g) => console.log(`${stamp()} ${g ? "🫥 사라짐" : "🟢 보임"}`),
    say: (p, l) => console.log(`${stamp()} 🦖 (${p}) ${l || "…(침묵)"}`),
    autotype: (t) => console.log(`${stamp()} ⌨️  주인> ${t}`),
    openPath: (p) => console.log(`${stamp()} 📂 열림: ${path.basename(p)}`),
    notify: (title, body) => console.log(`${stamp()} 🔔 ${title} — ${body}`),
  };

  // 엔딩 게이트(실사용 365일 · 일기 1000개)를 충족한 상태로 시작 → 실제 진입 로직.
  await fs.mkdir(dir, { recursive: true });
  await fs.writeFile(
    deps.stateFile,
    JSON.stringify({ phase: "normal", onboarded: true, lastSeen: "2024-12-30", activeDays: 365, diaryCount: 1000, fadeStep: 0 }),
    "utf8"
  );

  console.log("\n════════════ 🌅 1년 뒤 — 엔딩 아크 (실제 로직, 시간배속) ════════════");
  console.log("── ① 자연 성숙 엔딩: 전조증상(침묵 곡선) → 사라짐 → 복귀 ──");
  const dirr = new EndingDirector(deps, false, 0.0001); // 비데모 = 실제 게이트로 진입
  await dirr.init(); // 365일·1000일기 충족 → fading 진입

  for (let i = 0; i < 30 && !dirr.isGone(); i++) {
    dirr.onActivity();
    await sleep(4600); // fadeGapMs(≈4.3s)보다 길게 → 다음 전조 비트
  }
  await sleep(9000); // '며칠 뒤'
  dirr.onActivity(); // 복귀 트리거
  for (let i = 0; i < 400 && !dirr.isQuiet(); i++) {
    await sleep(20);
  }

  console.log("\n── ② 끄는 엔딩: 담담하지만 슬픈 작별 → 안녕.txt 강제 오픈 → 종료 ──");
  await dirr.turnOff();
  console.log(`${stamp()} ✅ 끝. (안녕.txt 가 디스크에 남았다)`);

  // 마지막으로 안녕.txt 를 그대로 보여준다.
  try {
    const letter = await fs.readFile(path.join(rootDir, "일기", "안녕.txt"), "utf8");
    console.log("\n────────── 📄 안녕.txt ──────────\n" + letter);
  } catch { /* 조용히 */ }
}

async function main(): Promise<void> {
  if (!process.env.GEMINI_API_KEY) {
    console.error("GEMINI_API_KEY 환경변수가 필요하다.");
    process.exit(1);
  }
  const days = Number(process.env.KKOJI_DAYS || 365);
  const rootDir = await fs.mkdtemp(path.join(os.tmpdir(), "kkoji-year-"));
  const start = new Date(Date.UTC(2024, 0, 1));

  const settings = new EnvSettings();
  const gemini = new GeminiClient(settings);
  const memory = new MemoryStore(path.join(rootDir, "memory.json"), gemini);
  const episodic = new EpisodicStore(path.join(rootDir, "episodic.json"), gemini);
  const conversation = new ConversationStore(path.join(rootDir, "conversation.json"), gemini);
  await Promise.all([memory.load(), episodic.load(), conversation.load()]);
  const deps: RealDeps = {
    gemini, memory, episodic, conversation,
    evaluator: new Evaluator(gemini, memory),
    diary: new DiaryWriter(gemini, memory, path.join(rootDir, "일기"), episodic),
  };

  console.log(`🦖 꼬질룡 — 진짜 1년 시뮬 (model=${settings.textModel}, days=${days})`);
  console.log(`작업폴더: ${rootDir}`);
  const t0 = Date.now();
  await runYear(deps, days, start);
  await runEnding(rootDir);
  console.log(`\n⏱  총 ${(((Date.now() - t0) / 60000)).toFixed(1)}분 · 일기폴더: ${path.join(rootDir, "일기")}`);
}

if (require.main === module) {
  void main().catch((e) => { console.error(e); process.exit(1); });
}

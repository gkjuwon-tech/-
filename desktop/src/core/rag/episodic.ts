import * as fs from "fs/promises";
import * as path from "path";
import { Emotion } from "../types";
import { cosineSimilarity } from "./vector";

/** 임베딩만 필요한 최소 의존성. GeminiClient 가 그대로 만족한다(테스트 주입 쉬움). */
export interface Embedder {
  embed(text: string): Promise<number[]>;
}

/** 하루치 '성장 에피소드'. 개념 라벨이 아니라 '그날 무슨 일이 있었나'의 서사. */
export interface Episode {
  date: string; // YYYY-MM-DD
  summary: string; // 그날의 성장/감정 모먼트 한두 줄
  emotion: Emotion; // 그날 가장 강했던 감정
  embedding: number[];
}

/**
 * 에피소딕 성장 기억 = RAG 고도화의 본체.
 *
 * 기존 MemoryStore 는 "for문" 같은 '개념 라벨'만 임베딩해서, 무엇을 아는지는
 * 알아도 '어떻게 자라왔는지(서사·감정)'는 잃어버렸다. 이 저장소는 일기 본문에서
 * 뽑은 하루치 모먼트를 통째로 임베딩해, 일반 대화에서도 일기에서도 "그때 그날"을
 * 의미로 다시 꺼낸다. 그래서 성장의 맥락을 안 잃는다.
 *
 * 회상은 순수 코사인이 아니라 '현저성(salience) 재랭킹'을 쓴다:
 *  - 최신일수록 살짝 가산(recency) — 최근의 나를 더 또렷이.
 *  - 감정이 셌던 날 가산(moved/rage/joy) — 뭉클했던/빡쳤던 날은 더 잘 떠오른다.
 *  - 그리고 '맨 처음 날(origin)'은 따로 앵커로 보존 — 뿌리는 절대 안 잊는다.
 */
export class EpisodicStore {
  private static readonly RECALL_MIN = 0.55;
  private static readonly RECENCY_MAX = 0.05;
  private static readonly EMOTION_BOOST: Partial<Record<Emotion, number>> = {
    moved: 0.06,
    rage: 0.05,
    joy: 0.05,
    worry: 0.03,
    sulk: 0.03,
  };

  private episodes: Episode[] = [];
  private loaded = false;

  constructor(
    private readonly filePath: string,
    private readonly embedder: Embedder,
    /** 회상 컷오프. 임베딩 모델마다 유사도 스케일이 달라 노브로 뺀다(기본 0.55). */
    private readonly recallMin = EpisodicStore.RECALL_MIN
  ) {}

  async load(): Promise<void> {
    if (this.loaded) {
      return;
    }
    try {
      this.episodes = JSON.parse(await fs.readFile(this.filePath, "utf8"));
    } catch {
      this.episodes = []; // 아직 아무 날도 안 살았다.
    }
    this.loaded = true;
  }

  get size(): number {
    return this.episodes.length;
  }

  /** 성장의 뿌리 — 제일 처음 살았던 날. 회상에서 앵커로 쓴다. */
  origin(): Episode | undefined {
    if (this.episodes.length === 0) {
      return undefined;
    }
    return [...this.episodes].sort((a, b) => a.date.localeCompare(b.date))[0];
  }

  /**
   * 하루를 에피소드로 새긴다. 같은 날짜는 덮어쓴다(하루 한 장).
   * 임베딩 실패(키 없음 등)면 조용히 건너뛴다 — 일기 자체는 안 막는다.
   */
  async record(date: string, summary: string, emotion: Emotion): Promise<boolean> {
    await this.load();
    const clean = summary.replace(/\s+/g, " ").trim().slice(0, 400);
    if (!clean) {
      return false;
    }
    let embedding: number[];
    try {
      embedding = await this.embedder.embed(clean);
    } catch {
      return false;
    }
    this.episodes = this.episodes.filter((e) => e.date !== date);
    this.episodes.push({ date, summary: clean, emotion, embedding });
    await this.persist();
    return true;
  }

  /**
   * 질문/오늘의 맥락과 의미적으로 관련된 과거의 날들을 현저성 재랭킹으로 끌어온다.
   * 반환은 "(YYYY-MM-DD) 요약" 꼴 — 날짜 맥락이 같이 따라가서 시간감을 안 잃는다.
   */
  async recall(query: string, k = 3): Promise<string[]> {
    await this.load();
    if (this.episodes.length === 0) {
      return [];
    }
    let q: number[];
    try {
      q = await this.embedder.embed(query);
    } catch {
      return [];
    }
    const byDate = [...this.episodes].sort((a, b) =>
      a.date.localeCompare(b.date)
    );
    const last = byDate.length - 1;
    return byDate
      .map((e, i) => {
        const base = cosineSimilarity(q, e.embedding);
        const recency =
          last === 0 ? 0 : (i / last) * EpisodicStore.RECENCY_MAX;
        const emo = EpisodicStore.EMOTION_BOOST[e.emotion] ?? 0;
        return { e, base, score: base + recency + emo };
      })
      .filter((x) => x.base >= this.recallMin)
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map((x) => `(${x.e.date}) ${x.e.summary}`);
  }

  /**
   * 회상 + 뿌리 앵커를 한 번에. 관련 과거 날들 위에 '맨 처음 날'을 살며시 얹어,
   * 성장의 시작점을 늘 곁에 둔다(이미 회상에 들어왔으면 중복은 뺀다).
   */
  async recallWithOrigin(query: string, k = 3): Promise<string[]> {
    const recalled = await this.recall(query, k);
    const root = this.origin();
    if (!root) {
      return recalled;
    }
    const rootLine = `(${root.date}) ${root.summary}`;
    if (recalled.some((r) => r.startsWith(`(${root.date})`))) {
      return recalled;
    }
    return [...recalled, rootLine];
  }

  private async persist(): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.writeFile(
      this.filePath,
      JSON.stringify(this.episodes, null, 2),
      "utf8"
    );
  }
}

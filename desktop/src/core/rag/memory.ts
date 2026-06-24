import * as fs from "fs/promises";
import * as path from "path";
import { GeminiClient } from "../gemini/client";
import { MemoryEntry } from "../types";
import { cosineSimilarity } from "./vector";

/**
 * 꼬질룡의 기억 저장소 = RAG의 본체. (기획서 §11.4 The Memory)
 *
 * 한 번 본 개념을 임베딩과 함께 영속 저장하고, 의미 기반(코사인 유사도)으로
 * 회상한다. 표현이 살짝 달라도("for loop" vs "for문") 같은 개념이면 안다고 본다.
 * 단일 JSON으로 저장 → 재설치/앱 재시작에도 살아남는다.
 */
export class MemoryStore {
  private static readonly RECALL_THRESHOLD = 0.82;

  private entries: MemoryEntry[] = [];
  private loaded = false;

  constructor(
    private readonly filePath: string,
    private readonly gemini: GeminiClient
  ) {}

  async load(): Promise<void> {
    if (this.loaded) {
      return;
    }
    try {
      const raw = await fs.readFile(this.filePath, "utf8");
      this.entries = JSON.parse(raw);
    } catch {
      this.entries = []; // 첫 만남.
    }
    this.loaded = true;
  }

  knownConcepts(): string[] {
    return this.entries.map((e) => e.concept);
  }

  get size(): number {
    return this.entries.length;
  }

  oldest(): MemoryEntry | undefined {
    return this.entries.length
      ? [...this.entries].sort((a, b) =>
          a.firstSeen.localeCompare(b.firstSeen)
        )[0]
      : undefined;
  }

  async recall(
    concept: string,
    queryEmbedding?: number[]
  ): Promise<{ known: boolean; match?: MemoryEntry; score: number }> {
    await this.load();
    if (this.entries.length === 0) {
      return { known: false, score: 0 };
    }
    const q = queryEmbedding ?? (await this.gemini.embed(concept));
    let best: MemoryEntry | undefined;
    let bestScore = -1;
    for (const e of this.entries) {
      const score = cosineSimilarity(q, e.embedding);
      if (score > bestScore) {
        bestScore = score;
        best = e;
      }
    }
    return {
      known: bestScore >= MemoryStore.RECALL_THRESHOLD,
      match: best,
      score: bestScore,
    };
  }

  /**
   * 자유 문장(질문 등)과 의미적으로 관련된, 이미 배운 개념을 top-K로 끌어온다.
   * 대화에서 "이건 배운 거니까 (두루뭉술~정확) 답할 수 있다"를 판단하는 데 쓴다.
   */
  async recallTop(query: string, k = 5, minScore = 0.5): Promise<string[]> {
    await this.load();
    if (this.entries.length === 0) {
      return [];
    }
    const q = await this.gemini.embed(query).catch(() => null);
    if (!q) {
      return [];
    }
    return this.entries
      .map((e) => ({ e, s: cosineSimilarity(q, e.embedding) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .filter((x) => x.s >= minScore)
      .map((x) => x.e.concept);
  }

  async classify(
    concepts: string[]
  ): Promise<{ known: string[]; novel: string[] }> {
    await this.load();
    const known: string[] = [];
    const novel: string[] = [];
    for (const c of concepts) {
      const r = await this.recall(c);
      (r.known ? known : novel).push(c);
    }
    return { known, novel };
  }

  async learn(
    concept: string,
    context: string,
    diaryRef?: string
  ): Promise<boolean> {
    await this.load();
    const embedding = await this.gemini.embed(concept);
    const r = await this.recall(concept, embedding);
    if (r.known) {
      return false;
    }
    this.entries.push({
      concept,
      firstSeen: today(),
      embedding,
      context: context.slice(0, 280),
      diaryRef,
    });
    await this.persist();
    return true;
  }

  async learnMany(
    concepts: string[],
    context: string,
    diaryRef?: string
  ): Promise<string[]> {
    const fresh: string[] = [];
    for (const c of concepts) {
      if (await this.learn(c, context, diaryRef)) {
        fresh.push(c);
      }
    }
    return fresh;
  }

  private async persist(): Promise<void> {
    await fs.mkdir(path.dirname(this.filePath), { recursive: true });
    await fs.writeFile(
      this.filePath,
      JSON.stringify(this.entries, null, 2),
      "utf8"
    );
  }
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

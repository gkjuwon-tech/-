import * as vscode from "vscode";
import { GeminiClient } from "../gemini/client";
import { MemoryEntry } from "../types";
import { cosineSimilarity } from "./vector";

/**
 * 꼬질룡의 기억 저장소 = RAG의 본체.
 *
 * 기획서 §11.4 "The Memory":
 *   - 일기(.txt)는 사람이 읽는 표면.
 *   - 그 뒤에서 "꼬질룡이 아는 개념"을 임베딩과 함께 영속 저장하는 인덱스가 이거다.
 *   - 한 번 같이 공부한 건 다음에 다시 만나도 기억한다.
 *
 * 의미 기반(임베딩 코사인 유사도)으로 회상하기 때문에, 표현이 살짝 달라도
 * ("for loop" vs "for문") 같은 개념이면 안다고 인식한다.
 *
 * 저장 위치: globalStorage 안의 단일 JSON. 한 번 쓰면 재설치해도 살아남는다.
 */
export class MemoryStore {
  private static readonly FILE = "memory.json";
  /** 이 임계값 이상으로 비슷하면 "이미 아는 개념"으로 본다. */
  private static readonly RECALL_THRESHOLD = 0.82;

  private entries: MemoryEntry[] = [];
  private loaded = false;
  private readonly uri: vscode.Uri;

  constructor(
    context: vscode.ExtensionContext,
    private readonly gemini: GeminiClient
  ) {
    this.uri = vscode.Uri.joinPath(
      context.globalStorageUri,
      MemoryStore.FILE
    );
  }

  /** 디스크에서 기억을 깨운다. 멱등. */
  async load(): Promise<void> {
    if (this.loaded) {
      return;
    }
    try {
      const bytes = await vscode.workspace.fs.readFile(this.uri);
      this.entries = JSON.parse(Buffer.from(bytes).toString("utf8"));
    } catch {
      this.entries = []; // 첫 만남. 아직 아무것도 모른다.
    }
    this.loaded = true;
  }

  /** 지금까지 아는 모든 개념 이름. 페르소나 MEMORY 블록에 그대로 들어간다. */
  knownConcepts(): string[] {
    return this.entries.map((e) => e.concept);
  }

  get size(): number {
    return this.entries.length;
  }

  /** 가장 오래된 기억 (첫 만남 회수용). */
  oldest(): MemoryEntry | undefined {
    return this.entries.length
      ? [...this.entries].sort((a, b) => a.firstSeen.localeCompare(b.firstSeen))[0]
      : undefined;
  }

  /**
   * 개념 하나가 "이미 아는 것"인지 의미 기반으로 회상한다.
   * @returns 가장 비슷한 기억 + 점수. recall 여부는 known 플래그로.
   */
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
   * 한 묶음의 개념을 받아 RAG에 비추어 "아는 것 / 처음 보는 것"으로 가른다.
   * 평가 엔진과 일기 작성기가 공유하는 핵심 분기.
   */
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

  /**
   * 처음 본 개념을 기억에 새긴다. "오늘 배웠다."
   * 이미 비슷한 게 있으면 조용히 무시한다 (중복 학습 방지).
   */
  async learn(
    concept: string,
    context: string,
    diaryRef?: string
  ): Promise<boolean> {
    await this.load();
    const embedding = await this.gemini.embed(concept);
    const r = await this.recall(concept, embedding);
    if (r.known) {
      return false; // 이미 안다.
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

  /** 여러 개념을 한 번에 학습. 실제로 새로 배운 것만 돌려준다. */
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
    await vscode.workspace.fs.createDirectory(
      vscode.Uri.joinPath(this.uri, "..")
    );
    const json = JSON.stringify(this.entries, null, 2);
    await vscode.workspace.fs.writeFile(this.uri, Buffer.from(json, "utf8"));
  }
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

import * as fs from "fs/promises";
import * as path from "path";
import { GeminiClient } from "./gemini/client";
import { cosineSimilarity } from "./rag/vector";

interface Turn {
  role: "user" | "kkoji";
  text: string;
}

/** 오래된 대화를 1~2문장으로 압축해 임베딩과 함께 보관한 "기억 카드". */
interface MemoryCard {
  summary: string;
  embedding: number[];
  date: string;
}

interface Stored {
  live: Turn[];
  cards: MemoryCard[];
}

/**
 * 대화판 RAG. 꼬질룡과 나눈 대화의 맥락을 보존한다.
 *
 * Gemini 컨텍스트가 깡패라 웬만하면 라이브 윈도우(최근 대화)를 통째로 들고 간다.
 * 라이브가 예산(글자 수)을 넘치면, 오래된 절반을 Gemini가 1~2문장으로 압축해
 * 임베딩과 함께 "기억 카드"로 보관한다. 다음에 관련된 얘기가 나오면 의미 검색
 * (코사인 유사도)으로 그 카드를 다시 꺼내 맥락에 끼운다. 카드가 너무 쌓이면
 * 카드끼리 또 압축한다(압축의 압축).
 */
export class ConversationStore {
  private static readonly LIVE_BUDGET = 16000; // 라이브로 들고 갈 최대 글자
  private static readonly RECALL_K = 4;
  private static readonly RECALL_MIN = 0.68;
  private static readonly MAX_CARDS = 40;

  private live: Turn[] = [];
  private cards: MemoryCard[] = [];
  private loaded = false;

  constructor(
    private readonly file: string,
    private readonly gemini: GeminiClient
  ) {}

  async load(): Promise<void> {
    if (this.loaded) {
      return;
    }
    try {
      const raw: Stored = JSON.parse(await fs.readFile(this.file, "utf8"));
      this.live = raw.live ?? [];
      this.cards = raw.cards ?? [];
    } catch {
      /* 첫 실행 */
    }
    this.loaded = true;
  }

  /** 최근 대화(라이브 윈도우)를 프롬프트용 텍스트로. */
  recentText(): string {
    return this.live
      .map((t) => `${t.role === "user" ? "주인" : "꼬질룡"}: ${t.text}`)
      .join("\n");
  }

  /** 지금 메시지와 의미적으로 관련된 옛 대화 요약을 끌어온다. */
  async recall(query: string): Promise<string[]> {
    await this.load();
    if (this.cards.length === 0) {
      return [];
    }
    const q = await this.gemini.embed(query).catch(() => null);
    if (!q) {
      return [];
    }
    return this.cards
      .map((c) => ({ c, s: cosineSimilarity(q, c.embedding) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, ConversationStore.RECALL_K)
      .filter((x) => x.s >= ConversationStore.RECALL_MIN)
      .map((x) => x.c.summary);
  }

  /** 한 턴 추가 → 필요하면 압축 → 저장. */
  async append(role: "user" | "kkoji", text: string): Promise<void> {
    await this.load();
    this.live.push({ role, text: text.slice(0, 2000) });
    await this.compressIfNeeded();
    await this.persist();
  }

  private liveChars(): number {
    return this.live.reduce((n, t) => n + t.text.length, 0);
  }

  private async compressIfNeeded(): Promise<void> {
    if (this.liveChars() <= ConversationStore.LIVE_BUDGET) {
      return;
    }
    // 오래된 절반을 한 장의 기억 카드로 압축.
    const half = this.live.splice(0, Math.max(2, Math.ceil(this.live.length / 2)));
    const card = await this.summarize(half.map(turnText).join("\n"));
    if (card) {
      this.cards.push(card);
    }
    // 카드가 너무 쌓이면 카드끼리 또 압축한다 (압축의 압축).
    if (this.cards.length > ConversationStore.MAX_CARDS) {
      const old = this.cards.splice(0, Math.ceil(this.cards.length / 2));
      const merged = await this.summarize(old.map((c) => c.summary).join("\n"));
      if (merged) {
        this.cards.unshift(merged);
      }
    }
  }

  private async summarize(text: string): Promise<MemoryCard | undefined> {
    try {
      const summary = await this.gemini.generateText(
        "다음 대화를 꼬질룡(공룡)이 기억할 1~2문장으로 압축해라. 주인에 대한 사실/취향/약속/맥락 위주로. 캐릭터 연기 없이 사실만.",
        text,
        { temperature: 0.3 }
      );
      const clean = summary.replace(/\s+/g, " ").trim().slice(0, 300);
      if (!clean) {
        return undefined;
      }
      const embedding = await this.gemini.embed(clean);
      return { summary: clean, embedding, date: today() };
    } catch {
      return undefined; // 키 없거나 실패하면 압축은 건너뛴다(라이브만 좀 잘림).
    }
  }

  private async persist(): Promise<void> {
    try {
      await fs.mkdir(path.dirname(this.file), { recursive: true });
      const data: Stored = { live: this.live, cards: this.cards };
      await fs.writeFile(this.file, JSON.stringify(data), "utf8");
    } catch {
      /* 조용히 */
    }
  }
}

function turnText(t: Turn): string {
  return `${t.role === "user" ? "주인" : "꼬질룡"}: ${t.text}`;
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

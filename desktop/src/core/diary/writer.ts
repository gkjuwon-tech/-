import * as fs from "fs/promises";
import * as path from "path";
import { GeminiClient } from "../gemini/client";
import { MemoryStore } from "../rag/memory";
import { SYSTEM_PERSONA, buildMemoryBlock } from "../persona";
import { diaryMaturity } from "../maturity";
import { DayLog } from "../types";

/**
 * The Pen. 하루의 관찰을 짧고 귀여운 일기 .txt로 남기고, 새 개념을 RAG에 새긴다.
 * (기획서 §9) 오늘 "배웠다"고 적은 건 내일부터 "안다".
 */
export class DiaryWriter {
  constructor(
    private readonly gemini: GeminiClient,
    private readonly memory: MemoryStore,
    private readonly folder: string
  ) {}

  /**
   * AI 없이 고정 내용으로 일기를 남긴다. 1일차(수미상관)용.
   * @returns 작성된 파일 경로.
   */
  async writeRaw(content: string, date = today()): Promise<string> {
    await fs.mkdir(this.folder, { recursive: true });
    const file = path.join(this.folder, `${date}.txt`);
    await fs.writeFile(file, content, "utf8");
    return file;
  }

  /** 오늘 일기가 이미 있으면 true (덮어쓰기 방지용). */
  async hasToday(): Promise<boolean> {
    return !!(await this.todayPath());
  }

  /** @returns 작성된 일기 파일의 절대 경로. */
  async write(day: DayLog): Promise<string> {
    await fs.mkdir(this.folder, { recursive: true });
    const file = path.join(this.folder, `${day.date}.txt`);
    const diaryRef = `${day.date}.txt`;

    const { novel } = await this.memory.classify(day.learned);
    const text = await this.compose(day, novel);
    await fs.writeFile(file, text, "utf8");

    await this.memory.learnMany(novel, day.moments.join(" / "), diaryRef);
    return file;
  }

  private async compose(day: DayLog, novel: string[]): Promise<string> {
    const memoryBlock = buildMemoryBlock(this.memory.knownConcepts());
    const maturity = diaryMaturity(this.memory.size);

    const prompt = `${memoryBlock}

오늘은 ${day.date}.
[오늘 처음 본 개념] ${novel.join(", ") || "(딱히 새로운 건 없었다)"}
[오늘 주인의 모습] ${day.moments.join(" / ") || "조용히 코딩했다."}
[오늘 가장 강했던 감정] ${day.peakEmotion}

이걸로 오늘의 일기를 써라.
- 너의 성숙도: ${maturity}
- 학습 일기(처음 본 개념을 "오늘 배웠다"며 신기해함) + 감정 일기(주인의 하루)를 섞어라.
- 5~10줄. 일기체. 제목/머리말/이모지/서명 없이 본문만.`;

    const body = await this.gemini.generateText(SYSTEM_PERSONA, prompt, {
      temperature: 0.9,
    });
    return `${body.trim()}\n`;
  }

  async todayPath(): Promise<string | undefined> {
    const file = path.join(this.folder, `${today()}.txt`);
    try {
      await fs.access(file);
      return file;
    } catch {
      return undefined;
    }
  }
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

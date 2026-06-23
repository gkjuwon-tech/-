import * as vscode from "vscode";
import * as fs from "fs/promises";
import * as path from "path";
import { GeminiClient } from "../gemini/client";
import { MemoryStore } from "../rag/memory";
import { Config } from "../config";
import { SYSTEM_PERSONA, buildMemoryBlock } from "../persona";
import { DayLog } from "../types";

/**
 * The Pen. 하루의 관찰을 짧고 귀여운 일기 .txt로 남긴다. (기획서 §9)
 *
 * 일기는 사람이 읽는 평문 표면이고, 그 안에서 "오늘 처음 본 개념"은 RAG에
 * 학습으로 새겨진다. 그래서 오늘 일기에 "배웠다"고 적은 건 내일부터 "안다".
 */
export class DiaryWriter {
  constructor(
    private readonly gemini: GeminiClient,
    private readonly memory: MemoryStore,
    private readonly config: Config
  ) {}

  /** 하루 누적을 일기로 쓰고, 새 개념을 RAG에 학습시킨다. */
  async write(day: DayLog): Promise<vscode.Uri> {
    const folder = this.config.diaryFolder;
    await fs.mkdir(folder, { recursive: true });
    const file = path.join(folder, `${day.date}.txt`);
    const diaryRef = `${day.date}.txt`;

    // 오늘 처음 본 것만 진짜 새 개념. (이미 아는 건 또 안 배운다)
    const { novel } = await this.memory.classify(day.learned);

    const text = await this.compose(day, novel);
    await fs.writeFile(file, text, "utf8");

    // 일기에 적었으니 이제 안다. RAG에 새긴다.
    await this.memory.learnMany(novel, day.moments.join(" / "), diaryRef);

    return vscode.Uri.file(file);
  }

  /** 멍청할 땐 멍청하게, 똑똑해지면 깊게. 톤은 아는 개념 수가 정한다. */
  private async compose(day: DayLog, novel: string[]): Promise<string> {
    const memoryBlock = buildMemoryBlock(this.memory.knownConcepts());
    const maturity =
      this.memory.size < 50
        ? "아직 멍청하고 모르는 게 많다. 짧고 순수하게, 가끔 맞춤법도 틀리게."
        : this.memory.size < 300
        ? "조금 자랐다. 패턴이 보이기 시작한다."
        : "많이 자랐다. 가끔 또박또박, 회고가 깊어진다.";

    const prompt = `${memoryBlock}

오늘은 ${day.date}.
[오늘 처음 본 개념] ${novel.join(", ") || "(딱히 새로운 건 없었다)"}
[오늘 주인의 모습] ${day.moments.join(" / ") || "조용히 코딩했다."}
[오늘 가장 강했던 감정] ${day.peakEmotion}

이걸로 오늘의 일기를 써라.
- 너의 성숙도: ${maturity}
- 학습 일기(처음 본 개념을 "오늘 배웠다"며 신기해함) + 감정 일기(주인의 하루)를 섞어라.
- 5~10줄. 일기체. 제목/머리말/이모지 없이 본문만. 마지막 줄 서명도 없이.`;

    const body = await this.gemini.generateText(SYSTEM_PERSONA, prompt, {
      temperature: 0.9,
    });
    return `${body.trim()}\n`;
  }

  /** 오늘 일기 파일 경로 (없으면 undefined). */
  async todayUri(): Promise<vscode.Uri | undefined> {
    const date = new Date().toISOString().slice(0, 10);
    const file = path.join(this.config.diaryFolder, `${date}.txt`);
    try {
      await fs.access(file);
      return vscode.Uri.file(file);
    } catch {
      return undefined;
    }
  }
}

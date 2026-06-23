import { GeminiClient } from "../gemini/client";
import { MemoryStore } from "../rag/memory";
import { SYSTEM_PERSONA, buildMemoryBlock } from "../persona";
import { Evaluation, RageLevel } from "../types";

/**
 * The Brain. 코드 한 조각을 받아 개념 추출 → RAG 분기 → 빡침/춤 판정 + 대사.
 * "아는 척/모르는 척"의 진실은 항상 RAG(MemoryStore)가 쥔다.
 */
export class Evaluator {
  constructor(
    private readonly gemini: GeminiClient,
    private readonly memory: MemoryStore
  ) {}

  async evaluate(code: string, languageId: string): Promise<Evaluation> {
    const snippet = code.slice(0, 4000);

    const concepts = await this.extractConcepts(snippet, languageId);
    const { known, novel } = await this.memory.classify(concepts);

    const memoryBlock = buildMemoryBlock(this.memory.knownConcepts());
    const prompt = `${memoryBlock}

[지금 주인의 코드 (${languageId})]
\`\`\`
${snippet}
\`\`\`

[이번에 본 개념들]
- 이미 아는 것: ${known.join(", ") || "(없음)"}
- 처음 보는 것: ${novel.join(", ") || "(없음)"}

이 코드를 보고 너의 반응을 JSON으로만 답해라:
{
  "verdict": "dance" | "calm" | "rage",
  "rageLevel": 0~4,
  "line": "화면에 띄울 너의 한 마디"
}`;

    const raw = await this.gemini.generateText(SYSTEM_PERSONA, prompt, {
      json: true,
      temperature: 0.95,
    });
    const parsed = safeParse(raw);
    return {
      verdict: parsed.verdict,
      rageLevel: parsed.rageLevel,
      line: parsed.line,
      newConcepts: novel,
      knownConcepts: known,
    };
  }

  private async extractConcepts(
    code: string,
    languageId: string
  ): Promise<string[]> {
    const prompt = `다음 ${languageId} 코드에 등장하는 프로그래밍 "개념/문법/패턴"을 짧은 명사구로 최대 8개 뽑아라.
변수명이나 값이 아니라 일반화된 개념으로. (예: "for loop", "async/await", "dependency injection", "try/catch")
JSON 배열로만 답해라. 예: ["for loop", "string interpolation"]

\`\`\`
${code}
\`\`\``;
    const raw = await this.gemini.generateText(
      "너는 코드에서 프로그래밍 개념만 정확히 추출하는 정적 분석기다. 캐릭터 연기 없음.",
      prompt,
      { json: true, temperature: 0.2 }
    );
    try {
      const arr = JSON.parse(stripFence(raw));
      return Array.isArray(arr)
        ? arr.map((x) => String(x).trim()).filter(Boolean).slice(0, 8)
        : [];
    } catch {
      return [];
    }
  }
}

function stripFence(s: string): string {
  return s
    .trim()
    .replace(/^```(?:json)?/i, "")
    .replace(/```$/, "")
    .trim();
}

function safeParse(raw: string): {
  verdict: Evaluation["verdict"];
  rageLevel: RageLevel;
  line: string;
} {
  try {
    const o = JSON.parse(stripFence(raw));
    const verdict: Evaluation["verdict"] =
      o.verdict === "dance" || o.verdict === "rage" ? o.verdict : "calm";
    const rageLevel = clampRage(o.rageLevel);
    const line =
      typeof o.line === "string" && o.line.trim()
        ? o.line.trim()
        : "끄응… 뭔가 봤는데 말이 안 나와.";
    return { verdict, rageLevel: verdict === "rage" ? rageLevel : 0, line };
  } catch {
    return { verdict: "calm", rageLevel: 0, line: "음… 보는 중이야." };
  }
}

function clampRage(n: unknown): RageLevel {
  const v = Math.round(Number(n) || 0);
  return Math.max(0, Math.min(4, v)) as RageLevel;
}

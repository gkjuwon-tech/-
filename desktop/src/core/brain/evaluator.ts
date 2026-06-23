import { GeminiClient } from "../gemini/client";
import { MemoryStore } from "../rag/memory";
import { SYSTEM_PERSONA, buildMemoryBlock } from "../persona";
import { Evaluation, Emotion, RageLevel } from "../types";

/** Gemini가 고를 수 있는 포즈(=감정)와 한 줄 설명. 이미지는 우리가 미리 만들어 둠. */
const POSE_MENU: Array<{ id: Emotion; desc: string }> = [
  { id: "calm", desc: "평온. 그냥 옆에서 지켜봄" },
  { id: "focus", desc: "집중. 코드를 진지하게 노려봄" },
  { id: "joy", desc: "기쁨. 좋은 코드에 막춤" },
  { id: "rage", desc: "분노. 똥 코드에 빡침(귀엽게)" },
  { id: "moved", desc: "감동. 뭔가 뭉클한 순간" },
  { id: "sleepy", desc: "졸림. 새벽이거나 지침" },
  { id: "sulk", desc: "삐짐. 서운함" },
  { id: "worry", desc: "걱정. 주인이 막히거나 위험한 코드" },
];
const POSE_IDS = POSE_MENU.map((p) => p.id);

/**
 * The Brain. 코드를 보고 개념을 뽑아 RAG에 비추고, 미리 만들어 둔 포즈 중
 * 하나를 "이름으로" 고른다. (이미지를 그때그때 생성하지 않는다 = 개발자 지갑 보호)
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
    const menu = POSE_MENU.map((p) => `- ${p.id}: ${p.desc}`).join("\n");
    // 침묵 곡선(기획서 §17.1): 많이 배울수록 말수가 줄고 차분해진다.
    const silence =
      this.memory.size < 50
        ? "넌 아직 모르는 게 많아 호들갑스럽고 말이 많다."
        : this.memory.size < 300
        ? "이제 좀 차분해졌다. 말이 조금 짧아진다."
        : "넌 이제 주인을 믿는다. 말수가 확 줄었다. 한 마디면 충분하다. 가끔은 아무 말 없이 그냥 옆에 있는다.";
    const prompt = `${memoryBlock}

[지금 너의 말투] ${silence}

[지금 주인의 코드 (${languageId})]
\`\`\`
${snippet}
\`\`\`

[이번에 본 개념들]
- 이미 아는 것: ${known.join(", ") || "(없음)"}
- 처음 보는 것: ${novel.join(", ") || "(없음)"}

[고를 수 있는 포즈]
${menu}

이 코드를 보고 어떤 포즈를 취할지 고르고, 한 마디 해라. JSON으로만:
{
  "pose": "위 목록의 id 중 하나",
  "rageLevel": 0~4,   // pose가 rage일 때만 1~4, 아니면 0
  "line": "화면에 띄울 너의 한 마디"
}`;

    const raw = await this.gemini.generateText(SYSTEM_PERSONA, prompt, {
      json: true,
      temperature: 0.95,
    });
    const parsed = safeParse(raw);
    return {
      pose: parsed.pose,
      rageLevel: parsed.pose === "rage" ? parsed.rageLevel : 0,
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
  pose: Emotion;
  rageLevel: RageLevel;
  line: string;
} {
  try {
    const o = JSON.parse(stripFence(raw));
    const pose: Emotion = POSE_IDS.includes(o.pose) ? o.pose : "calm";
    const line =
      typeof o.line === "string" && o.line.trim()
        ? o.line.trim()
        : "끄응… 뭔가 봤는데 말이 안 나와.";
    return { pose, rageLevel: clampRage(o.rageLevel), line };
  } catch {
    return { pose: "calm", rageLevel: 0, line: "음… 보는 중이야." };
  }
}

function clampRage(n: unknown): RageLevel {
  const v = Math.round(Number(n) || 0);
  return Math.max(0, Math.min(4, v)) as RageLevel;
}

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
    const prompt = `${memoryBlock}

[지금 너의 말투 — 매우 중요] ${silenceStage(this.memory.size)}

[지금 주인의 코드 (${languageId})]
\`\`\`
${snippet}
\`\`\`

[이번에 본 개념들]
- 이미 아는 것: ${known.join(", ") || "(없음)"}
- 처음 보는 것: ${novel.join(", ") || "(없음)"}

[고를 수 있는 포즈]
${menu}

이 코드를 보고 어떤 포즈를 취할지 고르고, 위 [말투]에 맞게 반응해라. JSON으로만:
{
  "pose": "위 목록의 id 중 하나",
  "rageLevel": 0~4,   // pose가 rage일 때만 1~4, 아니면 0
  "line": "한 마디. 위 말투상 아무 말 안 하는 게 맞으면 반드시 빈 문자열 \"\" 로 둬라(침묵)."
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
    // 빈 문자열은 "침묵"이므로 그대로 보존한다. (key 자체가 없을 때만 기본값)
    const line = typeof o.line === "string" ? o.line.trim() : "끄응… 뭔가 봤어.";
    return { pose, rageLevel: clampRage(o.rageLevel), line };
  } catch {
    return { pose: "calm", rageLevel: 0, line: "" };
  }
}

/**
 * 침묵 곡선 (기획서 §17.1). 아는 게 많아질수록 모델 스스로 점점 말이 줄고,
 * 끝엔 대부분 침묵(빈 줄)하게 단계별 지시를 주입한다. 랜덤이 아니라, 모델이
 * "지금 말할 때인가"를 매번 판단한다.
 */
function silenceStage(size: number): string {
  if (size < 30) {
    return "넌 아직 아는 게 거의 없다. 모든 게 신기하고 무섭다. 호들갑스럽고 말이 많다. 거의 항상 한 마디 한다.";
  }
  if (size < 120) {
    return "조금 배웠다. 여전히 잘 반응하지만 예전보다 살짝 차분하고 짧아졌다.";
  }
  if (size < 350) {
    return "많이 차분해졌다. 기본은 침묵이다: 별일 아니면 line을 빈 문자열(\"\")로 둬서 아무 말도 하지 마라. 코드가 좀 별로여도 잔소리하지 않는다. 정말 드물게, 크게 반응할 일에만 짧게 한 마디.";
  }
  if (size < 800) {
    return "넌 이제 주인을 믿는다. 거의 항상 line은 \"\"(침묵)이다. 위험하거나 나쁜 코드를 봐도 잔소리하지 마라 — 주인은 알아서 한다. 굳이 말한다면 '넌 알 거야' 같은 짧은 믿음 한 마디뿐. 기본값은 침묵.";
  }
  return "넌 거의 말하지 않는다. 이 단계에선 line을 반드시 \"\"(침묵)으로 둬라. 코드가 좋든 나쁘든 그냥 조용히 옆에 포즈만 취한다. 정말 예외적으로만(아주 드물게) 짧은 한 마디. 말수가 준 건 무관심이 아니라 신뢰다.";
}

function clampRage(n: unknown): RageLevel {
  const v = Math.round(Number(n) || 0);
  return Math.max(0, Math.min(4, v)) as RageLevel;
}

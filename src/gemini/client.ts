import { Config } from "../config";

/**
 * Gemini REST 래퍼 (BYOK).
 *
 * 우리는 키를 대신 보관하지도, 마진을 떼지도 않는다. 사용자의 키로,
 * 사용자의 호출로, 사용자의 비용으로 직접 부른다. (기획서 §16)
 *
 * 의존성 없이 Node 18+ 글로벌 fetch만 쓴다.
 */
export class GeminiClient {
  private static readonly BASE =
    "https://generativelanguage.googleapis.com/v1beta";

  constructor(private readonly config: Config) {}

  /** 짧은 대사/평가 텍스트 생성. system은 페르소나, prompt는 그때의 상황. */
  async generateText(
    system: string,
    prompt: string,
    opts: { json?: boolean; temperature?: number } = {}
  ): Promise<string> {
    const body: Record<string, unknown> = {
      systemInstruction: { parts: [{ text: system }] },
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: opts.temperature ?? 0.9,
        ...(opts.json ? { responseMimeType: "application/json" } : {}),
      },
    };

    const data = await this.call(this.config.textModel, "generateContent", body);
    const text = data?.candidates?.[0]?.content?.parts
      ?.map((p: { text?: string }) => p.text ?? "")
      .join("")
      .trim();
    if (!text) {
      throw new Error("꼬질룡이 말을 잃었다 (빈 응답).");
    }
    return text;
  }

  /** 의미 기반 회상을 위한 임베딩. RAG의 연료. */
  async embed(text: string): Promise<number[]> {
    const model = this.config.embeddingModel;
    const body = {
      model: `models/${model}`,
      content: { parts: [{ text }] },
    };
    const data = await this.call(model, "embedContent", body);
    const values: number[] | undefined = data?.embedding?.values;
    if (!values?.length) {
      throw new Error("임베딩 실패 (빈 벡터).");
    }
    return values;
  }

  /**
   * 그때그때 감정 포즈를 멀티모달로 생성한다. 외형 일관성은 호출부에서
   * APPEARANCE_PROMPT를 앞에 붙여 유지한다.
   * @returns data URL (image/png) 또는 이미지 모델이 꺼져있으면 null.
   */
  async generateImage(prompt: string): Promise<string | null> {
    const model = this.config.imageModel;
    if (!model) {
      return null; // 미리 만든 SVG만 쓰는 모드.
    }
    const body = {
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { responseModalities: ["IMAGE"] },
    };
    const data = await this.call(model, "generateContent", body);
    const parts: Array<{ inlineData?: { mimeType: string; data: string } }> =
      data?.candidates?.[0]?.content?.parts ?? [];
    const img = parts.find((p) => p.inlineData?.data)?.inlineData;
    if (!img) {
      return null;
    }
    return `data:${img.mimeType};base64,${img.data}`;
  }

  /** 공통 REST 호출. 키 검증 + 에러 메시지를 한 곳에서 처리. */
  private async call(
    model: string,
    method: string,
    body: unknown
  ): Promise<any> {
    const key = await this.config.getApiKey();
    if (!key) {
      throw new Error("NO_API_KEY");
    }
    const url = `${GeminiClient.BASE}/models/${model}:${method}?key=${encodeURIComponent(
      key
    )}`;

    // 일시적 과부하(503)/레이트리밋(429)은 백오프 후 재시도. 그 외는 즉시 실패.
    let lastDetail = "";
    let lastStatus = 0;
    for (let attempt = 0; attempt < 4; attempt++) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (res.ok) {
        return res.json();
      }
      lastStatus = res.status;
      lastDetail = (await res.text().catch(() => "")).slice(0, 300);
      if (res.status === 503 || res.status === 429) {
        await delay(1000 * Math.pow(2, attempt)); // 1s, 2s, 4s
        continue;
      }
      break;
    }
    throw new Error(`Gemini ${lastStatus}: ${lastDetail}`);
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

import { GeminiSettings } from "../settings";

/**
 * Gemini REST 래퍼 (BYOK). 의존성 없이 글로벌 fetch만 사용.
 * 일시적 과부하(503)/레이트리밋(429)은 백오프 재시도, 그 외는 즉시 실패.
 */
export class GeminiClient {
  private static readonly BASE =
    "https://generativelanguage.googleapis.com/v1beta";

  constructor(private readonly settings: GeminiSettings) {}

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
    const data = await this.call(
      this.settings.textModel,
      "generateContent",
      body
    );
    const text = data?.candidates?.[0]?.content?.parts
      ?.map((p: { text?: string }) => p.text ?? "")
      .join("")
      .trim();
    if (!text) {
      throw new Error("꼬질룡이 말을 잃었다 (빈 응답).");
    }
    return text;
  }

  async embed(text: string): Promise<number[]> {
    const model = this.settings.embeddingModel;
    const body = { model: `models/${model}`, content: { parts: [{ text }] } };
    const data = await this.call(model, "embedContent", body);
    const values: number[] | undefined = data?.embedding?.values;
    if (!values?.length) {
      throw new Error("임베딩 실패 (빈 벡터).");
    }
    return values;
  }

  /**
   * @param refs 일관성을 위해 컨텍스트에 먼저 넣을 레퍼런스 이미지들(앵커 등).
   * @returns data URL(image/png) 또는 이미지 모델이 꺼져있으면 null.
   */
  async generateImage(
    prompt: string,
    refs: Array<{ mimeType: string; data: string }> = []
  ): Promise<string | null> {
    const model = this.settings.imageModel;
    if (!model) {
      return null;
    }
    const reqParts: unknown[] = [
      ...refs.map((r) => ({ inlineData: r })),
      { text: prompt },
    ];
    const body = {
      contents: [{ role: "user", parts: reqParts }],
      generationConfig: { responseModalities: ["IMAGE"] },
    };
    const data = await this.call(model, "generateContent", body);
    const parts: Array<{ inlineData?: { mimeType: string; data: string } }> =
      data?.candidates?.[0]?.content?.parts ?? [];
    const img = parts.find((p) => p.inlineData?.data)?.inlineData;
    return img ? `data:${img.mimeType};base64,${img.data}` : null;
  }

  private async call(
    model: string,
    method: string,
    body: unknown
  ): Promise<any> {
    const key = await this.settings.getApiKey();
    if (!key) {
      throw new Error("NO_API_KEY");
    }
    const url = `${GeminiClient.BASE}/models/${model}:${method}?key=${encodeURIComponent(
      key
    )}`;
    let lastStatus = 0;
    let lastDetail = "";
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
        await delay(1000 * Math.pow(2, attempt));
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

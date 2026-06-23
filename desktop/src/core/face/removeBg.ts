import { RemoveBgSettings } from "../settings";

/**
 * remove.bg 클라이언트. Gemini가 생성한 이미지는 배경이 꽉 차 있으니(누끼 X),
 * 모든 포즈 생성의 "마지막 단계"로 여기를 통과시켜 배경을 투명하게 만든다.
 * JSON API(image_file_b64)로 multipart 의존성 없이 fetch만 사용.
 */
export class RemoveBgClient {
  private static readonly ENDPOINT = "https://api.remove.bg/v1.0/removebg";

  constructor(private readonly settings: RemoveBgSettings) {}

  /** @param pngBase64 원본 PNG base64(data URL 접두사 없이). @returns 투명 PNG base64. */
  async strip(pngBase64: string): Promise<string> {
    const key = await this.settings.getRemoveBgKey();
    if (!key) {
      throw new Error("NO_REMOVEBG_KEY");
    }
    const res = await fetch(RemoveBgClient.ENDPOINT, {
      method: "POST",
      headers: { "X-Api-Key": key, "Content-Type": "application/json" },
      body: JSON.stringify({
        image_file_b64: pngBase64,
        size: "auto",
        format: "png",
      }),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`remove.bg ${res.status}: ${detail.slice(0, 300)}`);
    }
    const buf = Buffer.from(await res.arrayBuffer());
    return buf.toString("base64");
  }
}

import { Config } from "../config";

/**
 * remove.bg 클라이언트.
 *
 * Gemini가 생성한 이미지는 배경이 꽉 차서 나온다(누끼 없음). 그래서 모든 포즈
 * 생성의 "마지막 단계"로 여기를 통과시켜 배경을 투명하게 만든다. 포즈 프롬프트가
 * 배경을 순백 단색으로 강제하기 때문에 누끼가 깨끗하게 떨어진다.
 *
 * BYOK 동일 원칙: 키는 사용자 거. SecretStorage에 보관. 우리 서버 안 거침.
 * JSON API(image_file_b64)를 써서 multipart 의존성 없이 fetch만으로 처리한다.
 */
export class RemoveBgClient {
  private static readonly ENDPOINT = "https://api.remove.bg/v1.0/removebg";

  constructor(private readonly config: Config) {}

  /**
   * 배경을 제거한다.
   * @param pngBase64 원본 PNG의 base64 (data URL 접두사 없이).
   * @returns 투명 배경 PNG의 base64.
   */
  async strip(pngBase64: string): Promise<string> {
    const key = await this.config.getRemoveBgKey();
    if (!key) {
      throw new Error("NO_REMOVEBG_KEY");
    }
    const res = await fetch(RemoveBgClient.ENDPOINT, {
      method: "POST",
      headers: {
        "X-Api-Key": key,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image_file_b64: pngBase64,
        size: "auto",
        format: "png", // 투명도 유지를 위해 반드시 PNG.
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

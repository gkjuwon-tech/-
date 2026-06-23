/**
 * 코어가 필요로 하는 설정 계약. 코어 로직을 Electron/파일시스템 같은 구체
 * 구현에서 분리한다 (의존성 역전).
 */
export interface GeminiSettings {
  getApiKey(): Promise<string | undefined>;
  readonly textModel: string;
  readonly embeddingModel: string;
  /** 빈 문자열이면 이미지 생성 비활성(미리 만든 SVG/번들 PNG만). */
  readonly imageModel: string;
}

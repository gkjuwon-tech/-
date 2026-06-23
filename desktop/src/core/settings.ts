/**
 * 코어가 필요로 하는 설정 계약(인터페이스). 이렇게 분리하면 코어 로직이
 * Electron/파일시스템 같은 구체 구현에 묶이지 않는다 (의존성 역전).
 */
export interface GeminiSettings {
  getApiKey(): Promise<string | undefined>;
  readonly textModel: string;
  readonly embeddingModel: string;
  /** 빈 문자열이면 이미지 생성 비활성(미리 만든 SVG만). */
  readonly imageModel: string;
}

export interface RemoveBgSettings {
  getRemoveBgKey(): Promise<string | undefined>;
}

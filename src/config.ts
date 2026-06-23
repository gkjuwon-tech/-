import * as vscode from "vscode";
import * as os from "os";
import * as path from "path";

/**
 * 설정 + 비밀(API 키) 접근을 한 곳으로 모은다.
 * API 키는 settings.json이 아니라 SecretStorage에 넣는다 (코드/동기화에 새지 않게).
 */
export class Config {
  private static readonly KEY = "kkojilryong.geminiApiKey";
  private static readonly REMOVEBG_KEY = "kkojilryong.removeBgApiKey";

  constructor(private readonly context: vscode.ExtensionContext) {}

  private get cfg(): vscode.WorkspaceConfiguration {
    return vscode.workspace.getConfiguration("kkojilryong");
  }

  async getApiKey(): Promise<string | undefined> {
    return this.context.secrets.get(Config.KEY);
  }

  async setApiKey(key: string): Promise<void> {
    await this.context.secrets.store(Config.KEY, key.trim());
  }

  async hasApiKey(): Promise<boolean> {
    return !!(await this.getApiKey());
  }

  async getRemoveBgKey(): Promise<string | undefined> {
    return this.context.secrets.get(Config.REMOVEBG_KEY);
  }

  async setRemoveBgKey(key: string): Promise<void> {
    await this.context.secrets.store(Config.REMOVEBG_KEY, key.trim());
  }

  async hasRemoveBgKey(): Promise<boolean> {
    return !!(await this.getRemoveBgKey());
  }

  get textModel(): string {
    return this.cfg.get<string>("geminiModel", "gemini-2.0-flash");
  }

  get embeddingModel(): string {
    return this.cfg.get<string>("embeddingModel", "text-embedding-004");
  }

  /** 빈 문자열이면 이미지 생성을 끄고 미리 만든 SVG만 쓴다. */
  get imageModel(): string {
    return this.cfg.get<string>("imageModel", "");
  }

  get diaryFolder(): string {
    const custom = this.cfg.get<string>("diaryFolder", "").trim();
    return custom || path.join(os.homedir(), "꼬질룡_일기");
  }

  get debounceMs(): number {
    return this.cfg.get<number>("debounceMs", 1500);
  }

  get reduceMotion(): boolean {
    return this.cfg.get<boolean>("reduceMotion", false);
  }

  get mute(): boolean {
    return this.cfg.get<boolean>("mute", false);
  }
}

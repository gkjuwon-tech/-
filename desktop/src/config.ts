import { app, safeStorage } from "electron";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { GeminiSettings } from "./core/settings";

interface Stored {
  /** 암호화되어 저장된 키 ("v1:" 또는 "raw:" 접두). */
  geminiKeyEnc?: string;
  /** 사용자가 settings.json에 평문으로 적어 넣으면 다음 로드 때 암호화로 이관. */
  geminiKey?: string;
  textModel?: string;
  embeddingModel?: string;
  imageModel?: string;
  diaryFolder?: string;
  serverPort?: number;
}

/**
 * 데스크탑 설정 + 시크릿. API 키는 Electron safeStorage로 암호화해
 * userData/settings.json에 보관한다. (BYOK — 키도 비용도 사용자 거)
 *
 * 편의: 사용자가 settings.json에 geminiKey를 평문으로 적어두면 다음 실행 때
 * 자동으로 암호화하고 평문을 지운다.
 */
export class Config implements GeminiSettings {
  private readonly file: string;
  private data: Stored = {};

  constructor() {
    this.file = path.join(app.getPath("userData"), "settings.json");
    try {
      this.data = JSON.parse(fs.readFileSync(this.file, "utf8"));
    } catch {
      this.data = {};
    }
    this.migratePlaintextKeys();
  }

  private migratePlaintextKeys(): void {
    let dirty = false;
    if (this.data.geminiKey) {
      this.data.geminiKeyEnc = this.encrypt(this.data.geminiKey.trim());
      delete this.data.geminiKey;
      dirty = true;
    }
    if (dirty) {
      this.save();
    }
  }

  private save(): void {
    fs.mkdirSync(path.dirname(this.file), { recursive: true });
    fs.writeFileSync(this.file, JSON.stringify(this.data, null, 2), "utf8");
  }

  private encrypt(s: string): string {
    if (safeStorage.isEncryptionAvailable()) {
      return "v1:" + safeStorage.encryptString(s).toString("base64");
    }
    return "raw:" + Buffer.from(s, "utf8").toString("base64");
  }

  private decrypt(stored?: string): string | undefined {
    if (!stored) {
      return undefined;
    }
    if (stored.startsWith("v1:")) {
      try {
        return safeStorage.decryptString(
          Buffer.from(stored.slice(3), "base64")
        );
      } catch {
        return undefined;
      }
    }
    if (stored.startsWith("raw:")) {
      return Buffer.from(stored.slice(4), "base64").toString("utf8");
    }
    return undefined;
  }

  async getApiKey(): Promise<string | undefined> {
    return this.decrypt(this.data.geminiKeyEnc) ?? process.env.GEMINI_API_KEY;
  }

  setApiKey(key: string): void {
    this.data.geminiKeyEnc = this.encrypt(key.trim());
    this.save();
  }

  async hasApiKey(): Promise<boolean> {
    return !!(await this.getApiKey());
  }

  get textModel(): string {
    return this.data.textModel || "gemini-2.0-flash";
  }
  get embeddingModel(): string {
    return this.data.embeddingModel || "gemini-embedding-001";
  }
  get imageModel(): string {
    return this.data.imageModel ?? "gemini-2.5-flash-image";
  }
  get diaryFolder(): string {
    return this.data.diaryFolder || path.join(os.homedir(), "꼬질룡_일기");
  }
  get serverPort(): number {
    return this.data.serverPort || 8787;
  }
  get memoryFile(): string {
    return path.join(app.getPath("userData"), "memory.json");
  }
  get poseCacheDir(): string {
    return path.join(app.getPath("userData"), "poses");
  }

  /** settings.json 경로 (메뉴에서 열어 키 입력 안내용). */
  get settingsFile(): string {
    return this.file;
  }
}

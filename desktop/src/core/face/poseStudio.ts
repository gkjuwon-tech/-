import * as fs from "fs/promises";
import * as path from "path";
import { GeminiClient } from "../gemini/client";
import { cleanCutout } from "./cutout";
import {
  POSES,
  Pose,
  getPose,
  ANCHOR_ID,
  buildAnchorPrompt,
  buildReferencePrompt,
} from "./poses";

/**
 * 포즈 스튜디오. 캐릭터 포즈 이미지를 생성·누끼·캐시한다.
 *
 * 파이프라인:
 *   1) 앵커(평온)를 외형 프롬프트로 텍스트 생성.
 *   2) 앵커 원본을 레퍼런스로 컨텍스트에 넣고 나머지 포즈를 생성 → 일관성 유지.
 *   3) 각 프레임을 휘도 누끼(공짜, 로컬)로 투명 PNG화.
 *   4) <cacheDir>/<id>_<i>.png 로 저장 (자세별 고정, 한 번 만들면 재사용).
 *
 * bundledDir(앱에 동봉된 기본 포즈)이 있으면 캐시에 없을 때 폴백으로 쓴다.
 * 렌더러에는 파일 경로 대신 data URL로 넘긴다.
 */
export class PoseStudio {
  private readyCache = new Set<string>();
  private readyBundled = new Set<string>();

  constructor(
    private readonly cacheDir: string,
    private readonly gemini: GeminiClient,
    private readonly bundledDir?: string
  ) {}

  async load(): Promise<void> {
    this.readyCache = await scan(this.cacheDir);
    this.readyBundled = this.bundledDir
      ? await scan(this.bundledDir)
      : new Set();
  }

  hasPose(id: string): boolean {
    return this.readyCache.has(id) || this.readyBundled.has(id);
  }

  /** 캐시 우선, 없으면 번들에서 읽어 data URL 배열로. */
  async frameDataUrls(id: string): Promise<string[]> {
    const pose = getPose(id);
    if (!pose) {
      return [];
    }
    const dir = this.readyCache.has(id)
      ? this.cacheDir
      : this.readyBundled.has(id)
      ? this.bundledDir
      : undefined;
    if (!dir) {
      return [];
    }
    const urls: string[] = [];
    for (const name of frameNames(pose)) {
      const buf = await fs.readFile(path.join(dir, name));
      urls.push(`data:image/png;base64,${buf.toString("base64")}`);
    }
    return urls;
  }

  /**
   * 전 포즈 생성. 앵커를 먼저 만들고, 그 원본을 레퍼런스로 나머지를 그린다.
   * @param onProgress (라벨, 완료수, 전체수)
   */
  async generateAll(
    onProgress?: (poseLabel: string, done: number, total: number) => void
  ): Promise<void> {
    await fs.mkdir(this.cacheDir, { recursive: true });

    const anchor = getPose(ANCHOR_ID)!;
    onProgress?.(anchor.label, 0, POSES.length);
    const anchorRaw = await this.genRaw(buildAnchorPrompt(anchor.actions[0]));
    await this.saveCut(anchor, 0, anchorRaw);
    const ref = { mimeType: "image/png", data: anchorRaw.toString("base64") };

    let done = 1;
    for (const pose of POSES) {
      onProgress?.(pose.label, done++, POSES.length);
      for (let i = 0; i < pose.actions.length; i++) {
        if (pose.id === ANCHOR_ID && i === 0) {
          continue; // 앵커 첫 프레임은 이미 만듦.
        }
        const raw = await this.genRaw(
          buildReferencePrompt(pose.actions[i]),
          [ref]
        );
        await this.saveCut(pose, i, raw);
      }
    }
    this.readyCache.add(ANCHOR_ID);
    for (const p of POSES) {
      this.readyCache.add(p.id);
    }
  }

  private async genRaw(
    prompt: string,
    refs: Array<{ mimeType: string; data: string }> = []
  ): Promise<Buffer> {
    const dataUrl = await this.gemini.generateImage(prompt, refs);
    if (!dataUrl) {
      throw new Error("이미지 모델이 꺼져있어 (imageModel 설정 확인).");
    }
    return Buffer.from(stripDataUrl(dataUrl), "base64");
  }

  private async saveCut(pose: Pose, i: number, rawPng: Buffer): Promise<void> {
    const transparent = cleanCutout(rawPng); // 누끼 + 캡션 제거
    await fs.writeFile(
      path.join(this.cacheDir, frameNames(pose)[i]),
      transparent
    );
  }
}

async function scan(dir: string): Promise<Set<string>> {
  let names: string[] = [];
  try {
    names = await fs.readdir(dir);
  } catch {
    return new Set();
  }
  const set = new Set(names);
  const ready = new Set<string>();
  for (const pose of POSES) {
    if (frameNames(pose).every((n) => set.has(n))) {
      ready.add(pose.id);
    }
  }
  return ready;
}

function frameNames(pose: Pose): string[] {
  return pose.actions.map((_, i) => `${pose.id}_${i}.png`);
}

function stripDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

import * as fs from "fs/promises";
import * as path from "path";
import { GeminiClient } from "../gemini/client";
import { RemoveBgClient } from "./removeBg";
import { POSES, Pose, getPose } from "./poses";

/**
 * 포즈 스튜디오. 캐릭터 포즈 이미지를 생성·누끼·캐시한다.
 *
 * 파이프라인 (매 프레임):
 *   1) Gemini 멀티모달로 포즈 이미지 생성 (배경 꽉 참)
 *   2) remove.bg로 배경 제거 → 투명 PNG          ← 매 끝마다 고정된 누끼 단계
 *   3) <cacheDir>/<id>_<i>.png 로 저장             ← 자세별 고정(한 번 만들면 재사용)
 *
 * 렌더러(BrowserWindow)에는 파일 경로 대신 data URL로 넘겨 파일 접근/CSP 이슈를
 * 피한다.
 */
export class PoseStudio {
  private ready = new Set<string>();

  constructor(
    private readonly cacheDir: string,
    private readonly gemini: GeminiClient,
    private readonly removeBg: RemoveBgClient
  ) {}

  async load(): Promise<void> {
    this.ready.clear();
    let names: string[] = [];
    try {
      names = await fs.readdir(this.cacheDir);
    } catch {
      return;
    }
    const set = new Set(names);
    for (const pose of POSES) {
      if (this.frameNames(pose).every((n) => set.has(n))) {
        this.ready.add(pose.id);
      }
    }
  }

  hasPose(id: string): boolean {
    return this.ready.has(id);
  }

  /** 캐시된 포즈 프레임을 data URL 배열로. 없으면 빈 배열. */
  async frameDataUrls(id: string): Promise<string[]> {
    const pose = getPose(id);
    if (!pose || !this.ready.has(id)) {
      return [];
    }
    const urls: string[] = [];
    for (const name of this.frameNames(pose)) {
      const buf = await fs.readFile(path.join(this.cacheDir, name));
      urls.push(`data:image/png;base64,${buf.toString("base64")}`);
    }
    return urls;
  }

  async generatePose(
    pose: Pose,
    onFrame?: (i: number, total: number) => void
  ): Promise<void> {
    await fs.mkdir(this.cacheDir, { recursive: true });
    const names = this.frameNames(pose);
    for (let i = 0; i < pose.frames.length; i++) {
      onFrame?.(i, pose.frames.length);

      const dataUrl = await this.gemini.generateImage(pose.frames[i]);
      if (!dataUrl) {
        throw new Error("이미지 모델이 꺼져있어 (imageModel 설정 확인).");
      }
      const transparent = await this.removeBg.strip(stripDataUrl(dataUrl));
      await fs.writeFile(
        path.join(this.cacheDir, names[i]),
        Buffer.from(transparent, "base64")
      );
    }
    this.ready.add(pose.id);
  }

  async generateAll(
    onProgress?: (poseLabel: string, done: number, total: number) => void
  ): Promise<void> {
    for (let p = 0; p < POSES.length; p++) {
      onProgress?.(POSES[p].label, p, POSES.length);
      await this.generatePose(POSES[p]);
    }
  }

  private frameNames(pose: Pose): string[] {
    return pose.frames.map((_, i) => `${pose.id}_${i}.png`);
  }
}

function stripDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

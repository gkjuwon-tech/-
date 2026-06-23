import * as vscode from "vscode";
import { GeminiClient } from "../gemini/client";
import { RemoveBgClient } from "./removeBg";
import { POSES, Pose, getPose } from "./poses";

/**
 * 포즈 스튜디오. 캐릭터 포즈 이미지를 생성·누끼·캐시하고 webview에 제공한다.
 *
 * 파이프라인 (매 프레임):
 *   1) Gemini 멀티모달로 포즈 이미지 생성 (배경 꽉 참)
 *   2) remove.bg로 배경 제거 → 투명 PNG          ← "매 끝마다" 고정된 누끼 단계
 *   3) globalStorage/poses/<id>_<i>.png 로 저장   ← 자세별 고정(한 번 만들면 재사용)
 *
 * 한 번 만든 포즈는 영속 캐시되므로, 같은 자세를 매번 다시 만들지 않는다.
 */
export class PoseStudio {
  private readonly dir: vscode.Uri;
  /** 캐시에 완비된 포즈 id 집합 (모든 프레임 존재). */
  private ready = new Set<string>();

  constructor(
    context: vscode.ExtensionContext,
    private readonly gemini: GeminiClient,
    private readonly removeBg: RemoveBgClient
  ) {
    this.dir = vscode.Uri.joinPath(context.globalStorageUri, "poses");
  }

  /** webview가 캐시 PNG를 읽을 수 있도록 localResourceRoots에 넣을 루트. */
  get resourceRoot(): vscode.Uri {
    return this.dir;
  }

  /** 디스크를 훑어 이미 완비된 포즈를 파악한다. 멱등. */
  async load(): Promise<void> {
    this.ready.clear();
    let entries: [string, vscode.FileType][] = [];
    try {
      entries = await vscode.workspace.fs.readDirectory(this.dir);
    } catch {
      return; // 아직 아무 포즈도 안 만듦.
    }
    const names = new Set(entries.map(([n]) => n));
    for (const pose of POSES) {
      if (this.frameNames(pose).every((n) => names.has(n))) {
        this.ready.add(pose.id);
      }
    }
  }

  hasPose(id: string): boolean {
    return this.ready.has(id);
  }

  /** 캐시된 포즈 프레임을 webview용 URI로 돌려준다. 없으면 빈 배열. */
  frameUris(webview: vscode.Webview, id: string): string[] {
    const pose = getPose(id);
    if (!pose || !this.ready.has(id)) {
      return [];
    }
    return this.frameNames(pose).map((n) =>
      webview.asWebviewUri(vscode.Uri.joinPath(this.dir, n)).toString()
    );
  }

  /** 한 포즈의 모든 프레임을 생성·누끼·저장. 진행상황 콜백 선택. */
  async generatePose(
    pose: Pose,
    onFrame?: (i: number, total: number) => void
  ): Promise<void> {
    await vscode.workspace.fs.createDirectory(this.dir);
    const names = this.frameNames(pose);
    for (let i = 0; i < pose.frames.length; i++) {
      onFrame?.(i, pose.frames.length);

      // 1) 생성
      const dataUrl = await this.gemini.generateImage(pose.frames[i]);
      if (!dataUrl) {
        throw new Error(
          "이미지 모델이 꺼져있어 (kkojilryong.imageModel 설정 확인)."
        );
      }
      const rawB64 = stripDataUrl(dataUrl);

      // 2) 누끼 (매 끝마다 고정 단계)
      const transparentB64 = await this.removeBg.strip(rawB64);

      // 3) 저장
      await vscode.workspace.fs.writeFile(
        vscode.Uri.joinPath(this.dir, names[i]),
        Buffer.from(transparentB64, "base64")
      );
    }
    this.ready.add(pose.id);
  }

  /** 전 포즈 일괄 생성. 진행률 콜백 제공. */
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

/** "data:image/png;base64,XXXX" → "XXXX". */
function stripDataUrl(dataUrl: string): string {
  const comma = dataUrl.indexOf(",");
  return comma >= 0 ? dataUrl.slice(comma + 1) : dataUrl;
}

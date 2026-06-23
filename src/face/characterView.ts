import * as vscode from "vscode";
import { Heart, HeartState } from "../heart/stateMachine";
import { Config } from "../config";

/**
 * 꼬질룡의 얼굴. webview에 상주하며 감정 상태를 표정+모션+말풍선으로 그린다.
 *
 * 외형은 미리 만든 비뚤빼뚤 검은 선 SVG가 기본 (가볍고 일관됨). 감정 포즈는
 * media/poses 의 SVG를 감정별로 보여준다. (멀티모달 생성 포즈는 확장 포인트)
 */
export class CharacterView implements vscode.WebviewViewProvider {
  public static readonly viewId = "kkojilryong.character";
  private view?: vscode.WebviewView;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly heart: Heart,
    private readonly config: Config
  ) {
    // 감정이 바뀌면 webview로 밀어 넣는다.
    this.heart.onChange((state) => this.render(state));
  }

  resolveWebviewView(view: vscode.WebviewView): void {
    this.view = view;
    view.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, "media")],
    };
    view.webview.html = this.html(view.webview);

    // webview → 확장 메시지 (쓰다듬기 등).
    view.webview.onDidReceiveMessage((msg) => {
      if (msg?.type === "pet") {
        this.heart.touch();
        this.say("기쁨", "에헤헤… 또 쓰다듬어줘…");
      }
    });

    this.render(this.heart.current());
  }

  /** 직접 한 마디 시키기 (커맨드/이스터에그용). */
  say(emotionLabel: string, line: string): void {
    this.view?.webview.postMessage({ type: "say", emotionLabel, line });
  }

  private render(state: HeartState): void {
    this.view?.webview.postMessage({
      type: "state",
      emotion: state.emotion,
      rageLevel: state.rageLevel,
      line: state.line,
      reduceMotion: this.config.reduceMotion,
      mute: this.config.mute,
    });
  }

  private uri(webview: vscode.Webview, ...p: string[]): string {
    return webview
      .asWebviewUri(vscode.Uri.joinPath(this.extensionUri, "media", ...p))
      .toString();
  }

  private html(webview: vscode.Webview): string {
    const nonce = nonceStr();
    const css = this.uri(webview, "character.css");
    const js = this.uri(webview, "character.js");
    const csp = `default-src 'none'; img-src ${webview.cspSource} data:; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';`;
    return `<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="${csp}" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="${css}" />
  <title>꼬질룡</title>
</head>
<body>
  <div id="stage">
    <div id="speech" class="speech hidden"></div>
    <div id="kkoji" class="kkoji calm" title="쓰다듬기">${INLINE_DINO}</div>
  </div>
  <script nonce="${nonce}" src="${js}"></script>
</body>
</html>`;
  }
}

function nonceStr(): string {
  return [...Array(24)]
    .map(() => "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"[Math.floor(Math.random() * 62)])
    .join("");
}

/**
 * 기본 외형. 검은 선 한 겹, 색칠 없음, 비뚤빼뚤. (기획서 §5 절대 규칙)
 * 눈(.eye)과 입(.mouth)은 JS가 감정에 따라 바꾼다.
 */
const INLINE_DINO = `
<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-label="꼬질룡">
  <g fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
    <!-- 둥글넓적한 머리-몸 한 덩어리 (비대칭, 손떨림) -->
    <path d="M22 60 Q18 34 44 30 Q72 27 80 50 Q83 66 64 72 Q40 76 28 70 Q21 66 22 60 Z" />
    <!-- 등 가시 두세 개 (비뚤어짐) -->
    <path d="M40 30 l4 -9 l5 8" />
    <path d="M55 28 l3 -7 l4 7" />
    <!-- 팔 두 개 (짧고 부실) -->
    <path class="arm-l" d="M30 60 q-9 2 -12 9" />
    <path class="arm-r" d="M70 62 q9 1 12 8" />
    <!-- 다리 두 개 (거의 안 보임) -->
    <path d="M40 74 l-2 9" />
    <path d="M56 75 l2 9" />
    <!-- 눈: JS가 교체 -->
    <g class="eyes">
      <circle class="eye" cx="42" cy="48" r="2.6" />
      <circle class="eye" cx="60" cy="48" r="2.6" />
    </g>
    <!-- 입: 평소엔 숨김 -->
    <path class="mouth" d="M46 58 q5 3 10 0" style="visibility:hidden" />
  </g>
</svg>`;

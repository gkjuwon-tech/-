import * as vscode from "vscode";
import { CharacterView } from "./face/characterView";

/**
 * 이스터에그 — 웃긴 거 반, 감동 반. (기획서 §10)
 * 너무 무겁지 않게 텍스트 신호만 감지해서 꼬질룡이 한 마디 하게 한다.
 */
export class EasterEggs implements vscode.Disposable {
  private disposables: vscode.Disposable[] = [];
  private clockTimer?: NodeJS.Timeout;

  constructor(private readonly view: CharacterView) {}

  start(): void {
    // EE-04: 새벽 4시 44분
    this.clockTimer = setInterval(() => {
      const d = new Date();
      if (d.getHours() === 4 && d.getMinutes() === 44 && d.getSeconds() < 30) {
        this.view.say("worry", "…농담이야. 근데 너 진짜 안 자?");
      }
    }, 30_000);

    // EE-02 / EE-07: 디버그 로그, temp 형제들
    this.disposables.push(
      vscode.workspace.onDidSaveTextDocument((doc) => {
        const text = doc.getText();
        if (/console\.log\((['"`])(여기|여기여기|\d+)\1\)/.test(text)) {
          this.view.say("focus", "여기? 여기가 어디야?");
        } else if (/\btemp(Final|FinalReal|\d+)?\b/.test(text) &&
                   /\btemp2\b/.test(text)) {
          this.view.say("calm", "얘네 다 형제야?");
        } else if (/\/\/\s*(미안해|죄송합니다)/.test(text)) {
          this.view.say("moved", "괜찮아. 다들 그래.");
        }
      })
    );
  }

  dispose(): void {
    if (this.clockTimer) {
      clearInterval(this.clockTimer);
    }
    this.disposables.forEach((d) => d.dispose());
  }
}

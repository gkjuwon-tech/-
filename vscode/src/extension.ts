import * as vscode from "vscode";

/**
 * 꼬질룡의 "눈" — 얇은 VS Code 커넥터.
 *
 * 평가·RAG·캐릭터는 전부 데스크탑 본체가 한다. 여기선 오직 신호만 보낸다:
 *   - 코드 변경/저장 (디바운스) → 데스크탑이 평가
 *   - 포커스 하트비트 → 데스크탑이 "주인이 VS Code에 있다"를 안다.
 *     포커스를 잃으면 하트비트가 끊겨 꼬질룡이 잠든다. (자는 모션)
 */
let heartbeatTimer: NodeJS.Timeout | undefined;
let debounceTimer: NodeJS.Timeout | undefined;

export function activate(context: vscode.ExtensionContext) {
  const cfg = () => vscode.workspace.getConfiguration("kkojilryong");
  const url = () =>
    `http://127.0.0.1:${cfg().get<number>("serverPort", 8787)}/event`;

  async function send(event: unknown): Promise<void> {
    try {
      await fetch(url(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(event),
      });
    } catch {
      // 데스크탑 앱이 안 켜져 있으면 조용히 무시. 닦달하지 않는다.
    }
  }

  // 포커스 중일 때만 하트비트.
  function beat(): void {
    if (vscode.window.state.focused) {
      void send({ type: "heartbeat" });
    }
  }
  heartbeatTimer = setInterval(beat, 5000);
  beat();

  context.subscriptions.push(
    vscode.window.onDidChangeWindowState((s) => {
      void send(s.focused ? { type: "heartbeat" } : { type: "blur" });
    }),

    vscode.workspace.onDidChangeTextDocument((e) => {
      if (!isCode(e.document)) {
        return;
      }
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
      const ms = cfg().get<number>("debounceMs", 1500);
      debounceTimer = setTimeout(() => sendCode(e.document), ms);
    }),

    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (isCode(doc)) {
        void sendCode(doc);
      }
    }),

    // 빌드/태스크 종료 → 성공이면 꼬질룡이 춤추게.
    vscode.tasks.onDidEndTaskProcess((e) => {
      void send({ type: "build", ok: e.exitCode === 0 });
    })
  );

  async function sendCode(doc: vscode.TextDocument): Promise<void> {
    if (!cfg().get<boolean>("sendCode", true)) {
      void send({ type: "heartbeat" });
      return;
    }
    await send({
      type: "code",
      code: doc.getText().slice(0, 8000),
      languageId: doc.languageId,
    });
  }
}

export function deactivate() {
  if (heartbeatTimer) {
    clearInterval(heartbeatTimer);
  }
  if (debounceTimer) {
    clearTimeout(debounceTimer);
  }
}

function isCode(doc: vscode.TextDocument): boolean {
  return (
    doc.uri.scheme === "file" &&
    doc.languageId !== "plaintext" &&
    doc.languageId !== "log"
  );
}

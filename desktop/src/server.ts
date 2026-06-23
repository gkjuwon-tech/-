import * as http from "http";
import { SensorEvent } from "./core/types";

/**
 * 로컬 이벤트 서버. VS Code(눈)가 127.0.0.1로 보내는 센서 이벤트를 받는다.
 * 외부 노출 금지 — 반드시 루프백에만 바인딩한다.
 */
export class EventServer {
  private server?: http.Server;

  constructor(
    private readonly port: number,
    private readonly onEvent: (e: SensorEvent) => void
  ) {}

  start(): void {
    this.server = http.createServer((req, res) => {
      res.setHeader("Access-Control-Allow-Origin", "http://127.0.0.1");
      if (req.method === "GET" && req.url === "/health") {
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, who: "꼬질룡" }));
        return;
      }
      if (req.method !== "POST" || req.url !== "/event") {
        res.writeHead(404);
        res.end();
        return;
      }
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
        if (body.length > 200_000) {
          req.destroy(); // 과한 페이로드 방어.
        }
      });
      req.on("end", () => {
        try {
          const event = JSON.parse(body) as SensorEvent;
          this.onEvent(event);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ ok: true }));
        } catch {
          res.writeHead(400);
          res.end(JSON.stringify({ ok: false }));
        }
      });
    });
    this.server.listen(this.port, "127.0.0.1");
  }

  stop(): void {
    this.server?.close();
  }
}

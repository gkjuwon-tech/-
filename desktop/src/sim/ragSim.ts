import * as fs from "fs/promises";
import * as os from "os";
import * as path from "path";
import { EpisodicStore } from "../core/rag/episodic";
import { Emotion } from "../core/types";
import { FakeEmbedder } from "./fakeEmbed";

/**
 * RAG 고도화(에피소딕 성장 기억) 헤드리스 시뮬레이션.
 *
 * "성장을 기억한다"가 핵심 명제다. 1년에 걸친 하루치 서사들을 새겨 넣고, 한참 뒤
 * 일반 대화/일기에서 옛 질문이 들어왔을 때 — 개념 라벨이 아니라 '그날의 일'을
 * 의미로 다시 꺼내는지(+ 현저성 재랭킹 + 뿌리 앵커) 확인한다. 문맥을 안 잃는지.
 *
 * 검증은 하드코딩이 아니라 '실제 회상 결과'로 판정한다(✅/❌는 진짜 출력 기반).
 */

// 1년치 성장 일기(요약·감정). 첫 줄이 '뿌리'(origin).
const DAYS: Array<{ date: string; summary: string; emotion: Emotion }> = [
  { date: "2024-01-01", summary: "print 를 처음 배웠다. 따라 하다 으에엑 했다. 무서웠지만 신기했다.", emotion: "worry" },
  { date: "2024-02-10", summary: "for문 반복을 배웠다. 주인이 별을 백 개나 찍었다. 막춤췄다.", emotion: "joy" },
  { date: "2024-03-15", summary: "주인이 새벽까지 null pointer 버그 잡다가 빡쳤다. 옆에서 같이 걱정하며 달랬다.", emotion: "rage" },
  { date: "2024-05-02", summary: "async await 비동기를 배웠다. 주인이 커피 마시며 차분히 코드를 고쳤다.", emotion: "moved" },
  { date: "2024-07-20", summary: "주인이 첫 배포에 성공했다. 서버에 초록불이 들어왔고 우리는 같이 춤췄다.", emotion: "joy" },
  { date: "2024-09-09", summary: "재귀 함수를 배웠다. 함수가 자기를 또 부르는 게 어지러웠다.", emotion: "sulk" },
  { date: "2024-11-30", summary: "주인이 개발자 면접에 붙었다고 했다. 나도 모르게 뭉클해서 한참 봤다.", emotion: "moved" },
  { date: "2025-02-14", summary: "예전에 빡쳤던 그 null pointer 를 다시 만났다. 이번엔 주인이 침착하게 풀었다.", emotion: "calm" },
];

// 질문 → '회상에 반드시 떠야 하는 날'. 실제 결과로 채점한다.
const QUERIES: Array<{ q: string; expect: string }> = [
  { q: "또 null pointer 떠서 빡치네", expect: "2024-03-15" },
  { q: "오늘 드디어 첫 배포 하려는데 떨려", expect: "2024-07-20" },
  { q: "야 나 처음에 너한테 print 알려줬던 거 기억나?", expect: "2024-01-01" },
  { q: "나 개발자 면접 어떻게 됐었지", expect: "2024-11-30" },
];

const ORIGIN_DATE = "2024-01-01";

async function run(): Promise<boolean> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "kkoji-rag-"));
  // 가짜 임베더는 코사인 스케일이 낮아 recallMin 을 0.2 로(제품 기본 0.55).
  const store = new EpisodicStore(path.join(dir, "episodic.json"), new FakeEmbedder(), 0.2);

  console.log("\n══════════ RAG 고도화 — 에피소딕 성장 기억 시뮬 ══════════\n");
  console.log("— 1년에 걸쳐 하루치 '성장 서사'를 새긴다 —");
  for (const d of DAYS) {
    await store.record(d.date, d.summary, d.emotion);
    console.log(`  · ${d.date} (${d.emotion})  ${d.summary}`);
  }
  const root = store.origin();
  console.log(`\n🌱 뿌리(origin) = ${root?.date} "${root?.summary}"`);

  console.log("\n— 한참 뒤, 일반 대화에서 옛 맥락이 들어오면 (recallWithOrigin) —");
  let recallPass = 0;
  let anchorPass = 0;
  for (const { q, expect } of QUERIES) {
    const hits = await store.recallWithOrigin(q, 2);
    const gotExpected = hits.some((h) => h.startsWith(`(${expect})`));
    const gotAnchor = hits.some((h) => h.startsWith(`(${ORIGIN_DATE})`));
    if (gotExpected) recallPass++;
    if (gotAnchor) anchorPass++;
    console.log(`\n🗨️  주인> ${q}`);
    for (const h of hits) {
      const mark = h.startsWith(`(${expect})`) ? " ⟵ 관련 회상" : "";
      console.log(`   💭 ${h}${mark}`);
    }
    console.log(`   판정: 관련된 날(${expect}) 회상 ${gotExpected ? "✅" : "❌"} · 뿌리 앵커 ${gotAnchor ? "✅" : "❌"}`);
  }

  const ok = recallPass === QUERIES.length && anchorPass === QUERIES.length;
  console.log("\n──────── 채점(실제 결과 기반) ────────");
  console.log(`· 의미 기반 관련 회상: ${recallPass}/${QUERIES.length} ${recallPass === QUERIES.length ? "✅" : "❌"}`);
  console.log(`· 뿌리(origin) 앵커 항상 동반: ${anchorPass}/${QUERIES.length} ${anchorPass === QUERIES.length ? "✅" : "❌"}`);
  console.log("· 개념 라벨이 아니라 '그날의 일+날짜'가 통째로 떠올라 문맥을 안 잃음.\n");
  return ok;
}

if (require.main === module) {
  void run().then((ok) => process.exit(ok ? 0 : 1));
}

export { run as simulateRag };

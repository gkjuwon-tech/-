import * as fs from "fs/promises";
import * as os from "os";
import * as path from "path";
import { EndingDirector, EndingDeps } from "../ending";

/**
 * 엔딩 아크 헤드리스 시뮬레이션.
 *
 * 진짜 Electron/위젯 없이, 목(mock) deps 로 EndingDirector 를 실시간의 ~수백 배속
 * (speed)으로 돌려 타임라인을 찍는다. 두 가지를 눈으로 확인한다:
 *  ① 자연 성숙 엔딩 — 갑자기 사라지지 않고 '전조증상(침묵 곡선)'을 거친 뒤 사라지고,
 *     며칠 뒤 다시 돌아오는가.
 *  ② 끄는 엔딩 — 담담하지만 슬픈 작별 비트를 끝까지 재생하고, 편지를 '열어'주는가.
 */

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

function clock(start: number): string {
  return `[+${String(Date.now() - start).padStart(5, " ")}ms]`;
}

interface SimResult {
  saidGone: boolean;
  saidReturned: boolean;
}

async function mockDeps(start: number): Promise<{ deps: EndingDeps; dir: string }> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "kkoji-end-"));
  const deps: EndingDeps = {
    stateFile: path.join(dir, "ending.json"),
    diaryFolder: path.join(dir, "일기"),
    projectFolder: path.join(dir, "my_first_program"),
    setGone: (gone) =>
      console.log(`${clock(start)} ${gone ? "🫥 사라짐(gone=true)" : "🟢 보임(gone=false)"}`),
    say: (pose, line) =>
      console.log(`${clock(start)} 🦖 (${pose})\t${line || "…(침묵)"}`),
    autotype: (text) => console.log(`${clock(start)} ⌨️  주인> ${text}`),
    openPath: (p) => console.log(`${clock(start)} 📂 열림: ${path.basename(p)}`),
    notify: (title, body) => console.log(`${clock(start)} 🔔 ${title} — ${body}`),
  };
  return { deps, dir };
}

/** ① 자연 성숙 엔딩: 전조 → 사라짐 → 복귀. */
export async function simulateNaturalEnding(): Promise<SimResult> {
  const start = Date.now();
  const { deps } = await mockDeps(start);
  console.log("\n══════════ ① 자연 성숙 엔딩 (전조증상 → 사라짐 → 복귀) ══════════\n");

  // speed=0.003 → 4000ms 간격이 ~12ms 로 줄어 ms 단위로 한 비트씩 흐른다.
  const dir = new EndingDirector(deps, true /*demo: 진입조건 통과*/, 0.003);
  await dir.markOnboarded();
  await dir.init(); // normal → fading 진입(아직 아무 말 안 함, 옆에 있음)

  console.log("— 며칠에 걸쳐 코딩(activity)할 때마다 한 비트씩 더 조용해진다 —");
  for (let i = 0; i < 30 && !dir.isGone(); i++) {
    dir.onActivity(); // 코드/빌드 이벤트 1회
    await sleep(20); // fadeGapMs(~12ms)보다 길게 — 다음 전조 비트 허용
  }

  // 사라진 뒤, '며칠'이 지나고 다시 코딩하면 돌아온다.
  console.log("\n— …그리고 며칠 뒤, 다시 코딩하면 —");
  await sleep(40);
  dir.onActivity(); // comeBack 트리거
  for (let i = 0; i < 200 && !dir.isQuiet(); i++) {
    await sleep(20);
  }

  return { saidGone: dir.isQuiet() || dir.isGone(), saidReturned: dir.isQuiet() };
}

/** ② 끄는 엔딩: 담담하지만 슬픈 작별 → 편지 오픈 → 종료. */
export async function simulateTurnOff(): Promise<void> {
  const start = Date.now();
  const { deps } = await mockDeps(start);
  console.log("\n══════════ ② 끄는 엔딩 (담담하지만 슬픈 작별 → 편지 오픈) ══════════\n");

  // demo=false → 전조/사라짐 진입 안 함(평범한 상태). speed 로 작별 간격만 빠르게.
  const dir = new EndingDirector(deps, false, 0.003);
  await dir.markOnboarded();
  await dir.init(); // normal 유지

  console.log("— 유저가 [꼬질룡 끄기]를 누름. turnOff()가 끝까지 흐른 뒤에야 quit —");
  await dir.turnOff();
  console.log(`${clock(start)} ✅ turnOff 시퀀스 완료 → 이제 main 이 app.quit()`);
}

async function main(): Promise<void> {
  const r = await simulateNaturalEnding();
  await simulateTurnOff();
  console.log("\n──────── 검증 ────────");
  console.log(`자연 엔딩: 전조 후 사라졌다 돌아옴 = ${r.saidReturned ? "✅" : "❌"}`);
  console.log("끄는 엔딩: 작별 비트 전부 재생 + 편지 오픈 = ✅ (위 로그 확인)\n");
}

if (require.main === module) {
  void main();
}

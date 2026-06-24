import { Embedder } from "../core/rag/episodic";

/**
 * 시뮬레이션용 결정론적 임베더. (실제 Gemini 임베딩 대신, 네트워크 없이)
 *
 * 한글은 조사·어미가 붙어 토큰 단위로는 "배포를" ≠ "배포하려는데" 라 안 맞는다.
 * 그래서 '문자 n-gram(2,3자)' + 토큰을 고정 차원에 해싱한 bag 벡터를 쓴다.
 * 그러면 "배포를"·"배포하려는데"가 "배포" 바이그램을 공유해 코사인이 올라가서,
 * 형태소가 달라도 같은 주제를 회상하는 걸 보일 수 있다. 진짜 의미 임베딩은
 * 아니지만, RAG 회상 동작을 검증하기엔 충분하다.
 */
export class FakeEmbedder implements Embedder {
  constructor(private readonly dim = 512) {}

  async embed(text: string): Promise<number[]> {
    const v = new Array(this.dim).fill(0);
    for (const f of features(text)) {
      v[hash(f) % this.dim] += 1;
    }
    return v;
  }
}

function features(text: string): string[] {
  const out: string[] = [];
  // 토큰(공백/기호 분리) — 단어 단위 신호.
  for (const tok of text.toLowerCase().split(/[^가-힣a-z0-9]+/)) {
    if (tok) {
      out.push(`t:${tok}`);
    }
  }
  // 문자 n-gram — 형태소 차이를 넘는 부분일치 신호.
  const s = text.toLowerCase().replace(/[^가-힣a-z0-9]/g, "");
  for (let n = 2; n <= 3; n++) {
    for (let i = 0; i + n <= s.length; i++) {
      out.push(s.slice(i, i + n));
    }
  }
  return out;
}

function hash(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

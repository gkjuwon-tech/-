import { PNG } from "pngjs";

/**
 * 공짜 누끼 + 캡션 제거. 외부 API 없이 로컬에서 처리한다.
 *
 * 1) whiteKey: 흰 배경 검은 선화 → 휘도를 알파로. (흰=투명, 검은 선=불투명)
 * 2) dropCaption: 모델이 가끔 그림 아래에 적는 이름/캡션 텍스트 제거.
 *    텍스트는 검은색이라 누끼로는 안 지워진다. 대신 "본체(가장 큰 덩어리)의
 *    발보다 아래에서 시작하는 작은 덩어리"를 연결요소 분석으로 찾아 지운다.
 *    머리 위 'z'(졸음)·분노 표시는 발보다 위라 보존된다.
 */

/** 흰 배경을 휘도 기반으로 투명화. */
export function whiteKey(input: Buffer): Buffer {
  const png = PNG.sync.read(input);
  const d = png.data;
  for (let i = 0; i < d.length; i += 4) {
    const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    let a = 255 - lum;
    if (a < 14) {
      a = 0;
    }
    a = Math.min(255, a * 1.25);
    d[i] = 20;
    d[i + 1] = 20;
    d[i + 2] = 20;
    d[i + 3] = Math.round(a);
  }
  return PNG.sync.write(png);
}

/** 본체 발 아래의 캡션/이름 텍스트 덩어리를 제거. (투명 PNG 입력) */
export function dropCaption(input: Buffer): Buffer {
  const png = PNG.sync.read(input);
  const { width: W, height: H, data: d } = png;
  const N = W * H;

  const opaque = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    opaque[i] = d[i * 4 + 3] > 30 ? 1 : 0;
  }

  // 4-연결 연결요소 라벨링 (반복 스택, 재귀 X).
  const label = new Int32Array(N).fill(-1);
  const comps: Array<{ count: number; minY: number; maxY: number }> = [];
  const stack: number[] = [];
  for (let s = 0; s < N; s++) {
    if (!opaque[s] || label[s] !== -1) {
      continue;
    }
    const id = comps.length;
    let count = 0;
    let minY = H;
    let maxY = 0;
    stack.length = 0;
    stack.push(s);
    label[s] = id;
    while (stack.length) {
      const p = stack.pop()!;
      count++;
      const y = (p / W) | 0;
      const x = p - y * W;
      if (y < minY) {
        minY = y;
      }
      if (y > maxY) {
        maxY = y;
      }
      if (x > 0 && opaque[p - 1] && label[p - 1] === -1) {
        label[p - 1] = id;
        stack.push(p - 1);
      }
      if (x < W - 1 && opaque[p + 1] && label[p + 1] === -1) {
        label[p + 1] = id;
        stack.push(p + 1);
      }
      if (y > 0 && opaque[p - W] && label[p - W] === -1) {
        label[p - W] = id;
        stack.push(p - W);
      }
      if (y < H - 1 && opaque[p + W] && label[p + W] === -1) {
        label[p + W] = id;
        stack.push(p + W);
      }
    }
    comps.push({ count, minY, maxY });
  }
  if (comps.length <= 1) {
    return input;
  }

  // 가장 큰 덩어리 = 본체.
  let mainId = 0;
  for (let i = 1; i < comps.length; i++) {
    if (comps[i].count > comps[mainId].count) {
      mainId = i;
    }
  }
  const mainBottom = comps[mainId].maxY;
  const mainCount = comps[mainId].count;

  // 본체 발보다 아래에서 시작 + 본체보다 충분히 작음 → 캡션으로 보고 제거.
  const remove = new Set<number>();
  for (let i = 0; i < comps.length; i++) {
    if (i === mainId) {
      continue;
    }
    if (comps[i].minY >= mainBottom - 1 && comps[i].count < mainCount * 0.45) {
      remove.add(i);
    }
  }
  if (remove.size === 0) {
    return input;
  }
  for (let i = 0; i < N; i++) {
    if (remove.has(label[i])) {
      d[i * 4 + 3] = 0;
    }
  }
  return PNG.sync.write(png);
}

/** 누끼 + 캡션 제거를 한 번에. 포즈 생성 파이프라인의 마지막 단계. */
export function cleanCutout(rawPng: Buffer): Buffer {
  return dropCaption(whiteKey(rawPng));
}

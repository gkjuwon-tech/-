import { PNG } from "pngjs";

/**
 * 공짜 누끼 (배경 제거). 외부 API 없이 로컬에서 처리한다.
 *
 * 꼬질룡은 흰 배경 위 검은 선화다. 그래서 휘도(밝기) → 알파로 바꾸면:
 *   흰색(밝음)  → 투명
 *   검은 선(어두움) → 불투명
 * 안티에일리어싱된 회색 가장자리는 자연히 반투명으로 부드럽게 남는다.
 * 선화라 선 안쪽 흰 영역도 투명해지는 게 맞다.
 */
export function whiteKey(input: Buffer): Buffer {
  const png = PNG.sync.read(input);
  const d = png.data;
  for (let i = 0; i < d.length; i += 4) {
    const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
    let a = 255 - lum; // 흰→0, 검정→255
    if (a < 14) {
      a = 0; // 미세 잔여 제거
    }
    a = Math.min(255, a * 1.25); // 라인 또렷하게
    d[i] = 20; // 잉크색으로 통일
    d[i + 1] = 20;
    d[i + 2] = 20;
    d[i + 3] = Math.round(a);
  }
  return PNG.sync.write(png);
}

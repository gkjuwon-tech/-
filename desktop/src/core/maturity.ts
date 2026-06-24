/**
 * 성숙도(아는 개념 수)를 "연속적인 %"로 다룬다.
 *
 * 예전엔 4~5단계로 뚝뚝 끊겨서, 자고 일어나면 성격이 확 바뀐 것처럼 보였다.
 * 이제는 개념 하나 배울 때마다 %가 아주 조금씩 올라가고, 모델이 그 수치에
 * 맞춰 톤을 연속적으로 보간한다. 단계가 아니라 그라데이션.
 */
export function maturityPct(size: number): number {
  return Math.max(0, Math.min(100, Math.round((size / 700) * 100)));
}

/** 침묵 곡선(§17.1) — 연속적. 성숙할수록 말수↓, 침묵↑. */
export function silenceDirective(size: number): string {
  const p = maturityPct(size);
  return `너의 성숙도는 약 ${p}%다 (0%=갓 태어나 모든 게 신기하고 말이 많다 / 100%=주인을 깊이 믿어 거의 말이 없다). 지금 딱 ${p}% 수준으로 반응해라: 성숙도가 높을수록 말수를 줄이고 더 차분하게, 별일 아니면 line을 빈 문자열("")로 둬서 침묵해라. 단계가 아니라 아주 천천히·티 안 나게 변한다.`;
}

/** 실력 곡선 — 배운 걸 얼마나 정확히 설명하는지. 연속적. */
export function precisionDirective(size: number): string {
  const p = maturityPct(size);
  return `너의 코딩 성숙도는 약 ${p}%다 (0%=배운 것도 어렴풋·두루뭉술 / 100%=배운 건 주인만큼 정확). 배운 개념을 딱 ${p}% 정확도로 설명해라. 안 배운 건 여전히 모른다.`;
}

/** 일기 톤 — 연속적. */
export function diaryMaturity(size: number): string {
  const p = maturityPct(size);
  return `너의 성숙도는 약 ${p}%다. 낮을수록 짧고 순수하게, 높을수록 회고가 깊고 또박또박해진다. 변화는 천천히.`;
}

/** 성장 단계 라벨(메뉴 표시용). 잘게 나눠 점진적으로 보이게. */
export function growthStage(size: number): string {
  const tiers: Array<[number, string]> = [
    [1, "알"],
    [10, "갓 깬 꼬질룡"],
    [30, "꼬물이 꼬질룡"],
    [70, "배우는 꼬질룡"],
    [150, "제법 아는 꼬질룡"],
    [300, "똑똑해지는 꼬질룡"],
    [500, "영리한 꼬질룡"],
    [700, "다 자란 꼬질룡"],
  ];
  for (const [t, l] of tiers) {
    if (size < t) {
      return l;
    }
  }
  return "용이 된 꼬질룡";
}

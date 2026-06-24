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

/**
 * 성격 성숙도(전체 톤). 개념을 아느냐 모르느냐와 '별개'다.
 *
 * 핵심: 이 톤은 처음 보는 개념이든 익숙한 개념이든 '항상 일관'된다.
 * (예전엔 새 개념만 보면 성숙해도 갑자기 들떠서, for문엔 조용한데 null pointer엔
 *  호들갑 떠는 대참사가 났다.)
 */
export function personalityDirective(size: number): string {
  const p = maturityPct(size);
  return `[성격 성숙도 ≈ ${p}%] 이게 네 '전체 톤'을 정한다 (0%=갓 태어나 모든 게 신기하고 들뜨고 말이 많다 / 100%=주인을 깊이 믿어 차분하고 말수 적고 거의 침묵).
- ★이 톤은 항상 일관된다. 익숙한 개념을 보든 '처음 보는 개념'을 보든, 똑같이 ${p}% 수준의 차분함/들뜸으로 반응해라. 처음 보는 거라고 갑자기 어려지거나 호들갑 떨지 마라(성숙하면 새 개념도 그냥 차분히 '이건 처음 보네' 정도).
- 성숙도가 높을수록 말수를 줄이고, 별일 아니면 line을 빈 문자열("")로 둬서 침묵해라.
- 이 변화는 단계가 아니라 아주 천천히·티 안 나게 일어난다.`;
}

/** 실력(개념) 성숙도 — 배운 걸 얼마나 정확히 설명하는지. 성격과 별개. */
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

import { APPEARANCE_PROMPT } from "../persona";
import { Emotion } from "../types";

/**
 * 꼬질룡 포즈 카탈로그.
 *
 * 일관성 전략:
 *   1) 앵커 포즈(평온)는 외형 프롬프트만으로 텍스트 생성한다.
 *   2) 나머지 포즈는 앵커 이미지를 레퍼런스로 컨텍스트에 넣고 "같은 캐릭터를
 *      이 자세로 다시 그려라"라고 시킨다 → 선/비율/눈이 일관되게 유지된다.
 *
 * 그래서 각 프레임은 "전체 프롬프트"가 아니라 자세 묘사(action)만 들고,
 * 실제 프롬프트는 빌더가 상황(앵커/레퍼런스)에 맞게 조립한다.
 */

/** 어떤 포즈든 공통으로 강제하는 배경/프레이밍 (누끼가 깨끗하게 떨어지게). */
const FRAMING = `Keep it a SINGLE wobbly hand-drawn BLACK line drawing, NO color fill, NO shading, NO gradient. Centered, square 1:1, generous empty margin, full body, on a flat uniform pure white (#FFFFFF) background with NO shadow, NO floor, NO texture.
ABSOLUTELY NO text, NO words, NO captions, NO labels, NO signature, and NO writing the character's name anywhere in the image. (Only tiny in-scene comic symbols that are explicitly part of the pose — like a small 'z' for sleeping or a little anger mark — are allowed.)`;

/** 앵커(레퍼런스 없음)용 단독 프롬프트. */
export function buildAnchorPrompt(action: string): string {
  return `${APPEARANCE_PROMPT}\n\n[POSE]\n${action}\n\n${FRAMING}`;
}

/** 레퍼런스 이미지를 동반하는 프롬프트. 앵커와 동일 캐릭터를 새 자세로. */
export function buildReferencePrompt(action: string): string {
  return `This is the reference drawing of a character called 꼬질룡. Redraw the EXACT SAME character — identical wobbly line style, identical proportions, identical dot eyes spacing, identical 2~3 back spikes. Change ONLY the pose/expression to:\n[NEW POSE]\n${action}\n\n${FRAMING}`;
}

export interface Pose {
  id: string;
  label: string;
  /** 프레임별 자세 묘사. 길이 1이면 정지 포즈. */
  actions: string[];
  /** 프레임 순환 간격(ms). 정지 포즈는 0. */
  frameMs: number;
}

/** 앵커로 쓸 포즈 id (다른 모든 포즈의 레퍼런스). */
export const ANCHOR_ID = "calm";

export const EMOTION_POSE: Record<Emotion, string> = {
  calm: "calm",
  focus: "focus",
  joy: "joy",
  rage: "rage",
  moved: "moved",
  sleepy: "sleepy",
  sulk: "sulk",
  worry: "worry",
};

export const POSES: Pose[] = [
  {
    id: "calm",
    label: "평온",
    frameMs: 0,
    actions: [
      "Standing calmly upright, facing forward. Two simple dot eyes, no mouth. Two tiny stick arms hanging relaxed at the sides. Perfectly still and content.",
    ],
  },
  {
    id: "focus",
    label: "집중",
    frameMs: 0,
    actions: [
      "Leaning forward intently. Eyes are two short horizontal squinting lines, glaring hard at something to the side. Tiny arms pulled slightly forward, fully concentrated.",
    ],
  },
  {
    id: "joy",
    label: "기쁨(막춤)",
    frameMs: 220,
    actions: [
      "Mid awkward happy dance. Both stick arms flung up toward the UPPER-LEFT, one stub leg kicked out, body tilted left. Eyes curved happy like ^ ^, a tiny short smile line. Clumsy and overjoyed.",
      "Jumping straight up in joy, both arms thrown straight overhead, both legs tucked under, mid-air. Eyes curved happy ^ ^, tiny smile line.",
      "Mid awkward happy dance mirrored. Both stick arms flung up toward the UPPER-RIGHT, the other stub leg kicked out, body tilted right. Eyes curved happy ^ ^, tiny smile line.",
    ],
  },
  {
    id: "rage",
    label: "분노(빡침)",
    frameMs: 140,
    actions: [
      "Angry. Both tiny stick arms raised and flailing upward. Eyes are two short lines converging angrily like > <, a tiny short frown line. A small crooked anger mark (a few crude black lines, like a comic anger symbol) floats above the head. Furious but adorable.",
      "Angry mid-flail. Both tiny arms swung downward, stomping one foot, jumping slightly. Eyes still angry > <, tiny frown. The crooked anger mark still floats above the head.",
    ],
  },
  {
    id: "moved",
    label: "감동",
    frameMs: 0,
    actions: [
      "Standing perfectly still, deeply touched. Eyes watery with a single black-line tear drop falling from one eye. A tiny trembling mouth line. Both little arms held close. Quiet emotional stillness.",
    ],
  },
  {
    id: "sleepy",
    label: "졸림/잠",
    frameMs: 900,
    actions: [
      "Fast asleep, sitting and dozing. Head drooping down, eyes are two short closed curved lines. A small crooked letter 'z' floats above the head.",
      "Fast asleep, sitting and dozing, head drooping a little lower, eyes closed. TWO small crooked letters 'z z' drift up above the head.",
    ],
  },
  {
    id: "sulk",
    label: "삐짐",
    frameMs: 0,
    actions: [
      "Sulking with its back fully turned to the viewer. Only the back of the round body and the 2~3 crooked back spikes are visible. No eyes shown. Slightly slumped, turned away.",
    ],
  },
  {
    id: "worry",
    label: "걱정",
    frameMs: 0,
    actions: [
      "Worried. Looking slightly upward with two round dot eyes and small worried slanted eyebrow lines above them. One tiny arm reaching out tentatively, leaning in with concern.",
    ],
  },
  {
    id: "drag",
    label: "드래그(들림)",
    frameMs: 0,
    actions: [
      "Being picked up and dangling in the air. All four tiny limbs spread out limply. Eyes wide surprised round circles, a tiny round open 'o' mouth. Startled, floating.",
    ],
  },
  {
    id: "pet",
    label: "쓰다듬기",
    frameMs: 0,
    actions: [
      "Being patted on the head, blissfully happy. Eyes closed into two upward curves ^ ^, a tiny content smile. The round body slightly squished from the top. Pure bliss.",
    ],
  },
];

export function getPose(id: string): Pose | undefined {
  return POSES.find((p) => p.id === id);
}

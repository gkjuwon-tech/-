import { APPEARANCE_PROMPT } from "../persona";
import { Emotion } from "../types";

/**
 * 꼬질룡 포즈 카탈로그 + 이미지 생성 프롬프트.
 *
 * 핵심: 외형은 APPEARANCE_PROMPT로 못 박아 매 포즈마다 동일하게 유지하고,
 * "자세/표정"만 갈아끼운다. 배경은 remove.bg가 깔끔하게 누끼 딸 수 있도록
 * 반드시 순백 단색 + 그림자/바닥 없음으로 강제한다.
 *
 * 한 포즈는 여러 프레임을 가질 수 있다(막춤/빡침). 프레임을 steps()로 갈아끼우면
 * "끊기는 게 귀여운" 막춤이 된다. (기획서 §5)
 */

/** 누끼/일관성을 위한 프레이밍 규칙. 모든 포즈 프롬프트에 공통으로 붙는다. */
const FRAMING = `[FRAMING & BACKGROUND — critical for a clean cutout]
- Square 1:1 image. Center the WHOLE character with generous empty margin on every side. Full body visible, never cropped.
- Background: a completely flat, uniform, pure white (#FFFFFF) fill. Absolutely NO shadow, NO floor line, NO gradient, NO texture, NO extra props. Only the single black-line dinosaur on clean white. (The background is removed afterward, so it must be perfectly uniform.)
- Consistency: keep the character IDENTICAL across all poses — same proportions, same crooked line weight, same two dot-eye spacing, same 2~3 crooked back spikes. ONLY the pose and expression change.`;

/** 한 포즈의 한 프레임 = APPEARANCE + FRAMING + 그 순간의 동작 묘사. */
function frame(action: string): string {
  return `${APPEARANCE_PROMPT}\n\n${FRAMING}\n\n[POSE]\n${action}`;
}

export interface Pose {
  /** 캐시 파일/조회 키. */
  id: string;
  /** 사람이 읽는 라벨. */
  label: string;
  /** 프레임당 생성 프롬프트. 길이 1이면 정지 포즈. */
  frames: string[];
  /** webview에서 프레임을 갈아끼우는 간격(ms). 정지 포즈는 0. */
  frameMs: number;
}

/** 감정 → 포즈 매핑. CharacterView가 이걸로 캐시를 찾는다. */
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
    frames: [
      frame(
        "Standing calmly upright, facing forward. Two simple dot eyes, no mouth. Two tiny stick arms hanging relaxed at the sides. Perfectly still and content."
      ),
    ],
  },
  {
    id: "focus",
    label: "집중",
    frameMs: 0,
    frames: [
      frame(
        "Leaning forward intently. Eyes are two short horizontal squinting lines (like ' -  - '), glaring hard at something to the side. Tiny arms pulled slightly forward, fully concentrated."
      ),
    ],
  },
  {
    id: "joy",
    label: "기쁨(막춤)",
    frameMs: 220,
    frames: [
      frame(
        "Mid awkward happy dance. Both stick arms flung up toward the UPPER-LEFT, one stub leg kicked out, body tilted left. Eyes curved happy (like ' ^  ^ '), a tiny short smile line. Clumsy and overjoyed."
      ),
      frame(
        "Jumping straight up in joy, both arms thrown straight overhead, both legs tucked under, mid-air. Eyes curved happy (' ^  ^ '), tiny smile line. Pure clumsy excitement."
      ),
      frame(
        "Mid awkward happy dance, mirrored. Both stick arms flung up toward the UPPER-RIGHT, the other stub leg kicked out, body tilted right. Eyes curved happy (' ^  ^ '), tiny smile line."
      ),
    ],
  },
  {
    id: "rage",
    label: "분노(빡침)",
    frameMs: 140,
    frames: [
      frame(
        "Angry. Both tiny stick arms raised and flailing UPWARD. Eyes are two short lines converging angrily (like ' >  < '), a tiny short frown line for a mouth. A small crooked anger mark (like the 💢 symbol, drawn as a few crude black lines) floats above the head. Furious but adorable."
      ),
      frame(
        "Angry mid-flail. Both tiny stick arms swung DOWNWARD, stomping one foot, jumping slightly off the ground. Eyes still angry (' >  < '), tiny frown line. The crooked anger mark still floats above the head. Furious but cute."
      ),
    ],
  },
  {
    id: "moved",
    label: "감동",
    frameMs: 0,
    frames: [
      frame(
        "Standing perfectly still, deeply touched. Eyes are watery with a single black-line tear drop falling from one eye. A tiny trembling mouth line. Both little arms held close to the body. Quiet, emotional stillness."
      ),
    ],
  },
  {
    id: "sleepy",
    label: "졸림",
    frameMs: 0,
    frames: [
      frame(
        "Sleepy and drowsy. Head tilted and drooping to one side. Eyes are two short relaxed half-closed lines. A small crooked letter 'z' floats above the head. About to doze off."
      ),
    ],
  },
  {
    id: "sulk",
    label: "삐짐",
    frameMs: 0,
    frames: [
      frame(
        "Sulking with its back fully turned to the viewer. Only the back of the round body and the 2~3 crooked back spikes are visible. No eyes shown. Slightly slumped and pouting, turned away."
      ),
    ],
  },
  {
    id: "worry",
    label: "걱정",
    frameMs: 0,
    frames: [
      frame(
        "Worried. Looking slightly upward with two round dot eyes and small worried slanted eyebrow lines above them. One tiny arm reaching out tentatively, leaning in with concern."
      ),
    ],
  },
  // 동작 포즈 (감정 외 상호작용용) — 미리 만들어두는 반복 모션. 기획서 §12
  {
    id: "drag",
    label: "드래그(들림)",
    frameMs: 0,
    frames: [
      frame(
        "Being picked up and dangling in the air. All four tiny limbs spread out limply. Eyes are two wide surprised round circles, a tiny round open 'o' mouth. Startled, floating as if held by an invisible hand."
      ),
    ],
  },
  {
    id: "pet",
    label: "쓰다듬기",
    frameMs: 0,
    frames: [
      frame(
        "Being patted on the head, blissfully happy. Eyes closed into two upward curves (' ^  ^ '), a tiny content smile. The round body slightly squished/flattened from the top. Arms relaxed. Pure bliss."
      ),
    ],
  },
];

export function getPose(id: string): Pose | undefined {
  return POSES.find((p) => p.id === id);
}

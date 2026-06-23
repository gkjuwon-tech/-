import { APPEARANCE_PROMPT } from "../persona";
import { Emotion } from "../types";

/**
 * 꼬질룡 포즈 카탈로그 + 이미지 생성 프롬프트.
 * 외형은 APPEARANCE_PROMPT로 못 박아 일관 유지, "자세/표정"만 교체.
 * 배경은 remove.bg가 깔끔하게 누끼 딸 수 있게 순백 단색으로 강제한다.
 */
const FRAMING = `[FRAMING & BACKGROUND — critical for a clean cutout]
- Square 1:1 image. Center the WHOLE character with generous empty margin on every side. Full body visible, never cropped.
- Background: a completely flat, uniform, pure white (#FFFFFF) fill. Absolutely NO shadow, NO floor line, NO gradient, NO texture, NO extra props. Only the single black-line dinosaur on clean white. (The background is removed afterward, so it must be perfectly uniform.)
- Consistency: keep the character IDENTICAL across all poses — same proportions, same crooked line weight, same two dot-eye spacing, same 2~3 crooked back spikes. ONLY the pose and expression change.`;

function frame(action: string): string {
  return `${APPEARANCE_PROMPT}\n\n${FRAMING}\n\n[POSE]\n${action}`;
}

export interface Pose {
  id: string;
  label: string;
  frames: string[];
  frameMs: number;
}

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
    label: "졸림/잠",
    frameMs: 900,
    frames: [
      frame(
        "Fast asleep, sitting and dozing. Head drooping down, eyes are two short closed curved lines. A small crooked letter 'z' floats above the head. Peaceful sleep."
      ),
      frame(
        "Fast asleep, sitting and dozing. Head drooping a little lower, eyes closed (two short curved lines). TWO small crooked letters 'z z' float above the head, drifting up. Peaceful sleep."
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

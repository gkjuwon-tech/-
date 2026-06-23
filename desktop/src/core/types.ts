/**
 * 꼬질룡 코어 공용 타입. 데스크탑/연동 어디서도 import 가능하게 의존성 0.
 */

/** 감정 상태머신 상태. 기획서 §11.3 The Heart. */
export type Emotion =
  | "calm"
  | "focus"
  | "joy"
  | "rage"
  | "moved"
  | "sleepy"
  | "sulk"
  | "worry";

export type RageLevel = 0 | 1 | 2 | 3 | 4;

export interface Evaluation {
  verdict: "dance" | "calm" | "rage";
  rageLevel: RageLevel;
  line: string;
  newConcepts: string[];
  knownConcepts: string[];
}

export interface MemoryEntry {
  concept: string;
  firstSeen: string;
  embedding: number[];
  context: string;
  diaryRef?: string;
}

export interface DayLog {
  date: string;
  learned: string[];
  moments: string[];
  peakEmotion: Emotion;
}

/** VS Code(눈)가 로컬 서버로 보내는 이벤트. */
export type SensorEvent =
  | { type: "heartbeat" }
  | { type: "blur" }
  | { type: "code"; code: string; languageId: string }
  | { type: "build"; ok: boolean; warnings?: number };

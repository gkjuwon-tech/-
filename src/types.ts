/**
 * 꼬질룡의 공용 타입 정의.
 * 모듈 간 계약은 전부 여기 한 곳에서만 바뀐다.
 */

/** 꼬질룡의 감정 상태머신 상태. 기획서 §11.3 The Heart. */
export type Emotion =
  | "calm" // 평온
  | "focus" // 집중 (코드 노려보는 중)
  | "joy" // 기쁨 (춤)
  | "rage" // 분노 (빡침)
  | "moved" // 감동
  | "sleepy" // 졸림 (새벽 코딩)
  | "sulk" // 삐짐 (오래 방치)
  | "worry"; // 걱정 (에러 폭발)

/** 빡침 단계. 기획서 §6 기능2 Rage Level. */
export type RageLevel = 0 | 1 | 2 | 3 | 4;

/** 코드 한 조각을 평가한 결과. 평가 엔진 → 감정 상태머신으로 흐른다. */
export interface Evaluation {
  /** 평가의 큰 방향. 춤출지, 빡칠지, 그냥 볼지. */
  verdict: "dance" | "calm" | "rage";
  /** 빡칠 때의 세기. verdict가 rage가 아니면 0. */
  rageLevel: RageLevel;
  /** 꼬질룡이 화면에서 할 한 마디. 짧고 귀엽게. */
  line: string;
  /** 이 코드에서 꼬질룡이 처음 본 개념들 (RAG에 없던 것). */
  newConcepts: string[];
  /** 이미 알던(=RAG에 있던) 개념들. "아 이거 알아!" */
  knownConcepts: string[];
}

/** RAG에 영속되는 기억 한 조각. "꼬질룡이 아는 개념" 하나. */
export interface MemoryEntry {
  /** 개념 이름. 예: "for loop", "dependency injection". */
  concept: string;
  /** 이 개념을 처음 본 날 (YYYY-MM-DD). */
  firstSeen: string;
  /** 임베딩 벡터. 의미 기반 회상에 쓴다. */
  embedding: number[];
  /** 처음 봤을 때의 짧은 맥락 (어떤 코드에서 봤는지). */
  context: string;
  /** 이 개념을 적은 일기 파일명. 추억 회수용. */
  diaryRef?: string;
}

/** 하루치 관찰 누적. 일기 작성기의 입력. */
export interface DayLog {
  date: string;
  /** 오늘 처음 본 개념들. */
  learned: string[];
  /** 오늘 관찰한 주인의 모습 (감정 일기 재료). */
  moments: string[];
  /** 오늘 가장 강했던 감정. */
  peakEmotion: Emotion;
}

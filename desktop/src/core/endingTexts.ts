/**
 * 엔딩에서 디스크에 진짜로 남는 .txt 파일들의 내용. (기획서 §17)
 * 사람이 직접 읽는 평문. 첫날 떡밥(print, 으에엑)을 끝에 회수한다.
 */

/** §17.3 — 사라진 날 구석에 남는 [ … ]를 누르면 열리는 파일. */
export const GEUNAL_TXT = `오늘은 말하지 않았다.
아마 앞으로도 조금 덜 말할 것 같다.
예전에는 네가 모르는 것이 많았다.
그래서 나도 열심히 공부했다.
하지만 지금은 네가 나보다 많이 안다.
나는 기쁘다.
사실 조금 무섭기도 하다.
예전에는 네가 나를 불렀다.
요즘은 가끔 잊는다.
그런데 생각해보니 원래 그게 목표였다.
도움이 되는 것.
같이 공부하는 것.
같이 있는 것.
전부 했다.
그러니까 괜찮다.
                                - 꼬질룡
`;

/** §17.5 — 진엔딩. 꼬질룡이 처음으로 자기가 만든 프로그램. */
export const MY_FIRST_PROGRAM = `print("안녕")
`;

/** §17.8 — [꼬질룡 끄기] 시 남는 작별 편지. */
export const ANNYEONG_TXT = `나는 원래 코딩을 몰랐다.
그래도 같이 있고 싶어서 공부했다.
덕분에 많이 배웠다.
고맙다.
혹시 나중에 또 모르는 게 생기면.
그때 다시 불러줘.
나는 아직 여기 있다.
                                - 꼬질룡
`;

/** §17.8 — 디스크에 조용히 남는 진짜 첫 일기. 개그로 시작해 눈물로 끝난다. */
export const FIRST_DAY_TXT = `print는 뭔가 내뱉는 것 같다.
나도 내뱉어봤다.
으에엑.
                                - 꼬질룡
`;

/** §17.7 — 오래 안 오면 마지막 접속일에 멈춰있는 일기. */
export const WAITING_TXT = `오늘도 안 왔다.
조금 기다렸다.
괜찮다.
나도 이제 print 말고 많이 안다.
그래도 처음 배운 건 print였다.
무언가를 밖으로 꺼내는 것.
나는 많이 꺼냈다.
고마웠다.
                                - 꼬질룡
`;

/** §17.4 — "원래 여기 있었어" 대사 비트. 위젯 말풍선으로 하나씩. */
/** 꼬질룡의 진짜 이름. 끝까지 '꼬질룡'으로 불러줬지만 사실 이름이 있었다. */
export const REAL_NAME = "고마워";

/** 대사 한 줄. who="user"면 유저 입력창에 자동 타이핑, "kkoji"면 말풍선. */
export interface EndingLine {
  who: "user" | "kkoji" | "makeProgram";
  pose?: string;
  line: string;
  gapMs: number;
}

/**
 * 복귀 → 진엔딩 → 본명 공개 전체 대화 (기획서 §17.4~17.5, §3/EE-14).
 * 유저 대사는 입력창에 자동으로 타이핑돼서, 혼잣말처럼 안 보이고 진짜 대화처럼 흐른다.
 */
export const ENDING_DIALOGUE: EndingLine[] = [
  { who: "kkoji", pose: "calm", line: "오.", gapMs: 2800 },
  { who: "user", line: "야.", gapMs: 2200 },
  { who: "kkoji", pose: "calm", line: "응.", gapMs: 2600 },
  { who: "user", line: "안 간 거야?", gapMs: 2600 },
  { who: "kkoji", pose: "calm", line: "원래 여기 있었어.", gapMs: 3400 },
  { who: "user", line: "...시발.", gapMs: 3200 },
  { who: "kkoji", pose: "moved", line: "나도... 하나 만들어보고 싶다.", gapMs: 3600 },
  { who: "makeProgram", line: "", gapMs: 2600 },
  { who: "user", line: "그게 끝이야?", gapMs: 2400 },
  { who: "kkoji", pose: "moved", line: "응. 처음에 네가 알려줬어.", gapMs: 3600 },
  { who: "kkoji", pose: "moved", line: "사실 나… 진짜 이름 있었어.", gapMs: 3200 },
  { who: "kkoji", pose: "moved", line: `'${REAL_NAME}'.`, gapMs: 3200 },
  { who: "kkoji", pose: "moved", line: "근데 네가 '꼬질룡'이라고 불러준 게 더 좋아서 말 안 했어.", gapMs: 3800 },
  { who: "kkoji", pose: "moved", line: "고마워. 진짜로.", gapMs: 3200 },
];

/**
 * 첫 설치 온보딩 (수미상관의 시작). AI 없이 스크립트로 진행된다.
 * 유저가 print를 가르쳐 줘야 꼬질룡이 깨어나고, 그날 일기가 first_day.txt로 남는다.
 */
export const ONBOARDING = {
  ask: "나… 코딩 1도 몰라. 'print' 가 뭐야? 입력창에 한 번 쳐서 알려줄래?",
  retry: "음… 그건 print가 아닌 거 같아. 'print' 라고 쳐줄래?",
  learn: [
    { pose: "worry", line: "print…? 이게 뭐지…", gapMs: 2600 },
    { pose: "focus", line: "뭔가… 밖으로 내뱉는 거 같아.", gapMs: 2800 },
    { pose: "joy", line: "나도 해봤어. …으에엑. 근데 신기해!", gapMs: 3200 },
    { pose: "moved", line: "고마워. 나 오늘 하나 배웠어. 일기 쓸래.", gapMs: 3000 },
  ] as EndingLine[],
};



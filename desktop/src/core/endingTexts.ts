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
export const RETURN_BEATS: Array<{ pose: string; line: string; gapMs: number }> = [
  { pose: "calm", line: "오.", gapMs: 2600 },
  { pose: "calm", line: "응.", gapMs: 2400 },
  { pose: "calm", line: "원래 여기 있었어.", gapMs: 3200 },
];

/** §17.5 — 진엔딩 비트. */
export const TRUE_ENDING_BEATS: Array<{ pose: string; line: string; gapMs: number }> = [
  { pose: "moved", line: "나도... 하나 만들어보고 싶다.", gapMs: 3400 },
  { pose: "moved", line: "처음에 네가 알려줬어.", gapMs: 3200 },
];

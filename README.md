# 꼬질룡 (KKOJILRYONG)

> **"코딩 1도 몰랐는데, 네가 좋아서 배웠어."**

데스크탑 한 구석에 사는 비뚤빼뚤한 검은 선 공룡. 네가 코드를 칠 때마다 어깨너머로 슥 보고, 똥 같은 코드엔 빡치고(귀엽게), 멋진 코드엔 막춤 추고, 매일 밤 본 걸로 일기(`.txt`)를 쓴다. 한 번 같이 공부한 건 절대 안 까먹고 — **VS Code 말고 딴 거 하면 쿨쿨 잔다.**

기획 전문은 [`꼬질룡_기획서.md`](./꼬질룡_기획서.md).

---

## 구조: 몸(데스크탑) + 눈(VS Code)

기획서 §11대로, 꼬질룡 본체는 **데스크탑 앱**이고 VS Code는 신호만 보내는 **얇은 눈**이다.

```
┌─ desktop/  꼬질룡 데스크탑 앱 (Electron, always-on 투명 캐릭터) ─┐
│  몸 + 뇌(Gemini) + 심장(감정) + 기억(RAG) + 일기 + 포즈        │
│  127.0.0.1 로컬 서버로 센서 이벤트 수신                        │
└───────────────────────▲──────────────────────────────────────┘
                        │ HTTP (코드 이벤트 + 포커스 하트비트)
        ┌───────────────┴───────────────┐
        │ vscode/  얇은 커넥터 (눈)       │
        │  저장/타이핑 + 포커스만 쏨       │
        └───────────────────────────────┘
```

### "VS Code 말고 딴 거 하면 자는 모션"

커넥터는 **VS Code가 포커스일 때만 하트비트**를 보낸다. 브라우저 등 다른 앱으로 옮기면 하트비트가 끊기고, 데스크탑 본체가 타임아웃(기본 12초)을 감지해 꼬질룡을 **재운다**. 다시 VS Code로 돌아오면 깬다. 네이티브 창 감지 없이 깔끔하게.

---

## desktop/ — 본체 (Electron)

| 모듈 | 역할 |
|---|---|
| `core/brain/evaluator` | 코드에서 개념 추출 → RAG 분기 → 빡침/춤 판정 + 대사 |
| `core/heart/stateMachine` | 평가 + 시간/방치 + **자리(presence)** → 감정 전이(8종) |
| `core/rag/memory` + `vector` | 임베딩·코사인 유사도 영속 기억. 한 번 본 건 안 까먹음 |
| `core/diary/writer` | 세션 끝 `.txt` 일기 + 새 개념 RAG 학습 |
| `core/face/poses` `cutout` `poseStudio` | Gemini 포즈 생성 → 무료 휘도 누끼 → 투명 PNG 캐시 |
| `config` | API 키 safeStorage 암호화 보관 (BYOK) |
| `presence` | 하트비트 → 깸/잠 |
| `server` | 127.0.0.1 이벤트 수신 |
| `main` | Electron 본체: 투명 always-on-top 창, 우클릭 메뉴, 배선 |

**실행:**
```bash
cd desktop && npm install && npm start
```
기본 포즈 이미지는 `desktop/assets/poses`에 동봉돼 있어 키 없이도 바로 나온다.
다시 뽑고 싶으면 캐릭터 우클릭 → `Gemini API 키 입력` ([키](https://aistudio.google.com/apikey)) → `포즈 이미지 생성하기`.
키는 settings.json에 `geminiKey`로 평문 입력해두면 다음 실행 때 자동 암호화된다.

조작: **클릭=쓰다듬기 · 드래그=이동 · 우클릭=메뉴**.

### 설치 파일(.exe / .dmg / .AppImage) 만들기

`electron-builder`로 패키징한다. 산출물은 `desktop/release/`.

```bash
cd desktop && npm install
npm run dist:win     # Windows: 설치본(NSIS) + 포터블 exe
npm run dist:mac     # macOS: dmg
npm run dist:linux   # Linux: AppImage
npm run pack         # 설치파일 없이 폴더로만 (빠른 확인용)
```

> Windows `.exe`는 **Windows에서 빌드**하는 게 가장 확실하다. 리눅스/맥에서 윈도우
> 설치본을 만들려면 `wine`이 필요하다(없으면 exe 본체는 나오지만 아이콘/메타데이터
> 새김 단계에서 멈춘다).

**Windows 머신이 없으면 → GitHub Actions로 받는다.** `.github/workflows/build-desktop.yml`이
`windows-latest`에서 네이티브로 빌드한다:

- **Actions 탭 → `build-desktop` → Run workflow** → 끝나면 `kkojilryong-windows-exe` 아티팩트(설치본 + 포터블 exe) 다운로드.
- 또는 `v0.2.0` 같은 **`v*` 태그를 푸시**하면 빌드 후 그 릴리스에 `.exe`가 자동 첨부된다.

## vscode/ — 눈 (커넥터)

```bash
cd vscode && npm install && npm run compile   # 그 후 F5 또는 패키징
```
설정: `kkojilryong.serverPort`(기본 8787), `debounceMs`, `sendCode`(끄면 하트비트만 — 코드 내용 전송 안 함).

---

## RAG가 만드는 "성장" (심장)

처음엔 아무것도 모른다. 네 코드에서 개념을 만나고, 일기에 적는 순간 RAG에 임베딩으로 학습된다. 다음에 같은 개념을 다시 만나면 유사도로 회상해 "아 이거 알아!" 한다. 어제 모른 건 오늘도 모르고, 어제 배운 건 오늘 안다. 이 점진성·일관성이 캐릭터의 진정성을 만든다.

## 그림체: Gemini 생성 + 무료 누끼 + 레퍼런스 일관성

기본 포즈는 동봉(`desktop/assets/poses`). 다시 뽑을 땐 자세별로:
```
1) 앵커(평온)를 외형 프롬프트로 생성
2) 앵커 원본을 레퍼런스로 컨텍스트에 넣고 나머지 포즈 생성  ← 같은 캐릭터 유지(일관성)
3) 휘도→알파 누끼(공짜, 로컬 pngjs)로 투명 PNG화            ← 외부 API 없음
4) userData/poses 캐시(자세별 고정, 재사용)
```
앵커 한 장을 모든 포즈의 레퍼런스로 먹여서 선/비율/눈이 일관되게 유지된다.
흰 배경 선화라 휘도가 곧 알파다(흰→투명, 검은 선→불투명) — remove.bg 같은 유료 서비스 불필요.
막춤(3프레임)·빡침(2프레임)·잠(2프레임)은 끊기게 순환.

## 프라이버시 / BYOK

평가·일기·기억은 로컬(userData JSON + 일기 `.txt`). 누끼도 로컬에서 처리(외부 전송 없음). Gemini는 네 키로 직접 — 우리 서버 안 거침. 로컬 서버는 127.0.0.1에만 바인딩. (기획서 §14)

---

 *오늘도 코딩하자. 똥 같아도 괜찮아. 내가 옆에 있을게.*

// @ts-check
// 꼬질룡 webview 프론트. 확장이 보내는 감정 상태를 표정/모션/말풍선으로 그린다.
(function () {
  const vscode = acquireVsCodeApi();
  const kkoji = /** @type {HTMLElement} */ (document.getElementById("kkoji"));
  const speech = /** @type {HTMLElement} */ (document.getElementById("speech"));
  const stage = /** @type {HTMLElement} */ (document.getElementById("stage"));

  const EMOTIONS = [
    "calm", "focus", "joy", "rage", "moved", "sleepy", "sulk", "worry",
  ];

  /** 감정별 눈 모양 (검은 선 최소 조합). 기획서 §5 표정 변주 */
  const EYES = {
    calm: dots(),
    focus: lines(),       // 실눈 - -
    joy: happy(),         // 휘어진 눈 ^ ^
    rage: angry(),        // 모인 눈 > <
    moved: teary(),       // 눈물 ; ;
    sleepy: lines(),      // - -
    sulk: dots(),         // 등 돌려서 어차피 안 보임
    worry: dots(),
  };

  let speechTimer;

  // 쓰다듬기
  kkoji.addEventListener("click", () => {
    vscode.postMessage({ type: "pet" });
    pat();
  });

  window.addEventListener("message", (e) => {
    const m = e.data;
    if (m.type === "state") {
      applyState(m);
    } else if (m.type === "say") {
      showSpeech(m.line);
    }
  });

  function applyState(m) {
    // 클래스 리셋 후 감정 적용
    kkoji.className = "kkoji " + m.emotion;
    if (m.emotion === "rage") {
      kkoji.classList.add("lv" + (m.rageLevel || 1));
      if (m.rageLevel >= 4 && !m.reduceMotion) {
        document.body.classList.add("shake");
        setTimeout(() => document.body.classList.remove("shake"), 800);
      }
    }
    document.body.classList.toggle("reduce-motion", !!m.reduceMotion);
    setEyes(m.emotion);
    setMark(m.emotion);
    if (m.line) showSpeech(m.line);
  }

  function setEyes(emotion) {
    const g = kkoji.querySelector(".eyes");
    if (!g) return;
    g.innerHTML = (EYES[emotion] || EYES.calm);
    const mouth = kkoji.querySelector(".mouth");
    if (mouth) {
      // 감정 클 때만 입 한 줄 (기획서 §5)
      mouth.style.visibility =
        emotion === "joy" || emotion === "rage" || emotion === "moved"
          ? "visible"
          : "hidden";
    }
  }

  /** 머리 위 작은 검은 선 표식: 💢 / z / 눈물 */
  function setMark(emotion) {
    const old = stage.querySelector(".mark");
    if (old) old.remove();
    const text =
      emotion === "rage" ? "💢" :
      emotion === "sleepy" ? "z" :
      emotion === "moved" ? "," : "";
    if (!text) return;
    const mark = document.createElement("div");
    mark.className = "mark";
    mark.textContent = text;
    mark.style.left = "62%";
    mark.style.top = "30px";
    stage.appendChild(mark);
  }

  function showSpeech(line) {
    speech.textContent = line;
    speech.classList.remove("hidden");
    clearTimeout(speechTimer);
    speechTimer = setTimeout(() => speech.classList.add("hidden"), 6000);
  }

  function pat() {
    kkoji.style.transform = "translateY(-6px)";
    setTimeout(() => (kkoji.style.transform = ""), 120);
  }

  // ── 눈 SVG 조각들 (currentColor = 검은 선) ──────────────
  function dots() {
    return `<circle cx="42" cy="48" r="2.6" fill="currentColor"/>
            <circle cx="60" cy="48" r="2.6" fill="currentColor"/>`;
  }
  function lines() {
    return `<path d="M38 48 h8" /><path d="M56 48 h8" />`;
  }
  function happy() {
    return `<path d="M38 50 q4 -5 8 0" /><path d="M56 50 q4 -5 8 0" />`;
  }
  function angry() {
    return `<path d="M38 46 l8 4" /><path d="M64 46 l-8 4" />`;
  }
  function teary() {
    return `<circle cx="42" cy="48" r="2.4" fill="currentColor"/>
            <circle cx="60" cy="48" r="2.4" fill="currentColor"/>
            <path d="M42 52 q-1 5 0 7" /><path d="M60 52 q1 5 0 7" />`;
  }

  // 초기 표정
  setEyes("calm");
  void EMOTIONS;
})();

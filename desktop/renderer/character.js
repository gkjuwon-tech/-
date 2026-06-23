// 꼬질룡 데스크탑 렌더러. 메인이 보내는 감정 상태를 표정/모션/말풍선으로 그린다.
(function () {
  const api = window.kkoji;
  const kkoji = document.getElementById("kkoji");
  const speech = document.getElementById("speech");
  const stage = document.getElementById("stage");
  const poseImg = document.getElementById("poseImg");

  let speechTimer;
  let poseTimer;

  const EYES = {
    calm: dots(),
    focus: lines(),
    joy: happy(),
    rage: angry(),
    moved: teary(),
    sleepy: closed(),
    sulk: dots(),
    worry: dots(),
  };

  function onPet() {
    api.pet();
    pat();
  }
  kkoji.addEventListener("click", onPet);
  poseImg.addEventListener("click", onPet);

  // 우클릭 → 메인 컨텍스트 메뉴
  window.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    api.menu();
  });

  api.onState(applyState);
  api.onSay((m) => {
    setEyes(m.emotion || "joy");
    showSpeech(m.line);
  });

  function applyState(m) {
    document.body.classList.toggle("reduce-motion", !!m.reduceMotion);
    if (m.poseFrames && m.poseFrames.length) {
      showPose(m.poseFrames, m.poseFrameMs, !!m.reduceMotion);
    } else {
      showSvg(m);
    }
    if (m.emotion === "rage" && m.rageLevel >= 4 && !m.reduceMotion) {
      document.body.classList.add("shake");
      setTimeout(() => document.body.classList.remove("shake"), 800);
    }
    if (m.line) showSpeech(m.line);
  }

  function showPose(frames, frameMs, reduceMotion) {
    clearInterval(poseTimer);
    kkoji.classList.add("hidden");
    poseImg.classList.remove("hidden");
    let i = 0;
    poseImg.src = frames[0];
    if (frames.length > 1 && frameMs > 0 && !reduceMotion) {
      poseTimer = setInterval(() => {
        i = (i + 1) % frames.length;
        poseImg.src = frames[i];
      }, frameMs);
    }
  }

  function showSvg(m) {
    clearInterval(poseTimer);
    poseImg.classList.add("hidden");
    kkoji.classList.remove("hidden");
    kkoji.className = "kkoji " + m.emotion;
    if (m.emotion === "rage") kkoji.classList.add("lv" + (m.rageLevel || 1));
    setEyes(m.emotion);
    setMark(m.emotion);
  }

  function setEyes(emotion) {
    const g = kkoji.querySelector(".eyes");
    if (g) g.innerHTML = EYES[emotion] || EYES.calm;
    const mouth = kkoji.querySelector(".mouth");
    if (mouth) {
      mouth.style.visibility =
        emotion === "joy" || emotion === "rage" || emotion === "moved"
          ? "visible"
          : "hidden";
    }
  }

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
    mark.style.left = "64%";
    mark.style.top = "26px";
    stage.appendChild(mark);
  }

  function showSpeech(line) {
    if (!line) return;
    speech.textContent = line;
    speech.classList.remove("hidden");
    clearTimeout(speechTimer);
    speechTimer = setTimeout(() => speech.classList.add("hidden"), 6000);
  }

  function pat() {
    const el = poseImg.classList.contains("hidden") ? kkoji : poseImg;
    el.style.transform = "translateY(-6px)";
    setTimeout(() => (el.style.transform = ""), 120);
  }

  function dots() {
    return `<circle cx="42" cy="48" r="2.6" fill="currentColor"/><circle cx="60" cy="48" r="2.6" fill="currentColor"/>`;
  }
  function lines() {
    return `<path d="M38 48 h8" fill="none" stroke="currentColor"/><path d="M56 48 h8" fill="none" stroke="currentColor"/>`;
  }
  function happy() {
    return `<path d="M38 50 q4 -5 8 0" fill="none" stroke="currentColor"/><path d="M56 50 q4 -5 8 0" fill="none" stroke="currentColor"/>`;
  }
  function angry() {
    return `<path d="M38 46 l8 4" fill="none" stroke="currentColor"/><path d="M64 46 l-8 4" fill="none" stroke="currentColor"/>`;
  }
  function teary() {
    return `<circle cx="42" cy="48" r="2.4" fill="currentColor"/><circle cx="60" cy="48" r="2.4" fill="currentColor"/><path d="M42 52 q-1 5 0 7" fill="none" stroke="currentColor"/><path d="M60 52 q1 5 0 7" fill="none" stroke="currentColor"/>`;
  }
  function closed() {
    return `<path d="M38 49 q4 3 8 0" fill="none" stroke="currentColor"/><path d="M56 49 q4 3 8 0" fill="none" stroke="currentColor"/>`;
  }

  setEyes("sleepy");
})();

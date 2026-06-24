// 꼬질룡 위젯 렌더러. 메인이 보내는 'state' 뷰를 그린다. 캐릭터는 언제나
// 미리 생성해 둔 일관된 포즈 PNG. 입력창으로 말도 걸 수 있고, 엔딩 땐 유저
// 대사가 자동으로 타이핑된다.
(function () {
  const api = window.kkoji;

  // 손글씨 폰트는 렌더를 막지 않게 비동기 주입.
  try {
    const f = document.createElement("link");
    f.rel = "stylesheet";
    f.href =
      "https://fonts.googleapis.com/css2?family=Nanum+Pen+Script&family=Gaegu:wght@400;700&display=swap";
    document.head.appendChild(f);
  } catch (_) {}

  const poseImg = document.getElementById("poseImg");
  const speech = document.getElementById("speech");
  const goneicon = document.getElementById("goneicon");
  const talk = document.getElementById("talk");

  let speechTimer, frameTimer, gone = false, autotyping = false;

  function onPet() { api.pet(); bump(); }
  poseImg.addEventListener("click", onPet);
  goneicon.addEventListener("click", () => api.goneClick());
  window.addEventListener("contextmenu", (e) => { e.preventDefault(); api.menu(); });

  // 입력창: Enter로 말 걸기 (자동 타이핑 중엔 막음)
  talk.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !autotyping) {
      const v = talk.value.trim();
      if (v) { api.talk(v); }
      talk.value = "";
    }
  });

  api.onState(render);
  if (api.onMode) api.onMode(applyMode);
  if (api.onAutotype) api.onAutotype((m) => autoType((m && m.text) || ""));

  function render(v) {
    if (!v) return;
    if (v.gone) {
      clearInterval(frameTimer);
      poseImg.classList.add("hidden");
      speech.classList.add("hidden");
      talk.classList.add("hidden");
      goneicon.classList.remove("hidden");
      gone = true;
      return;
    }
    gone = false;
    goneicon.classList.add("hidden");
    talk.classList.remove("hidden"); // 평소엔 입력창 보임 (말 걸 수 있게)
    if (Array.isArray(v.poseFrames) && v.poseFrames.length) {
      showFrames(v.poseFrames, v.poseFrameMs, !!v.reduceMotion);
    }
    if (v.line) showSpeech(v.line);
  }

  // 온보딩/특수 상황: 안내문(placeholder) 바꾸고 입력창에 포커스
  function applyMode(m) {
    if (!m) return;
    if (m.input) {
      talk.classList.remove("hidden");
      if (m.placeholder) talk.placeholder = m.placeholder;
      setTimeout(() => talk.focus(), 50);
    } else if (m.placeholder) {
      talk.placeholder = m.placeholder;
    }
  }

  // 유저 대사 자동 타이핑 (엔딩 대화 연출). 입력창에 한 글자씩 찍힌 뒤 사라짐.
  function autoType(text) {
    talk.classList.remove("hidden");
    autotyping = true;
    talk.value = "";
    let i = 0;
    const iv = setInterval(() => {
      talk.value = text.slice(0, ++i);
      if (i >= text.length) {
        clearInterval(iv);
        setTimeout(() => { talk.value = ""; autotyping = false; }, 1100);
      }
    }, 90);
  }

  function showFrames(frames, frameMs, reduceMotion) {
    clearInterval(frameTimer);
    poseImg.classList.remove("hidden");
    let i = 0;
    poseImg.src = frames[0];
    if (frames.length > 1 && frameMs > 0 && !reduceMotion) {
      frameTimer = setInterval(() => {
        i = (i + 1) % frames.length;
        poseImg.src = frames[i];
      }, frameMs);
    }
  }

  function showSpeech(line) {
    speech.textContent = line;
    speech.classList.remove("hidden");
    clearTimeout(speechTimer);
    speechTimer = setTimeout(() => speech.classList.add("hidden"), 6000);
  }

  function bump() {
    poseImg.style.transform = "translateY(-6px)";
    setTimeout(() => (poseImg.style.transform = ""), 120);
  }

  api.ready();
})();

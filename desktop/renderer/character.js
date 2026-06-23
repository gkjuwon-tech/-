// 꼬질룡 위젯 렌더러.
// 메인이 보내는 단일 'state' 뷰를 그린다. 캐릭터는 언제나 미리 생성해 둔
// 일관된 포즈 PNG. (허접한 SVG 폴백 없음)
(function () {
  const api = window.kkoji;

  // 손글씨 폰트는 렌더를 막지 않게 JS로 비동기 주입 (오프라인이면 그냥 폴백).
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

  let speechTimer;
  let frameTimer;

  function onPet() {
    api.pet();
    bump();
  }
  poseImg.addEventListener("click", onPet);
  goneicon.addEventListener("click", () => api.goneClick());
  window.addEventListener("contextmenu", (e) => {
    e.preventDefault();
    api.menu();
  });

  api.onState(render);

  function render(v) {
    if (!v) return;
    if (v.gone) {
      // 사라짐 (기획서 §17): 위젯에서 진짜 사라지고 구석에 [ … ]만.
      clearInterval(frameTimer);
      poseImg.classList.add("hidden");
      speech.classList.add("hidden");
      goneicon.classList.remove("hidden");
      return;
    }
    goneicon.classList.add("hidden");
    if (Array.isArray(v.poseFrames) && v.poseFrames.length) {
      showFrames(v.poseFrames, v.poseFrameMs, !!v.reduceMotion);
    }
    if (v.line) showSpeech(v.line);
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

  // 로드 완료 → 메인에 알리고 현재 뷰를 받는다.
  api.ready();
})();

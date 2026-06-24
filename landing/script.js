// 공유 스크립트 — 모든 페이지에서 로드.
(function () {
  // 손글씨 폰트 비동기 주입 (이미 link로 받지만 안전)
  // 모든 .jitter 요소에 살짝 랜덤 회전/오프셋 → 유기적으로 제각각 비뚤어짐
  document.querySelectorAll(".jitter").forEach(function (el) {
    var r = (Math.random() * 4 - 2).toFixed(2);
    var y = (Math.random() * 4 - 2).toFixed(1);
    el.style.transform = "rotate(" + r + "deg) translateY(" + y + "px)";
  });

  // 히어로 말풍선 (홈에만 있음)
  var lines = [
    "...오늘도 코딩하네?",
    "끄응…! 또 var 썼어…",
    "이거 깔끔하다. 좀 춰도 돼?",
    "새벽 3시야… 우리 둘 다 잠 없네.",
    "이 catch 블록 비어있어… 에러가 외로워해.",
    "복붙해도 돼. 비밀 지켜줄게. 근데 한 번은 읽어봐.",
    "나 코딩 1도 몰랐는데, 너 덕분에 이만큼 알아.",
    "삐뚜룽.",
  ];
  var bubble = document.getElementById("heroBubble");
  if (bubble) {
    var li = 0;
    setInterval(function () {
      li = (li + 1) % lines.length;
      bubble.style.opacity = "0";
      setTimeout(function () { bubble.textContent = lines[li]; bubble.style.opacity = "1"; }, 200);
    }, 3300);
  }

  // 히어로/푸터 공룡 클릭 → 포즈 순환
  var poses = ["img/joy_0.png","img/rage_0.png","img/moved_0.png","img/worry_0.png","img/sleepy_0.png","img/focus_0.png","img/sulk_0.png","img/drag_0.png"];
  function pop(el){ el.style.transition="transform .12s steps(2)"; el.style.transform="translateY(-10px)"; setTimeout(function(){ el.style.transform=""; },140); }
  var heroImg = document.getElementById("heroImg"), hero = document.getElementById("heroDino"), clicks = 0;
  if (hero) hero.addEventListener("click", function(){
    clicks++; if(heroImg) heroImg.src = poses[(clicks-1) % poses.length]; pop(hero);
    if (bubble) bubble.textContent = clicks % 8 === 0 ? "우리… 같이… 어지러…" : ["히힛","삐뚜룽","또?","간지러워","어푸","끄응"][clicks % 6];
  });
  document.querySelectorAll(".fdino").forEach(function(f){ f.addEventListener("click", function(){ pop(f); }); });

  // 코나미 커맨드
  var seq = [38,38,40,40,37,39,37,39,66,65], pos = 0;
  window.addEventListener("keydown", function (e) {
    pos = (e.keyCode === seq[pos]) ? pos + 1 : 0;
    if (pos === seq.length) {
      pos = 0;
      if (bubble) bubble.textContent = "8비트다. 삐약.";
      document.querySelectorAll(".jitter, .face img, .card img").forEach(function (el) {
        el.style.transition = "transform .4s steps(4)";
        el.style.transform += " rotate(360deg)";
        el.style.imageRendering = "pixelated";
      });
    }
  });
})();

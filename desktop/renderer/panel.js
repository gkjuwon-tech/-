// 인앱 패널: 성장 스탯 + 일기 브라우저 + 설정.
(function () {
  const api = window.kkoji;
  try {
    const f = document.createElement("link");
    f.rel = "stylesheet";
    f.href = "https://fonts.googleapis.com/css2?family=Nanum+Pen+Script&family=Gaegu:wght@400;700&display=swap";
    document.head.appendChild(f);
  } catch (_) {}

  const $ = (id) => document.getElementById(id);

  $("close").addEventListener("click", () => api.panel("close"));
  $("setKey").addEventListener("click", () => api.panel("setKey"));
  $("openFolder").addEventListener("click", () => api.panel("openFolder"));
  $("autostart").addEventListener("click", () => {
    const on = !$("autostart").classList.contains("on");
    api.panel("autostart", on);
  });

  api.onPanelData(render);
  api.panel("refresh"); // 로드 끝나고 직접 데이터 요청 (유실 방지)

  function render(d) {
    if (!d) return;
    if (d.dino) $("dino").src = d.dino;
    $("stage").textContent = d.stage || "알";
    $("pct").textContent = (d.pct || 0) + "%";
    $("bar").style.width = (d.pct || 0) + "%";
    $("concepts").textContent = d.concepts || 0;
    $("days").textContent = d.days || 0;
    $("diaryn").textContent = d.diaryCount || 0;

    const ul = $("diaries");
    ul.innerHTML = "";
    if (d.diaries && d.diaries.length) {
      d.diaries.forEach((name) => {
        const li = document.createElement("li");
        li.innerHTML = "<span>" + name.replace(/\.txt$/, "") + "</span><span class='d'>txt</span>";
        li.addEventListener("click", () => api.panel("openDiary", name));
        ul.appendChild(li);
      });
    } else {
      ul.innerHTML = "<li class='empty'>아직 일기가 없어. 같이 코딩하자.</li>";
    }

    const ks = $("keyStatus");
    if (d.hasKey) { ks.textContent = "(등록됨)"; ks.className = "ok"; }
    else { ks.textContent = "(필요)"; ks.className = "need"; }
    $("autostart").classList.toggle("on", !!d.autostart);
  }
})();

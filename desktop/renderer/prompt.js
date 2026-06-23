// 키 입력 모달. 메인이 prompt:init로 제목을 주면 표시하고, 확인 시 값을 돌려준다.
(function () {
  const api = window.kkoji;
  try {
    const f = document.createElement("link");
    f.rel = "stylesheet";
    f.href =
      "https://fonts.googleapis.com/css2?family=Nanum+Pen+Script&family=Gaegu:wght@400;700&display=swap";
    document.head.appendChild(f);
  } catch (_) {}
  const title = document.getElementById("title");
  const val = document.getElementById("val");

  api.onPromptInit((d) => {
    if (d && d.title) title.textContent = d.title;
    val.focus();
  });

  function submit() {
    api.promptSubmit(val.value);
  }
  document.getElementById("ok").addEventListener("click", submit);
  document.getElementById("cancel").addEventListener("click", () =>
    api.promptSubmit(null)
  );
  val.addEventListener("keydown", (e) => {
    if (e.key === "Enter") submit();
    if (e.key === "Escape") api.promptSubmit(null);
  });
})();

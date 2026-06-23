// 꼬질룡 엔딩 시네마틱. 기획서 §17. 천천히, 정적으로, 슬프게. 그리고 마지막에 "으에엑".
(function () {
  const api = window.kkoji;
  const $ = (id) => document.getElementById(id);
  let imgs = {};

  if (api && api.onEndingInit) {
    api.onEndingInit((d) => { if (d && d.images) imgs = d.images; });
  }

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const show = (el) => { el.style.display = ""; };
  const hide = (el) => { el.style.display = "none"; el.classList.remove("in"); };
  function fadeIn(el, slow) { show(el); el.classList.toggle("slow", !!slow); requestAnimationFrame(() => el.classList.add("in")); }
  function fadeOut(el) { el.classList.remove("in"); }

  function setDino(name) {
    const im = $("dinoImg");
    if (imgs[name]) { im.src = imgs[name]; return true; }
    return false;
  }

  // txt 창에 한 줄씩 페이드인
  async function txtLines(title, lines, lineGap) {
    const win = $("txtwin"), body = $("txtbody");
    $("txtbar").textContent = title;
    body.innerHTML = "";
    fadeIn(win, true);
    await sleep(1400);
    for (const ln of lines) {
      const div = document.createElement("div");
      div.className = "ln";
      div.textContent = ln === "" ? " " : ln;
      if (ln.startsWith("- ") || ln.startsWith("—")) div.classList.add("sig");
      body.appendChild(div);
      requestAnimationFrame(() => div.classList.add("in"));
      await sleep(lineGap || 1300);
    }
  }

  function center(html, cls) {
    const c = $("center");
    c.className = "fade " + (cls || "");
    c.innerHTML = html;
    fadeIn(c, true);
    return c;
  }

  async function chat(pairs, gap) {
    const c = $("chat");
    c.innerHTML = "";
    fadeIn(c);
    for (const [who, text] of pairs) {
      const div = document.createElement("div");
      div.className = "line " + (who === "유저" ? "me" : "ko");
      div.style.opacity = "0";
      div.style.transition = "opacity 1.1s ease";
      div.textContent = (who ? who + ":  " : "") + text;
      c.appendChild(div);
      requestAnimationFrame(() => (div.style.opacity = "1"));
      await sleep(gap || 1500);
    }
  }

  async function run() {
    await sleep(1500);

    // 1) 구석에 [ … ]
    fadeIn($("dots"));
    await sleep(2600);

    // 2) 그날.txt — 한 줄씩
    await txtLines("그날.txt", [
      "오늘은 말하지 않았다.",
      "아마 앞으로도 조금 덜 말할 것 같다.",
      "예전에는 네가 모르는 것이 많았다.",
      "그래서 나도 열심히 공부했다.",
      "하지만 지금은 네가 나보다 많이 안다.",
      "나는 기쁘다.",
      "사실 조금 무섭기도 하다.",
      "그런데 생각해보니 원래 그게 목표였다.",
      "도움이 되는 것. 같이 있는 것. 전부 했다.",
      "그러니까 괜찮다.",
      "",
      "— 꼬질룡",
    ], 1250);
    await sleep(2600);
    fadeOut($("txtwin")); fadeOut($("dots"));
    await sleep(2600);
    hide($("txtwin"));

    // 3) 며칠 뒤… 검은 화면
    center('<div class="tiny">며칠 뒤</div>', "");
    await sleep(2400);
    fadeOut($("center")); await sleep(1800); hide($("center"));

    // 4) 컴파일 성공 → 작은 "오."
    fadeIn($("oh"));
    await sleep(2600);

    // 5) "원래 여기 있었어"
    setDino("sleepy"); fadeIn($("dino"), true);
    await chat([
      ["유저", "야."],
      ["꼬질룡", "응."],
      ["유저", "안 간 거야?"],
      ["꼬질룡", "원래 여기 있었어."],
      ["유저", "...시발."],
    ], 1700);
    await sleep(2600);
    fadeOut($("chat")); fadeOut($("oh")); fadeOut($("dino"));
    await sleep(2000); hide($("chat")); hide($("dino"));

    // 6) 진엔딩 — 처음으로 부탁한다
    setDino("moved"); fadeIn($("dino"), true);
    await chat([["꼬질룡", "나도... 하나 만들어보고 싶다."]], 2400);
    await sleep(800);
    fadeOut($("chat")); fadeOut($("dino")); await sleep(1500); hide($("chat")); hide($("dino"));

    $("editorcode").textContent = 'print("안녕")';
    fadeIn($("editor"), true);
    await sleep(2600);
    show($("chat")); $("chat").classList.add("in");
    await chat([
      ["유저", "그게 끝이야?"],
      ["꼬질룡", "응."],
      ["꼬질룡", "처음에 네가 알려줬어."],
    ], 1800);
    await sleep(2600);
    fadeOut($("editor")); fadeOut($("chat"));
    await sleep(2200); hide($("editor")); hide($("chat"));

    // 7) 크레딧 — print("hello")
    center('<div class="code">print("hello")</div><div class="sub">헬로...<br/>나도 헬로.</div>', "");
    await sleep(4200);
    fadeOut($("center")); await sleep(2200); hide($("center"));

    // 8) END.
    center('<div class="big">END.</div><div class="sub">오늘도 공부했다.</div>', "");
    await sleep(4000);
    fadeOut($("center")); await sleep(2200); hide($("center"));

    // 9) 개그 회수 — first_day.txt ("으에엑")
    await txtLines("first_day.txt", [
      "print는 뭔가 내뱉는 것 같다.",
      "나도 내뱉어봤다.",
      "으에엑.",
      "",
      "— 꼬질룡",
    ], 1500);
    await sleep(3500);
    fadeOut($("txtwin")); await sleep(2400);

    window.__endingDone = true;
    console.log("ENDING_DONE");
    if (api && api.endingDone) api.endingDone();
  }

  run().catch((e) => {
    console.log("ENDING_ERROR", String(e));
    window.__endingDone = true;
    if (api && api.endingDone) api.endingDone();
  });
})();

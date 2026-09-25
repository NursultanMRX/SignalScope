/* VisionX site: language + theme toggles, lazy Plotly charts that follow the theme. */
(() => {
  const root = document.documentElement;
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
  };
  const DARK_MAP = window.SS_DARK_MAP || {};
  const darkMQ = window.matchMedia("(prefers-color-scheme: dark)");

  // ---------- language ----------
  const langBtn = document.getElementById("lang-toggle");
  function setLang(l) {
    root.dataset.lang = l;
    root.lang = l;
    langBtn.textContent = l === "uz" ? "EN" : "UZ";
    langBtn.setAttribute("aria-label", l === "uz" ? "Switch to English" : "O'zbek tiliga o'tish");
    store.set("ss-lang", l);
    if (typeof labelTheme === "function") labelTheme();
    rerenderAll();
  }
  langBtn.addEventListener("click", () => setLang(root.dataset.lang === "uz" ? "en" : "uz"));

  // ---------- theme (two states: system, or the pinned opposite) ----------
  const themeBtn = document.getElementById("theme-toggle");
  const meta = document.querySelector('meta[name="color-scheme"]');
  const isDark = () => (root.dataset.theme ? root.dataset.theme === "dark" : darkMQ.matches);
  function labelTheme() {
    const d = isDark();
    themeBtn.textContent = d ? (root.dataset.lang === "en" ? "Light" : "Yorug'") : (root.dataset.lang === "en" ? "Dark" : "Qorong'i");
    themeBtn.setAttribute("aria-label", d ? "Light theme / Yorug' mavzu" : "Dark theme / Qorong'i mavzu");
  }
  themeBtn.addEventListener("click", () => {
    const system = darkMQ.matches ? "dark" : "light";
    if (root.dataset.theme) {            // pinned -> back to system
      delete root.dataset.theme;
      meta.content = "light dark";
      store.set("ss-theme", "");
    } else {                             // system -> pin the opposite
      const pin = system === "dark" ? "light" : "dark";
      root.dataset.theme = pin;
      meta.content = pin;
      store.set("ss-theme", pin);
    }
    labelTheme();
    rerenderAll();
  });
  darkMQ.addEventListener("change", () => { labelTheme(); rerenderAll(); });

  // ---------- charts ----------
  const specs = new Map();     // id -> figure JSON (light colours)
  const rendered = new Set();

  function swapColors(obj) {
    if (typeof obj === "string") return DARK_MAP[obj.toLowerCase()] || obj;
    if (Array.isArray(obj)) return obj.map(swapColors);
    if (obj && typeof obj === "object") {
      const o = {};
      for (const k of Object.keys(obj)) o[k] = swapColors(obj[k]);
      return o;
    }
    return obj;
  }

  function themed(fig) {
    const f = isDark() ? swapColors(fig) : JSON.parse(JSON.stringify(fig));
    f.layout = f.layout || {};
    f.layout.autosize = true;
    delete f.layout.width;
    if (window.innerWidth < 720) {
      f.layout.margin = Object.assign({}, f.layout.margin, { l: 44, r: 8 });
      f.layout.legend = Object.assign({}, f.layout.legend, { orientation: "h", y: -0.25, x: 0, xanchor: "left", yanchor: "top" });
    }
    return f;
  }

  async function render(el) {
    const id = el.dataset.fig;
    if (!window.Plotly) return;          // CDN blocked: the <img> fallback stays visible
    let fig = specs.get(id);
    if (!fig) {
      const res = await fetch(`data/${id}.json`);
      if (!res.ok) return;
      fig = await res.json();
      specs.set(id, fig);
    }
    const f = themed(fig);
    await window.Plotly.react(el, f.data, f.layout, { responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "toImage"] });
    el.querySelector("img.fallback")?.remove();
    rendered.add(el);
  }

  function rerenderAll() { rendered.forEach((el) => render(el)); }

  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { io.unobserve(e.target); render(e.target); }
    }
  }, { rootMargin: "300px 0px" });
  document.querySelectorAll(".plot[data-fig]").forEach((el) => io.observe(el));

  // ---------- current-section highlight in the nav ----------
  const links = new Map([...document.querySelectorAll(".top nav a")].map((a) => [a.hash.slice(1), a]));
  const secObs = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        links.forEach((a) => a.removeAttribute("aria-current"));
        links.get(e.target.id)?.setAttribute("aria-current", "true");
      }
    }
  }, { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section[id]").forEach((s) => secObs.observe(s));

  setLang(root.dataset.lang || "uz");
  labelTheme();
})();

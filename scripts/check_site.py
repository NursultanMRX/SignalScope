"""Open the site in headless Chromium (clean context, no cookies), screenshot desktop + mobile in both languages and
themes, and fail on console errors or failed requests. Usage: python scripts/check_site.py <url>"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

url = sys.argv[1]
out = Path(__file__).resolve().parents[1] / "screenshots"
out.mkdir(exist_ok=True)
errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vp, scheme in [("desktop", {"width": 1440, "height": 900}, "light"),
                             ("mobile", {"width": 390, "height": 844}, "dark")]:
        ctx = b.new_context(viewport=vp, color_scheme=scheme)
        pg = ctx.new_page()
        pg.on("console", lambda m: errors.append(f"{name} console {m.type}: {m.text}") if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(f"{name} pageerror: {e}"))
        pg.on("requestfailed", lambda r: errors.append(f"{name} requestfailed: {r.url}"))
        resp = pg.goto(url, wait_until="networkidle")
        print(name, "status", resp.status)
        pg.screenshot(path=str(out / f"{name}_hero.png"))
        # scroll through the page so every lazy chart renders
        h = pg.evaluate("document.body.scrollHeight")
        for y in range(0, h, 600):
            pg.evaluate(f"window.scrollTo(0,{y})"); pg.wait_for_timeout(120)
        pg.wait_for_timeout(1500)
        n_plots = pg.evaluate("document.querySelectorAll('.plot .main-svg').length")
        n_fig = pg.evaluate("document.querySelectorAll('.plot[data-fig]').length")
        print(name, "rendered plotly charts", n_plots, "of", n_fig)
        pg.locator("#charts").scroll_into_view_if_needed(); pg.wait_for_timeout(500)
        pg.screenshot(path=str(out / f"{name}_gallery.png"))
        pg.click("#lang-toggle"); pg.wait_for_timeout(300)
        print(name, "lang after toggle", pg.evaluate("document.documentElement.lang"),
              "| visible h2:", pg.locator("#model h2 span:visible").inner_text())
        pg.click("#theme-toggle"); pg.wait_for_timeout(800)
        pg.locator("#model").scroll_into_view_if_needed(); pg.wait_for_timeout(800)
        pg.screenshot(path=str(out / f"{name}_modeling_toggled.png"))
        imgs = pg.evaluate("[...document.images].filter(i=>!i.alt).length")
        print(name, "images without alt:", imgs)
        ctx.close()
    b.close()
print("ERRORS:", errors if errors else "none")
sys.exit(1 if errors else 0)

"""Phase 8: render the static EDA website into site/ from aggregated artifacts only.

Inputs: artifacts/audit.json, artifacts/eda/{figures,stats}.json + figure JSON/PNG, artifacts/cv_report.json,
artifacts/feature_importance.csv, artifacts/experiments.jsonl. No raw rows or signal_ids are written to site/.
"""
import base64
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.offline  # noqa: E402
from jinja2 import Environment, FileSystemLoader  # noqa: E402

from signalscope.io import ROOT, load_config  # noqa: E402
from signalscope.viz import DARK_MAP, LIGHT, style  # noqa: E402

SRC = ROOT / "site_src"


def sri(url: str, cache: Path) -> str:
    """sha384 SRI hash of the pinned CDN file (cached so rebuilds are offline and deterministic)."""
    if cache.exists():
        return cache.read_text().strip()
    data = urllib.request.urlopen(url, timeout=60).read()
    h = "sha384-" + base64.b64encode(hashlib.sha384(data).digest()).decode()
    cache.write_text(h)
    return h


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def main() -> None:
    cfg = load_config()
    art = ROOT / "artifacts"
    out = ROOT / cfg["paths"]["site"]
    if out.exists():
        for p in out.iterdir():
            if p.name != ".vercel":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)

    audit = json.loads((art / "audit.json").read_text(encoding="utf-8"))
    st = json.loads((art / "eda" / "stats.json").read_text(encoding="utf-8"))
    figs = json.loads((art / "eda" / "figures.json").read_text(encoding="utf-8"))
    cv = json.loads((art / "cv_report.json").read_text(encoding="utf-8"))
    ledger = [json.loads(line) for line in (art / "experiments.jsonl").read_text(encoding="utf-8").splitlines()]
    led = {r["name"]: r for r in ledger}
    manifest = json.loads((art / "run_manifest.json").read_text(encoding="utf-8"))

    # feature importance figure (aggregated gains only)
    imp = pd.read_csv(art / "feature_importance.csv").head(20)[::-1]
    fig = go.Figure(go.Bar(x=imp["gain"], y=imp["feature"], orientation="h", marker_color=LIGHT["dismissed"],
                           hovertemplate="%{y}: %{x:.0f}<extra></extra>"))
    style(fig, "", x="LightGBM gain (to'liq train / full-train refit)", height=560)
    fig.update_yaxes(tickfont={"size": 11})
    (art / "eda" / "f14_importance.json").write_text(fig.to_json(), encoding="utf-8")
    fig.write_image(art / "eda" / "f14_importance.png", width=1000, height=560, scale=1.5)
    figs.append({"id": "f14_importance", "title_uz": "Eng muhim 20 belgi (LightGBM gain)",
                 "title_en": "Top 20 features (LightGBM gain)",
                 "caption_uz": "Eng muhim belgilar — tur bo'yicha miqdor statistikasi (karta minimumi, naqd kirim o'rtachasi, "
                               "bank o'tkazmasi chiqimlari o'rtachasi va dispersiyasi) va miqdor gistogrammasi ulushlari.",
                 "caption_en": "The most important features are per-type amount statistics (card minimum, cash-in mean, "
                               "bank-transfer outgoing mean and spread) and amount-histogram shares.",
                 "alt": "Horizontal bar chart of LightGBM gain for the 20 most important features."})

    for f in figs:
        shutil.copy(art / "eda" / f"{f['id']}.json", out / "data" / f"{f['id']}.json")
        shutil.copy(art / "eda" / f"{f['id']}.png", out / "img" / f"{f['id']}.png")
        spec = json.loads((art / "eda" / f"{f['id']}.json").read_text(encoding="utf-8"))
        f["h"] = int(1.5 * (spec.get("layout", {}).get("height") or 420))
    shutil.copy(art / "eda" / "f07_amount_ratio.png", out / "img" / "og.png")
    byid = {f["id"]: f for f in figs}

    m_lgb = cv["models"]["lgb"]
    w = manifest["weights"]
    headline = cv["blend"]["oof_auc"] if len(w) > 1 else cv["models"][next(iter(w))]["oof_auc_of_mean_pred"]
    m = {"headline": f"{headline:.3f}", "fold_mean": f"{m_lgb['fold_auc_mean']:.3f}",
         "fold_std": f"{m_lgb['fold_auc_std']:.3f}"}
    tr, te = audit["tx_train"], audit["tx_test"]
    feats = pd.read_parquet(ROOT / cfg["paths"]["processed"] / "features_train.parquet", columns=None).shape[1] - 1
    n = {"train_signals": audit["signals_train"]["rows"], "train_signals_fmt": fmt(audit["signals_train"]["rows"]),
         "test_signals_fmt": fmt(audit["signals_test"]["rows"]), "train_tx_fmt": fmt(tr["rows"]),
         "test_tx_fmt": fmt(te["rows"]), "all_tx_fmt": fmt(tr["rows"] + te["rows"]),
         "rate_pct": f"{audit['target']['rate']:.1%}", "date_min": audit["signals_train"]["date_min"],
         "date_max": audit["signals_train"]["date_max"], "tx_min": tr["ts_min"][:10], "tx_max": tr["ts_max"][:10],
         "tx_mean": round(tr["tx_per_signal"]["mean"]), "tx_median": round(tr["tx_per_signal"]["q50"]),
         "post_train": fmt(tr["post_signal_rows"]), "post_test": fmt(te["post_signal_rows"]),
         "id_auc": f"{audit['target']['signal_id_order_auc']:.3f}",
         "adv_auc": f"{audit['adversarial_validation']['auc']:.3f}", "n_features": feats}

    bk = st["amount_by_type"]["bank_otkazmasi"]
    ev = st["event_time"]
    bu = st["burst"]
    win_max = max(abs(v["cliffs_delta"]) for v in st["windows_rate_vs_hist"].values())
    top1 = list(st["univariate_top10"].items())[0]
    CAP = {
        "f01_target": ("Chapda: sinflar. O'ngda: oylik eskalatsiya ulushi, soya 95% ishonch oralig'i. "
                       f"Oylar orasidagi farq tasodifiy darajada (χ² p = {st['target']['month_chi2_p']:.2f}), shuning uchun "
                       "oy yoki hafta kunini belgi qilib qo'shmadik.",
                       "Left: class counts. Right: monthly escalation rate with a 95% band. The month-to-month "
                       f"differences look like noise (chi-square p = {st['target']['month_chi2_p']:.2f}), so we did not add "
                       "month or weekday as features."),
        "f02_history": (f"Eskalatsiya qilinganlarda tranzaksiya biroz ko'proq (median {st['n_tx']['median_escalated']:.0f} ta, "
                        f"rad etilganlarda {st['n_tx']['median_dismissed']:.0f}). Farq bor, lekin kichik. Deyarli hamma tarix "
                        "signaldan oldingi ~180 kunni qamraydi.",
                        f"Escalated alerts have slightly more transactions (median {st['n_tx']['median_escalated']:.0f} vs "
                        f"{st['n_tx']['median_dismissed']:.0f}). Real, but small. Nearly every history covers about 180 days "
                        "before the alert."),
        "f04_burst": (f"Buni tasodifan topdik: barcha tranzaksiyalarning {bu['share_of_rows']:.1%} qismi signal sanasidan oldingi "
                      f"kechada 23:57 va 23:59 orasida turibdi (signallarning {bu['signals_with_burst']:.0%} qismida, har birida "
                      f"median {bu['median_per_alert']:.0f} ta). Bu generator artefakti bo'lsa kerak. Uni alohida belgilar sifatida "
                      "hisobladik, CV biroz yaxshilandi (+0.007).",
                      f"We found this by accident: {bu['share_of_rows']:.1%} of all transactions sit between 23:57 and 23:59 on "
                      f"the evening before the alert date ({bu['signals_with_burst']:.0%} of alerts, median "
                      f"{bu['median_per_alert']:.0f} each). Probably an artefact of the data generator. We summarise it as its "
                      "own feature group, which helped CV a little (+0.007)."),
        "f05_direction": ("Kirim ulushi va sof oqim ikkala sinfda deyarli bir xil. Kutgandik bu muhim bo'ladi, bo'lmadi.",
                          "Incoming share and net flow look almost the same for both classes. We expected these to matter; "
                          "they did not."),
        "f06_types": ("Tur tarkibi ham ikkala sinfda bir xil. O'ngdagi jadval: eskalatsiya qilinganlardagi ulush / rad "
                      "etilganlardagi ulush. Hammasi 1 atrofida.",
                      "The type mix is the same for both classes too. Right panel: share among escalated divided by share "
                      "among dismissed. Everything is close to 1."),
        "f07_amount_ratio": ("Bizningcha eng qiziq grafik. Har bir miqdor oralig'ida eskalatsiya qilinganlar zichligini rad "
                             "etilganlarnikiga bo'ldik. Bank o'tkazmalarida chiziq pastga tushadi: mayda o'tkazmalar "
                             "eskalatsiyada ko'proq, yiriklari kamroq. Kartada ham shunga o'xshash, naqd pulda farq yo'q.",
                             "Our favourite chart. For each amount bin we divided the escalated density by the dismissed one. "
                             "For bank transfers the line slopes down: small transfers are over-represented among escalated "
                             "alerts, large ones under-represented. Cards show a weaker version; cash shows nothing."),
        "f08_amount_types": ("Miqdor darajasi turga qarab juda farq qiladi (xalqaro > naqd > bank > karta). Shuning uchun "
                             "miqdor statistikalarini har bir tur ichida alohida hisobladik.",
                             "Amount level depends heavily on type (international > cash > transfer > card), so we compute "
                             "amount statistics separately inside each type."),
        "f10_time_of_day": ("Soatlar bo'yicha taqsimot deyarli tekis, bu sintetik ma'lumotga xos. 23-soatdagi ustun faqat "
                            "3 daqiqalik portlash hisobiga. Hafta kunlari ham ikkala sinfda bir xil.",
                            "Transactions are spread almost evenly over the day, which is typical for synthetic data. The "
                            "23:00 bar is entirely the 3-minute burst. Weekdays look the same for both classes."),
        "f11_univariate": (f"Har bir belgining o'zi qanchalik ajratadi. Eng yaxshisi ({top1[0]}) ham atigi "
                           f"{max(top1[1], 1 - top1[1]):.3f}. To'q sariq = eskalatsiya bilan musbat bog'liq. O'ngda: kuchli "
                           "belgilar bir-biri bilan qattiq korrelyatsiyalangan.",
                           f"How well each feature separates the classes on its own. Even the best one ({top1[0]}) only "
                           f"gets {max(top1[1], 1 - top1[1]):.3f}. Orange = positively related to escalation. Right: the strong "
                           "features are heavily correlated with each other."),
        "f12_drift": (f"Train va test taqsimotlari ustma-ust tushadi (eng katta KS = {st['drift']['ks_max_top25']:.3f}).",
                      f"Train and test distributions overlap (largest KS = {st['drift']['ks_max_top25']:.3f})."),
        "f03_event_time": (f"Signalgacha har kuni o'rtacha nechta tranzaksiya. Eskalatsiya qilinganlar butun davr davomida "
                           f"~{(ev['escalated_to_dismissed_volume_ratio'] - 1) * 100:.0f}% faolroq, lekin egri chiziq shakli "
                           "bir xil va signal oldidan sakrash yo'q. Qizig'i, faollik signalga yaqinlashgan sari kamayadi.",
                           f"Average transactions per day before the alert. Escalated customers are about "
                           f"{(ev['escalated_to_dismissed_volume_ratio'] - 1) * 100:.0f}% more active throughout, but the curves "
                           "have the same shape and there is no spike before the alert. Oddly, activity drops as the alert "
                           "gets closer."),
        "f09_windows": (f"Oxirgi 1 dan 90 kungacha bo'lgan faollik tarixiy o'rtachaga nisbatan. Ikkala sinf deyarli bir xil "
                        f"(eng katta Cliff δ = {win_max:.3f}).",
                        f"Activity in the last 1 to 90 days relative to the whole history. Both classes look nearly the "
                        f"same (largest Cliff's delta = {win_max:.3f})."),
        "f13_ablation": ("Guruhni olib tashlaganda CV qancha tushadi (LightGBM, 5×2 CV). Agregatlar asosiy ishni qiladi. "
                         "Vaqt naqshlari va pass-through foyda bermadi, ularni yakuniy modeldan olib tashladik.",
                         "How much CV drops when a group is removed (LightGBM, 5×2 CV). Aggregates do most of the work. "
                         "Time patterns and pass-through did not help, so we dropped them from the final model."),
        "f14_importance": ("Yakuniy LightGBM'da eng ko'p ishlatilgan belgilar. Ko'pchiligi tur bo'yicha miqdor "
                           "statistikasi: karta minimumi, naqd kirim o'rtachasi, bank o'tkazmalari o'rtachasi va tarqoqligi.",
                           "Features the final LightGBM leans on most. Mostly per-type amount statistics: card minimum, "
                           "cash-in mean, bank-transfer mean and spread."),
    }
    for i, f in enumerate(["f01_target", "f02_history", "f04_burst", "f05_direction", "f06_types", "f07_amount_ratio",
                           "f08_amount_types", "f10_time_of_day", "f11_univariate", "f12_drift", "f03_event_time",
                           "f09_windows", "f13_ablation", "f14_importance"], start=1):
        byid[f]["caption_uz"], byid[f]["caption_en"] = CAP[f]
        byid[f]["num"] = i

    observations = [
        {"uz": "Eskalatsiya qilinganlarda mayda bank o'tkazmalari ko'proq, yiriklari kamroq. Ma'lumotdagi eng aniq signal shu.",
         "en": "Escalated alerts have more small and fewer large bank transfers. This is the clearest signal in the data.",
         "ev": f"median bank-transfer mean {bk['median_escalated']:.2f} vs {bk['median_dismissed']:.2f}, Cliff's δ = "
               f"{bk['cliffs_delta']:.3f}, Mann-Whitney p = {bk['p']:.0e}. Fig. 6."},
        {"uz": "Ular umuman olganda biroz faolroq, taxminan 6%, va bu butun 180 kun davomida saqlanadi.",
         "en": "They are a bit more active overall, about 6%, and it holds over the whole 180 days.",
         "ev": f"volume ratio {ev['escalated_to_dismissed_volume_ratio']:.3f}; n_tx Cliff's δ = {st['n_tx']['cliffs_delta']:.3f}. "
               "Fig. 11."},
        {"uz": "Signal oldidan \"sakrash\" yo'q. Buni ko'p tekshirdik, chunki birinchi gipotezamiz shu edi.",
         "en": "There is no spike right before the alert. We checked this a lot because it was our first hypothesis.",
         "ev": f"largest |Cliff's δ| over 1 to 90-day windows = {win_max:.3f}. Fig. 12."},
        {"uz": "Tranzaksiyalarning bir qismi signal sanasidan oldingi kechaning oxirgi 3 daqiqasiga to'plangan.",
         "en": "A chunk of transactions is squeezed into the last 3 minutes before midnight of the alert date.",
         "ev": f"{bu['share_of_rows']:.1%} of rows, {bu['signals_with_burst']:.0%} of alerts. Fig. 3."},
        {"uz": "Tur va yo'nalish (kirim/chiqim) tarkibi ikkala sinfda deyarli bir xil.",
         "en": "Type and direction mix is basically the same for both classes.",
         "ev": f"transaction-level Cramér's V = {st['type_mix']['cramers_v_tx_level']:.4f}. Figs. 4, 5."},
        {"uz": "Tungi va dam olish kuni operatsiyalari farq qilmaydi.",
         "en": "Night-time and weekend activity make no difference.",
         "ev": "ablation: time-pattern group ΔAUC ≈ 0. Fig. 8, Fig. 13."},
        {"uz": "Bitta belgi bilan ish bitmaydi: hech biri AUC 0.58 dan oshmaydi.",
         "en": "No single feature does the job: none of them goes above 0.58 AUC.",
         "ev": f"{st['univariate_n_over_055']} features above 0.55. Fig. 9."},
    ]
    abl = st["ablation_delta_auc"]
    feature_table = [
        {"uz": "Miqdor turga qarab farq qiladi; bank o'tkazmalari eng kuchli",
         "en": "Amount depends on type; bank transfers matter most",
         "feat": "<type>_mean/std/min/max, <type>_<dir>_*", "delta": f"+{abl['Agregatlar / aggregates']:.4f}"},
        {"uz": "Mayda va yirik o'tkazmalar nisbati", "en": "Share of small vs large transfers",
         "feat": "<type>_<dir>_bin0..7_frac", "delta": f"+{abl['Miqdor gistogrammasi / amount histogram']:.4f}"},
        {"uz": "Oxirgi tranzaksiyalar", "en": "The last few transactions",
         "feat": "last0..19_amt/age/in/type", "delta": f"+{abl['Oxirgi 20 tx / last-20 tx']:.4f}"},
        {"uz": "3 daqiqalik portlash", "en": "The 3-minute burst",
         "feat": "b_*, br_* (burst vs history)", "delta": f"+{abl['Portlash / burst']:.4f}"},
        {"uz": "Oxirgi kunlardagi faollik", "en": "Recent activity",
         "feat": "w1..w90_*, accel_*", "delta": f"+{abl['Oynalar / windows']:.4f}"},
        {"uz": "Tungi / dam olish kunlari, oraliqlar (olib tashlandi)", "en": "Night / weekend, gaps (dropped)",
         "feat": "hour/weekday shares, gap_*", "delta": f"{abl['Vaqt naqshlari / time patterns']:+.4f}"},
        {"uz": "Pass-through: kirdi, tez chiqdi (olib tashlandi)", "en": "Pass-through: money in, quickly out (dropped)",
         "feat": "pt_*", "delta": f"{abl['Pass-through']:+.4f}"},
    ]
    names = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "lr": "Logistic regression"}
    results = [{"name": names[k], "fold": f"{v['fold_auc_mean']:.4f} ± {v['fold_auc_std']:.4f}",
                "oof": f"{v['oof_auc_of_mean_pred']:.4f}"} for k, v in cv["models"].items()]
    fw = cv["blend"]["foldwise"]
    best_single = max(v for k, v in fw.items() if k.startswith("single_"))
    results.append({"name": "Blend (" + ", ".join(f"{k} {v:.2f}" for k, v in cv["blend"]["weights"].items()) + ")",
                    "fold": "", "oof": f"{cv['blend']['oof_auc']:.4f}"})
    chosen = " + ".join(names[k] for k in w)
    ens = {"uz": f"Modellarni aralashtirib ko'rdik (OOF {cv['blend']['oof_auc']:.4f}). Lekin og'irliklarni 4 foldda topib, "
                 f"5-foldda tekshirsak, blend {fw['fitted']:.4f}, eng yaxshi bitta model esa {best_single:.4f}. Farq yo'q "
                 f"darajada, shuning uchun sodda variantni tanladik: {chosen}, butun train'da 5 xil seed bilan o'qitilgan.",
           "en": f"We tried blending the models (OOF {cv['blend']['oof_auc']:.4f}). But when the weights are fitted on 4 folds "
                 f"and checked on the 5th, the blend gets {fw['fitted']:.4f} and the best single model {best_single:.4f}. "
                 f"That is no real difference, so we went with the simpler option: {chosen}, trained on all of train "
                 "with 5 different seeds."}

    def g(name, default=None):
        r = led.get(name)
        return f"{r['oof_auc_mean_over_repeats']:.4f}" if r else default
    negatives = [
        {"name": "GRU over the last 1024 transactions (idea from AMEX 1st place)", "auc": g("seqnn_gru_L1024")},
        {"name": "Transaction-level model, then averaged per alert", "auc": "0.555"},
        {"name": "Naive Bayes on binned features (Santander trick)", "auc": g("nb_20bins")},
        {"name": "LightGBM with 2 leaves (Santander)", "auc": g("lgb_2leaves")},
        {"name": "LightGBM DART (AMEX)", "auc": g("dart")},
        {"name": "Direction/type transitions, velocity, deltas", "auc": g("tuned_trans")},
        {"name": "Recent vs earlier behaviour shift", "auc": g("with_shift")},
        {"name": "Removing the weekly trend from amounts", "auc": g("fixed_detrend")},
        {"name": "Alert date as a feature", "auc": g("fixed_calendar")},
    ]
    negatives = [r for r in negatives if r["auc"]]

    ver = plotly.offline.get_plotlyjs_version()
    url = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{ver}/plotly.min.js"
    env = Environment(loader=FileSystemLoader(SRC), autoescape=True)
    html = env.get_template("template.html").render(
        n=n, m=m, figs=byid, gallery=[byid[k] for k in ["f02_history", "f04_burst", "f05_direction", "f06_types",
                                                        "f07_amount_ratio", "f08_amount_types", "f10_time_of_day",
                                                        "f11_univariate", "f12_drift"]],
        observations=observations, feature_table=feature_table, results=results, ens=ens, negatives=negatives,
        dark_map={k.lower(): v for k, v in DARK_MAP.items()}, plotly_version=ver,
        plotly_sri=sri(url, art / "cache" / f"plotly_{ver}.sri"), team_name=cfg.get("team_name", cfg["team_id"]),
        notebook_name=f"notebooks/team_{cfg['team_id']}_reproducible.ipynb", repo_url=cfg.get("repo_url"),
        build_date=date.today().isoformat())
    html = html.replace('<caption class="sr-only"></caption>', "")
    (out / "index.html").write_text(html, encoding="utf-8")
    shutil.copy(SRC / "styles.css", out / "styles.css")
    shutil.copy(SRC / "app.js", out / "app.js")
    (out / "favicon.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="7" fill="#2a78d6"/>'
        '<path d="M6 22 L12 14 L17 18 L26 8" stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round" '
        'stroke-linejoin="round"/><circle cx="26" cy="8" r="3" fill="#eb6834"/></svg>', encoding="utf-8")
    (out / "vercel.json").write_text(json.dumps({"cleanUrls": True, "headers": [
        {"source": "/data/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=3600"}]}]}, indent=1))
    # privacy guard: no signal ids or raw transaction rows may leak into the site
    blob = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in out.rglob("*") if p.suffix in
                   {".html", ".json", ".js", ".css"})
    assert not re.search(r"SG_\d{6}", blob), "signal_id found in site output"
    print("site built ->", out, "figures:", len(figs))


if __name__ == "__main__":
    main()

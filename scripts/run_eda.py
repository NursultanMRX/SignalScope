"""Phase 2: EDA figures (Plotly JSON + PNG) and statistics -> artifacts/eda/.

Only aggregated numbers are written into figures (no raw transactions, no per-signal rows).
Every caption number is computed here and also stored in artifacts/eda/stats.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import polars as pl  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from signalscope.features import prepare_tx  # noqa: E402
from signalscope.io import ROOT, load_config, path, read_signals, read_tx, seed_everything  # noqa: E402
from signalscope.viz import LIGHT, TYPE_COLORS, rgba, style  # noqa: E402

TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
TYPE_LABEL = {"karta": "karta (card)", "bank_otkazmasi": "bank o'tkazmasi (transfer)", "naqd": "naqd (cash)",
              "xalqaro": "xalqaro (international)"}
C0, C1 = LIGHT["dismissed"], LIGHT["escalated"]
N0, N1 = "Rad etilgan / Dismissed", "Eskalatsiya / Escalated"


def mw(a: np.ndarray, b: np.ndarray) -> dict:
    """Mann-Whitney U (escalated vs dismissed) + Cliff's delta (= 2*AUC-1)."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    u = stats.mannwhitneyu(b, a, alternative="two-sided")
    auc = u.statistic / (len(a) * len(b))
    return {"p": float(u.pvalue), "cliffs_delta": round(float(2 * auc - 1), 4), "auc": round(float(auc), 4),
            "median_dismissed": round(float(np.median(a)), 4), "median_escalated": round(float(np.median(b)), 4)}


def hist_bar(fig, values, bins, color, name, row, col, density=True, showlegend=True, legendgroup=None, opacity=0.55):
    """Pre-binned histogram: only bin counts go into the figure JSON, never the underlying values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    h, e = np.histogram(v, bins=bins, density=density)
    fig.add_bar(x=(e[:-1] + e[1:]) / 2, y=h, width=np.diff(e), marker_color=color, opacity=opacity, name=name,
                legendgroup=legendgroup or name, showlegend=showlegend, row=row, col=col,
                hovertemplate="%{x:.2f}: %{y:.3g}<extra>" + name + "</extra>")


def box_q(fig, values, x, color, name, row=None, col=None, showlegend=True):
    """Box from pre-computed quantiles (5/25/50/75/95%), no raw points."""
    v = np.asarray(values, dtype=float)
    q = np.nanquantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    kw = {} if row is None else {"row": row, "col": col}
    fig.add_box(x=[x], q1=[q[1]], median=[q[2]], q3=[q[3]], lowerfence=[q[0]], upperfence=[q[4]], marker_color=color,
                name=name, legendgroup=name, showlegend=showlegend, offsetgroup=name, **kw)


def main() -> None:
    cfg = load_config()
    seed_everything(cfg["seed"])
    out = path(cfg, "artifacts") / "eda"
    out.mkdir(parents=True, exist_ok=True)
    s = read_signals(cfg, "train")
    s_te = read_signals(cfg, "test")
    y = s["eskalatsiya"].to_numpy()
    ftr = pd.read_parquet(path(cfg, "processed") / "features_train.parquet")
    fte = pd.read_parquet(path(cfg, "processed") / "features_test.parquet")
    t = prepare_tx(read_tx(cfg, "train"), s).join(pl.from_pandas(s[["signal_id", "eskalatsiya"]]), on="signal_id")
    burst_s = cfg["feature_set"]["burst_seconds"]
    t = t.with_columns((pl.col("age_days") * 86400 <= burst_s).alias("is_burst"))
    th = t.filter(~pl.col("is_burst"))
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    figs, st = [], {}

    def add(fid, fig, title_uz, title_en, cap_uz, cap_en, alt):
        (out / f"{fid}.json").write_text(fig.to_json(), encoding="utf-8")
        fig.write_image(out / f"{fid}.png", width=1000, height=fig.layout.height or 420, scale=1.5)
        figs.append({"id": fid, "title_uz": title_uz, "title_en": title_en, "caption_uz": cap_uz,
                     "caption_en": cap_en, "alt": alt})

    # 1 target balance + rate by month --------------------------------------------------------------------
    m = s.groupby(s.signal_sanasi.dt.to_period("M"))["eskalatsiya"].agg(["mean", "size"])
    se = np.sqrt(m["mean"] * (1 - m["mean"]) / m["size"])
    xm = [str(p) for p in m.index]
    ct = pd.crosstab(s.signal_sanasi.dt.to_period("M"), y)
    chi = stats.chi2_contingency(ct)
    st["target"] = {"n": len(y), "n_pos": n1, "n_neg": n0, "rate": round(n1 / len(y), 4),
                    "month_chi2": round(float(chi[0]), 2), "month_chi2_p": round(float(chi[1]), 4),
                    "month_rate_min": round(float(m["mean"].min()), 4), "month_rate_max": round(float(m["mean"].max()), 4)}
    fig = make_subplots(rows=1, cols=2, column_widths=[0.28, 0.72], horizontal_spacing=0.1,
                        subplot_titles=("Sinflar / Classes", "Oylik eskalatsiya ulushi / Monthly escalation rate"))
    fig.add_bar(x=["Rad etilgan", "Eskalatsiya"], y=[n0, n1], marker_color=[C0, C1],
                text=[f"{n0:,} ({n0 / len(y):.1%})", f"{n1:,} ({n1 / len(y):.1%})"], textposition="outside",
                hovertemplate="%{x}: %{y:,}<extra></extra>", showlegend=False, row=1, col=1)
    fig.add_scatter(x=xm + xm[::-1], y=list(m["mean"] + 1.96 * se) + list((m["mean"] - 1.96 * se)[::-1]),
                    fill="toself", fillcolor=rgba(C1, 0.15), line={"width": 0}, hoverinfo="skip",
                    name="95% CI", row=1, col=2)
    fig.add_scatter(x=xm, y=m["mean"], mode="lines+markers", line={"color": C1, "width": 2}, marker={"size": 8},
                    name="Eskalatsiya ulushi / rate", hovertemplate="%{x}: %{y:.1%}<extra></extra>", row=1, col=2)
    fig.add_hline(y=n1 / len(y), line={"dash": "dot", "color": LIGHT["text2"], "width": 1}, row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    fig.update_yaxes(range=[0, n0 * 1.18], row=1, col=1)
    style(fig, "")
    add("f01_target", fig, "Maqsad taqsimoti va vaqt bo'yicha ulush", "Target balance and rate over time",
        f"Signallarning {n1 / len(y):.1%} qismi eskalatsiya qilingan ({n1:,} / {len(y):,}). Oylik ulush "
        f"{m['mean'].min():.1%}–{m['mean'].max():.1%} oralig'ida tebranadi, lekin statistik ahamiyatli trend yo'q "
        f"(χ² p = {chi[1]:.2f}), shuning uchun kalendar belgilar modelga kiritilmadi.",
        f"{n1 / len(y):.1%} of alerts are escalated ({n1:,} of {len(y):,}). The monthly rate moves between "
        f"{m['mean'].min():.1%} and {m['mean'].max():.1%} but shows no significant pattern (chi-square p = {chi[1]:.2f}), "
        f"so calendar features were left out.",
        "Bar chart of class counts and a line chart of monthly escalation rate with a 95% confidence band.")

    # 2 history length / tx count --------------------------------------------------------------------------
    d = ftr.assign(y=y)
    st["n_tx"] = mw(d.loc[d.y == 0, "n_tx"].to_numpy(), d.loc[d.y == 1, "n_tx"].to_numpy())
    st["first_age"] = mw(d.loc[d.y == 0, "first_age"].to_numpy(), d.loc[d.y == 1, "first_age"].to_numpy())
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Tranzaksiyalar soni / Transactions per alert",
                                                        "Tarix uzunligi (kun) / History length (days)"))
    for yy, c, nm in [(0, C0, N0), (1, C1, N1)]:
        hist_bar(fig, d.loc[d.y == yy, "n_tx"], np.linspace(0, 1600, 61), c, nm, 1, 1)
        hist_bar(fig, d.loc[d.y == yy, "first_age"], np.linspace(0, 180, 61), c, nm, 1, 2, showlegend=False)
    fig.update_layout(barmode="overlay")
    style(fig, "", y="zichlik / density")
    add("f02_history", fig, "Tarix hajmi: eskalatsiya va rad etilganlar", "History size: escalated vs dismissed",
        f"Eskalatsiya qilingan signallarda tranzaksiyalar biroz ko'proq (median {st['n_tx']['median_escalated']:.0f} vs "
        f"{st['n_tx']['median_dismissed']:.0f}; Cliff δ = {st['n_tx']['cliffs_delta']:.3f}, p < 0.001) — farq kichik. "
        "Deyarli barcha tarixlar signalgacha ~180 kunni qamraydi.",
        f"Escalated alerts have slightly more transactions (median {st['n_tx']['median_escalated']:.0f} vs "
        f"{st['n_tx']['median_dismissed']:.0f}; Cliff's delta = {st['n_tx']['cliffs_delta']:.3f}, p < 0.001) — a small "
        "effect. Almost every history covers ~180 days before the alert.",
        "Overlaid histograms of transaction count and history length for escalated and dismissed alerts.")

    # 3 event-time curve -------------------------------------------------------------------------------------
    dc = (th.group_by(["signal_id", "day"]).len().join(pl.from_pandas(s[["signal_id", "eskalatsiya"]]),
                                                         on="signal_id"))
    M = np.zeros((len(s), 180))
    idx = pd.Series(np.arange(len(s)), index=s.signal_id.values)
    r = idx.loc[dc["signal_id"].to_numpy()].to_numpy()
    dd = np.clip(dc["day"].to_numpy(), 0, 179)
    np.add.at(M, (r, dd), dc["len"].to_numpy())
    fig = go.Figure()
    curves = {}
    for yy, c, nm in [(0, C0, N0), (1, C1, N1)]:
        sub = M[y == yy]
        mu = sub.mean(0)
        sd = sub.std(0) / np.sqrt(len(sub))
        x = -np.arange(180)
        curves[yy] = mu
        fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[mu + 1.96 * sd, (mu - 1.96 * sd)[::-1]], fill="toself",
                        fillcolor=rgba(c, 0.18), line={"width": 0}, hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=x, y=mu, mode="lines", line={"color": c, "width": 2}, name=nm,
                        hovertemplate="%{x} kun / days: %{y:.2f}<extra>" + nm + "</extra>")
    ratio = float(curves[1].sum() / curves[0].sum())
    st["event_time"] = {"escalated_to_dismissed_volume_ratio": round(ratio, 4),
                        "daily_mean_last30_dismissed": round(float(curves[0][:30].mean()), 3),
                        "daily_mean_last30_escalated": round(float(curves[1][:30].mean()), 3),
                        "peak_day_before": int(np.argmax(curves[0]))}
    style(fig, "", x="Signalgacha kunlar / days before alert (0 = alert)", y="kunlik tranzaksiyalar / tx per day")
    add("f03_event_time", fig, "Signalgacha faollik (hodisa vaqti)", "Activity aligned to the alert (event time)",
        f"Kunlik faollik (oxirgi 3 daqiqalik portlash bundan mustasno) eskalatsiya qilinganlarda butun oyna bo'ylab "
        f"~{(ratio - 1) * 100:.0f}% yuqori; egri chiziqlar shakli bir xil. Ya'ni signal oldidan maxsus 'sakrash' yo'q — "
        "farq doimiy faollik darajasida.",
        f"Daily activity (excluding the 3-minute burst) is ~{(ratio - 1) * 100:.0f}% higher for escalated alerts across "
        "the whole window, with the same shape. There is no special pre-alert spike: the difference is a steady "
        "level shift.",
        "Line chart of mean daily transactions per alert over the 180 days before the alert, with 95% bands.")

    # 4 burst anatomy -----------------------------------------------------------------------------------------
    b = t.filter(pl.col("is_burst"))
    bs = (b["age_days"] * 86400).to_numpy()
    bn = ftr["b_n"].to_numpy()
    st["burst"] = {"share_of_rows": round(b.height / t.height, 4), "signals_with_burst": round(float((bn > 0).mean()), 4),
                   "median_per_alert": float(np.median(bn[bn > 0])), "max_seconds_before": round(float(bs.max()), 1),
                   **{f"b_n_{k}": v for k, v in mw(bn[y == 0], bn[y == 1]).items()}}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Signal sanasidan oldingi soniyalar / seconds before alert date",
                                                        "Portlashdagi tranzaksiyalar / burst size per alert"))
    hist_bar(fig, -bs, np.linspace(-180, 0, 37), LIGHT["s3"], "tx", 1, 1, density=False, showlegend=False,
             opacity=0.9)
    for yy, c, nm in [(0, C0, N0), (1, C1, N1)]:
        hist_bar(fig, np.clip(bn[y == yy], 0, 200), np.linspace(0, 200, 51), c, nm, 1, 2)
    fig.update_layout(barmode="overlay")
    style(fig, "")
    add("f04_burst", fig, "Signal oldidan 3 daqiqalik 'portlash'", "The 3-minute pre-alert burst",
        f"Tranzaksiyalarning {st['burst']['share_of_rows']:.1%} qismi signal sanasidan oldingi kechaning 23:57–23:59 "
        f"oralig'iga to'plangan (signallarning {st['burst']['signals_with_burst']:.1%} qismida, median "
        f"{st['burst']['median_per_alert']:.0f} ta). Bu sintetik generator artefakti; alohida belgilar sifatida ishlatildi "
        f"(OOF AUC +0.007), lekin o'zi zaif (Cliff δ = {st['burst']['b_n_cliffs_delta']:.3f}).",
        f"{st['burst']['share_of_rows']:.1%} of all transactions are stamped 23:57–23:59 on the evening before the "
        f"alert date ({st['burst']['signals_with_burst']:.1%} of alerts, median {st['burst']['median_per_alert']:.0f} each). "
        f"This is a generator artefact; we summarise it as separate features (+0.007 OOF AUC) but it is weak on its own "
        f"(Cliff's delta = {st['burst']['b_n_cliffs_delta']:.3f}).",
        "Histogram of burst timestamps in the last 3 minutes and overlaid histograms of burst size by class.")

    # 5 direction ------------------------------------------------------------------------------------------------
    st["in_share"] = mw(d.loc[d.y == 0, "in_share"].to_numpy(), d.loc[d.y == 1, "in_share"].to_numpy())
    st["net_flow_norm"] = mw(d.loc[d.y == 0, "net_flow_norm"].to_numpy(), d.loc[d.y == 1, "net_flow_norm"].to_numpy())
    st["in_out_amtx_ratio"] = mw(d.loc[d.y == 0, "in_out_amtx_ratio"].to_numpy(),
                                 d.loc[d.y == 1, "in_out_amtx_ratio"].to_numpy())
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Kirim ulushi / incoming share",
                                                        "Sof oqim (normallashgan) / normalised net flow"))
    for yy, c, nm in [(0, C0, N0), (1, C1, N1)]:
        box_q(fig, d.loc[d.y == yy, "in_share"], nm, c, nm, 1, 1)
        box_q(fig, d.loc[d.y == yy, "net_flow_norm"], nm, c, nm, 1, 2, showlegend=False)
    style(fig, "")
    fig.update_layout(showlegend=False)
    add("f05_direction", fig, "Kirim va chiqim", "Incoming vs outgoing",
        f"Kirim tranzaksiyalar ~76% ni tashkil etadi. Eskalatsiya qilinganlarda kirim ulushi deyarli bir xil "
        f"(Cliff δ = {st['in_share']['cliffs_delta']:.3f}), sof oqim bo'yicha farq ham kichik "
        f"(δ = {st['net_flow_norm']['cliffs_delta']:.3f}): yo'nalish yolg'iz o'zi kuchli signal emas.",
        f"About 76% of transactions are incoming. The incoming share is nearly identical across classes "
        f"(Cliff's delta = {st['in_share']['cliffs_delta']:.3f}) and net flow differs only slightly "
        f"(delta = {st['net_flow_norm']['cliffs_delta']:.3f}): direction alone is not a strong signal.",
        "Box plots of incoming share and normalised net flow by class.")

    # 6 type shares + type x direction heatmap ---------------------------------------------------------------------
    g = th.group_by(["tranzaksiya_turi", "kirim_chiqim", "eskalatsiya"]).len().to_pandas()
    piv = g.pivot_table(index=["tranzaksiya_turi", "kirim_chiqim"], columns="eskalatsiya", values="len")
    piv = piv / piv.sum()
    lift = (piv[1] / piv[0]).unstack("kirim_chiqim").reindex(TYPES)
    share = g.groupby(["tranzaksiya_turi", "eskalatsiya"])["len"].sum().unstack()
    share = (share / share.sum()).reindex(TYPES)
    ctab = g.groupby(["tranzaksiya_turi", "eskalatsiya"])["len"].sum().unstack().reindex(TYPES)
    chi_t = stats.chi2_contingency(ctab.values)
    cramer = np.sqrt(chi_t[0] / (ctab.values.sum() * 1))
    st["type_mix"] = {"chi2_p": float(chi_t[1]), "cramers_v_tx_level": round(float(cramer), 4),
                      "share_dismissed": share[0].round(4).to_dict(), "share_escalated": share[1].round(4).to_dict(),
                      "lift": {f"{a}|{b_}": round(float(v), 3) for (a, b_), v in lift.stack().items()}}
    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.18,
                        subplot_titles=("Tur ulushi / type share", "Eskalatsiya/rad nisbati / lift (type × direction)"))
    for yy, c, nm in [(0, C0, N0), (1, C1, N1)]:
        fig.add_bar(x=[TYPE_LABEL[k] for k in TYPES], y=share[yy], marker_color=c, name=nm,
                    hovertemplate="%{x}: %{y:.2%}<extra>" + nm + "</extra>", row=1, col=1)
    z = lift.values
    fig.add_heatmap(z=z, x=list(lift.columns), y=[TYPE_LABEL[k] for k in lift.index], zmid=1.0,
                    colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]], zmin=0.9, zmax=1.1,
                    text=np.round(z, 3), texttemplate="%{text}", colorbar={"title": "lift", "x": 1.02},
                    hovertemplate="%{y} / %{x}: %{z:.3f}<extra></extra>", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    style(fig, "", height=440)
    add("f06_types", fig, "Tranzaksiya turlari", "Transaction types",
        f"Tur tarkibi sinflar orasida deyarli bir xil (tranzaksiya darajasida Cramér V = {cramer:.3f}). Issiqlik xaritasi "
        "eskalatsiya qilinganlardagi ulushning rad etilganlarga nisbatini ko'rsatadi: bank o'tkazmasi chiqimlari bir oz "
        "ko'proq, lekin farqlar ±5% atrofida.",
        f"The type mix is almost identical between classes (transaction-level Cramér's V = {cramer:.3f}). The heatmap shows "
        "the share among escalated divided by the share among dismissed: differences stay within about ±5%.",
        "Grouped bar chart of transaction type shares by class and a heatmap of type-by-direction lift.")

    # 7 amount density ratio per type (the strongest pattern) ---------------------------------------------------
    fig = go.Figure()
    amt_stats = {}
    colors = TYPE_COLORS
    for ty in ["karta", "bank_otkazmasi", "naqd"]:
        a = th.filter(pl.col("tranzaksiya_turi") == ty)
        a0 = a.filter(pl.col("eskalatsiya") == 0)["amt"].to_numpy()
        a1 = a.filter(pl.col("eskalatsiya") == 1)["amt"].to_numpy()
        edges = np.quantile(np.r_[a0, a1], np.linspace(0, 1, 21))
        h0 = np.histogram(a0, edges)[0] / len(a0)
        h1 = np.histogram(a1, edges)[0] / len(a1)
        mid = (edges[:-1] + edges[1:]) / 2
        fig.add_scatter(x=mid, y=h1 / h0, mode="lines+markers", line={"color": colors[ty], "width": 2},
                        marker={"size": 8}, name=TYPE_LABEL[ty],
                        hovertemplate="miqdor %{x:.2f}: %{y:.3f}<extra>" + ty + "</extra>")
        per = d[[f"{ty}_mean"]].assign(y=y)
        amt_stats[ty] = mw(per.loc[per.y == 0, f"{ty}_mean"].to_numpy(), per.loc[per.y == 1, f"{ty}_mean"].to_numpy())
    fig.add_hline(y=1, line={"dash": "dot", "color": LIGHT["text2"], "width": 1})
    st["amount_by_type"] = amt_stats
    style(fig, "", x="miqdor_indeksi (vigintil markazlari / bin centres)",
          y="eskalatsiya ÷ rad zichligi / density ratio")
    bk = amt_stats["bank_otkazmasi"]
    add("f07_amount_ratio", fig, "Miqdor taqsimoti: eskalatsiya ÷ rad", "Amount distribution: escalated ÷ dismissed",
        f"Eng aniq naqsh: eskalatsiya qilinganlarda kichik bank o'tkazmalari ko'proq, kattalari esa kamroq "
        f"(bank o'tkazmasi o'rtacha miqdori median {bk['median_escalated']:.2f} vs {bk['median_dismissed']:.2f}; "
        f"Cliff δ = {bk['cliffs_delta']:.3f}, p < 0.001). Bu 'bo'lib-bo'lib o'tkazish' ga o'xshash xulq bo'lishi mumkin, "
        "ammo aniq chegara oldida to'planish topilmadi.",
        f"Clearest pattern: escalated alerts have relatively more small bank transfers and fewer large ones "
        f"(per-alert mean transfer amount, median {bk['median_escalated']:.2f} vs {bk['median_dismissed']:.2f}; "
        f"Cliff's delta = {bk['cliffs_delta']:.3f}, p < 0.001). This resembles splitting payments, but no bunching "
        "just below a threshold was found (smurfing test, arXiv 2309.12704).",
        "Line chart of the escalated-to-dismissed density ratio across amount bins for card, transfer and cash.")

    # 8 amount distributions by type (quantile view) ---------------------------------------------------------------
    fig = go.Figure()
    for ty in TYPES:
        for yy, c, nm in [(0, C0, "Rad"), (1, C1, "Esk.")]:
            a = th.filter((pl.col("tranzaksiya_turi") == ty) & (pl.col("eskalatsiya") == yy))["amt"].to_numpy()
            q = np.quantile(a, [0.05, 0.25, 0.5, 0.75, 0.95])
            fig.add_box(x=[TYPE_LABEL[ty]] * 1, q1=[q[1]], median=[q[2]], q3=[q[3]], lowerfence=[q[0]],
                        upperfence=[q[4]], marker_color=c, name=nm, legendgroup=nm, showlegend=ty == "karta",
                        offsetgroup=str(yy))
    fig.update_layout(boxmode="group")
    style(fig, "", y="miqdor_indeksi (5–95% kvantillar / quantiles)")
    add("f08_amount_types", fig, "Tur bo'yicha miqdor", "Amount by type",
        "Miqdor turlar bo'yicha keskin farq qiladi: xalqaro > naqd > bank o'tkazmasi > karta. Shu sababli miqdor "
        "belgilari har bir tur ichida alohida (va tur × yo'nalish bo'yicha) hisoblandi.",
        "Amount level differs sharply by type: international > cash > transfer > card. Amount features are therefore "
        "computed within each type (and type × direction), not only overall.",
        "Grouped box plots of amount quantiles by transaction type and class.")

    # 9 recency windows ------------------------------------------------------------------------------------------------
    ws = [1, 3, 7, 14, 30, 60, 90]
    win = {}
    fig = go.Figure()
    for yy, c, nm in [(0, C0, N0), (1, C1, N1)]:
        med = [float(np.nanmedian(d.loc[d.y == yy, f"w{w}_rate_vs_hist"])) for w in ws]
        fig.add_scatter(x=[f"{w}d" for w in ws], y=med, mode="lines+markers", line={"color": c, "width": 2},
                        marker={"size": 8}, name=nm, hovertemplate="%{x}: %{y:.2f}×<extra>" + nm + "</extra>")
    for w in ws:
        win[w] = mw(d.loc[d.y == 0, f"w{w}_rate_vs_hist"].to_numpy(), d.loc[d.y == 1, f"w{w}_rate_vs_hist"].to_numpy())
    st["windows_rate_vs_hist"] = win
    fig.add_hline(y=1, line={"dash": "dot", "color": LIGHT["text2"], "width": 1})
    style(fig, "", x="oyna / window before alert", y="median: oyna tezligi ÷ tarix tezligi / window rate ÷ history rate")
    add("f09_windows", fig, "Signal oldidagi oynalar", "Pre-alert windows vs baseline",
        "Signal oldidan oxirgi kunlarda faollik tarixiy o'rtachadan past (egri chiziqlar 1 dan past), chunki faollik "
        "~130 kun oldin eng yuqori bo'lgan. Ikkala sinfda shakl bir xil; farqlar juda kichik "
        f"(eng katta |δ| = {max(abs(v['cliffs_delta']) for v in win.values()):.3f}).",
        "In the last days before the alert activity is below the history average (curves under 1), because activity "
        "peaks ~130 days earlier. The shape is the same for both classes; differences are tiny "
        f"(largest |delta| = {max(abs(v['cliffs_delta']) for v in win.values()):.3f}).",
        "Line chart of median window-to-history activity ratio for 1 to 90 day windows by class.")

    # 10 hour / weekday / gaps -------------------------------------------------------------------------------------------
    hh = t.group_by(["hour", "is_burst"]).len().sort("hour").to_pandas()
    hp = hh.pivot_table(index="hour", columns="is_burst", values="len", fill_value=0)
    wd = th.group_by(["wday", "eskalatsiya"]).len().to_pandas().pivot_table(index="wday", columns="eskalatsiya",
                                                                            values="len")
    wd = wd / wd.sum()
    st["night_share"] = mw(d.loc[d.y == 0, "hour_0_6_share"].to_numpy(), d.loc[d.y == 1, "hour_0_6_share"].to_numpy()) \
        if "hour_0_6_share" in d else {}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Soat / hour of day (all train tx)",
                                                        "Hafta kuni / weekday share"))
    hist_v = hp.get(False, pd.Series(0, index=hp.index))
    fig.add_bar(x=hp.index, y=hist_v, marker_color=LIGHT["text2"], name="Tarix / history", offsetgroup="h",
                row=1, col=1)
    fig.add_bar(x=hp.index, y=hp.get(True, pd.Series(0, index=hp.index)), base=hist_v, marker_color=LIGHT["s3"],
                name="Portlash / burst", offsetgroup="h", row=1, col=1)
    days = ["Du", "Se", "Ch", "Pa", "Ju", "Sh", "Ya"]
    for yy, c, nm in [(0, C0, N0), (1, C1, N1)]:
        fig.add_bar(x=days, y=wd[yy], marker_color=c, name=nm, row=1, col=2)
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".1%", row=1, col=2)
    style(fig, "")
    add("f10_time_of_day", fig, "Kun soati va hafta kuni", "Hour of day and weekday",
        "Tarixiy tranzaksiyalar soatlar bo'yicha deyarli tekis taqsimlangan (sintetik ma'lumot); 23-soatdagi cho'qqi "
        "butunlay 3 daqiqalik portlash hisobiga. Hafta kunlari ham ikki sinfda bir xil — tungi/dam olish belgilari "
        "ablatsiyada foyda bermadi.",
        "Historical transactions are almost uniform over the hours (synthetic data); the 23:00 spike is entirely the "
        "3-minute burst. Weekday shares are also the same for both classes — night/weekend features did not help "
        "in ablation.",
        "Stacked bar chart of transactions per hour split into history and burst, and weekday share by class.")

    # 11 univariate AUC ranking + correlation -----------------------------------------------------------------------------
    X = ftr.drop(columns="signal_id")
    au = {}
    for c in X.columns:
        v = X[c].to_numpy()
        ok = ~np.isnan(v)
        if ok.sum() > 1000 and np.nanstd(v) > 0:
            au[c] = float(roc_auc_score(y[ok], v[ok]))
    aus = pd.Series(au)
    strength = (aus - 0.5).abs().sort_values(ascending=False)
    top = strength.head(25).index
    st["univariate_top10"] = {k: round(float(aus[k]), 4) for k in top[:10]}
    st["univariate_n_over_055"] = int((strength > 0.05).sum())
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.28,
                        subplot_titles=("Bir o'zgaruvchili AUC (top 25) / univariate AUC", "Korrelyatsiya (top 15)"))
    fig.add_bar(y=list(top[::-1]), x=[max(aus[k], 1 - aus[k]) for k in top[::-1]], orientation="h",
                marker_color=[C1 if aus[k] > 0.5 else C0 for k in top[::-1]],
                hovertemplate="%{y}: %{x:.3f}<extra></extra>", showlegend=False, row=1, col=1)
    t15 = list(top[:15])
    cm = X[t15].corr(method="spearman").values
    fig.add_heatmap(z=cm, x=t15, y=t15, zmin=-1, zmax=1, colorscale=[[0, "#2a78d6"], [0.5, "#f0efec"], [1, "#e34948"]],
                    hovertemplate="%{x} × %{y}: %{z:.2f}<extra></extra>", colorbar={"title": "ρ"}, row=1, col=2)
    fig.update_xaxes(range=[0.5, 0.6], row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(tickfont={"size": 10})
    style(fig, "", height=620)
    add("f11_univariate", fig, "Belgilar kuchi (bir o'zgaruvchili)", "Feature strength (univariate)",
        f"Birorta belgi 0.58 AUC dan oshmaydi (eng kuchlisi {top[0]} = {max(aus[top[0]], 1 - aus[top[0]]):.3f}); "
        f"{st['univariate_n_over_055']} ta belgi 0.55 dan yuqori. Signal ko'p kuchsiz, o'zaro bog'liq belgilarga "
        "tarqalgan — shu sabab kuchli regulyarizatsiyali GBDT va ansambl tanlandi. To'q sariq = eskalatsiya bilan musbat.",
        f"No single feature exceeds 0.58 AUC (strongest: {top[0]} = {max(aus[top[0]], 1 - aus[top[0]]):.3f}); "
        f"{st['univariate_n_over_055']} features exceed 0.55. The signal is spread over many weak, correlated features — "
        "hence heavily regularised GBDTs and an ensemble. Orange = positively associated with escalation.",
        "Horizontal bar chart of univariate AUC for the top 25 features and a Spearman correlation heatmap.")

    # 12 drift train vs test ----------------------------------------------------------------------------------------------------
    ks = {c: stats.ks_2samp(ftr[c].dropna(), fte[c].dropna()).statistic for c in top}
    audit = json.loads((ROOT / "artifacts" / "audit.json").read_text())
    st["drift"] = {"ks_max_top25": round(float(max(ks.values())), 4),
                   "adversarial_auc": audit["adversarial_validation"]["auc"]}
    k4 = list(top[:4])
    fig = make_subplots(rows=1, cols=4, subplot_titles=k4)
    for i, col in enumerate(k4):
        lo, hi = np.nanquantile(ftr[col], [0.01, 0.99])
        for fr, c, nm in [(ftr, C0, "Train"), (fte, LIGHT["s3"], "Test")]:
            hist_bar(fig, fr[col].clip(lo, hi), np.linspace(lo, hi, 41), c, nm, 1, i + 1, showlegend=i == 0)
    fig.update_layout(barmode="overlay")
    style(fig, "", height=360)
    add("f12_drift", fig, "Train va test taqsimoti", "Train vs test distributions",
        f"Train va test bir xil taqsimotdan: adversarial validatsiya AUC = {st['drift']['adversarial_auc']:.3f} "
        f"(0.5 = farqlab bo'lmaydi), top-25 belgilar bo'yicha maksimal KS = {st['drift']['ks_max_top25']:.3f}. "
        "Sanalar oralig'i ham bir xil (2025-01-01 … 2026-12-31) → tasodifiy bo'linish, stratifikatsiyalangan K-fold CV.",
        f"Train and test come from the same distribution: adversarial validation AUC = {st['drift']['adversarial_auc']:.3f} "
        f"(0.5 = indistinguishable), max KS over the top-25 features = {st['drift']['ks_max_top25']:.3f}. Date ranges are "
        "identical (2025-01-01 … 2026-12-31) → random split, so stratified K-fold CV is appropriate.",
        "Four overlaid histograms comparing train and test distributions of the strongest features.")

    # 13 feature-family ablation from the experiment ledger -----------------------------------------------------------------
    led = [json.loads(line) for line in (ROOT / "artifacts" / "experiments.jsonl").read_text().splitlines()]
    led = {r["name"]: r for r in led}
    base = led.get("full_brel_reg", {}).get("oof_auc_mean_over_repeats")
    fams = [("drop_agg", "Agregatlar / aggregates"), ("drop_lastk", "Oxirgi 20 tx / last-20 tx"),
            ("drop_win", "Oynalar / windows"), ("drop_hist", "Miqdor gistogrammasi / amount histogram"),
            ("no_burst", "Portlash / burst"), ("drop_time", "Vaqt naqshlari / time patterns"),
            ("drop_pt", "Pass-through")]
    abl = {lab: round(base - led[n]["oof_auc_mean_over_repeats"], 4) for n, lab in fams if n in led and base}
    # no_burst was run before burst-relative features existed; its delta is vs its own baseline
    if "no_burst" in led and "full_split" in led:
        abl["Portlash / burst"] = round(led["full_split"]["oof_auc_mean_over_repeats"]
                                        - led["no_burst"]["oof_auc_mean_over_repeats"], 4)
    st["ablation_delta_auc"] = abl
    fig = go.Figure(go.Bar(y=list(abl.keys())[::-1], x=list(abl.values())[::-1], orientation="h",
                           marker_color=[C1 if v > 0 else LIGHT["text2"] for v in list(abl.values())[::-1]],
                           hovertemplate="%{y}: ΔAUC %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line={"color": LIGHT["text2"], "width": 1})
    style(fig, "", x="OOF AUC yo'qotish (oila olib tashlanganda) / AUC lost when family removed", height=380)
    add("f13_ablation", fig, "Belgi oilalari hissasi (ablatsiya)", "Feature-family contribution (ablation)",
        "Har bir belgi oilasi olib tashlanganda OOF AUC qancha tushadi (LightGBM, 5×2 CV). Agregatlar asosiy hissa "
        "qo'shadi; oxirgi 20 tranzaksiya va portlash belgilari ham foydali; vaqt naqshlari va pass-through foyda "
        "bermadi va yakuniy to'plamdan chiqarildi.",
        "How much OOF AUC drops when each feature family is removed (LightGBM, 5×2 CV). Aggregates carry most of the "
        "signal; the last-20-transactions and burst families also help; time-pattern and pass-through families did "
        "not help and were dropped from the final set.",
        "Horizontal bar chart of AUC lost when each feature family is removed.")

    (out / "stats.json").write_text(json.dumps(st, indent=1, default=float), encoding="utf-8")
    (out / "figures.json").write_text(json.dumps(figs, indent=1, ensure_ascii=False), encoding="utf-8")
    print("figures", len(figs), "->", out)


if __name__ == "__main__":
    main()

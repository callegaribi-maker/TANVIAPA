"""
APA do passo à frente — L5 (ML/AP) antes do heel-off detectado no vertical da perna.

Rodar:
    pip install streamlit plotly pandas numpy scipy
    streamlit run app_apa.py
"""
import io
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# =============================================================================
# Núcleo de processamento (sem dependência do Streamlit)
# =============================================================================

def read_acc(file_or_buf):
    """Lê arquivo 'tempo(ms), X, Y, Z' (cabeçalho qualquer)."""
    d = pd.read_csv(file_or_buf, skipinitialspace=True)
    d = d.iloc[:, :4]
    d.columns = ["t", "x", "y", "z"]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    d = d.sort_values("t").drop_duplicates("t")
    d["t"] = d["t"] / 1000.0  # ms -> s
    return d.reset_index(drop=True)


def lowpass(s, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, s)


def sampling_info(d):
    dt = np.diff(d.t.values) * 1000
    return dict(n=len(d), dur_s=d.t.iloc[-1] - d.t.iloc[0],
                fs_med=1000 / np.median(dt), dt_max_ms=dt.max())


def find_jump_and_lag(L, G, fs, search_until, max_lag=0.5):
    """Salto = maior |a| da perna até `search_until` s. Lag por correlação cruzada
    das normas (|a|) numa janela de ±0.9 s em torno do salto. Retorna (t_salto, lag)
    com lag > 0 significando que a perna está atrasada em relação a L5."""
    t = np.arange(0, min(L.t.iloc[-1], G.t.iloc[-1]), 1 / fs)
    mL = np.sqrt(sum(np.interp(t, L.t, L[a]) ** 2 for a in "xyz"))
    mG = np.sqrt(sum(np.interp(t, G.t, G[a]) ** 2 for a in "xyz"))
    sel = t < search_until
    tj = t[sel][np.argmax(mG[sel])]
    w = (t > tj - 0.9) & (t < tj + 0.8)
    a_, b_ = mL[w] - mL[w].mean(), mG[w] - mG[w].mean()
    c = np.correlate(b_, a_, "full")
    lags = np.arange(-len(a_) + 1, len(a_)) / fs
    k = np.abs(lags) <= max_lag
    lag = lags[k][np.argmax(c[k])]
    return tj, lag


def build_signals(L, G, p):
    """Reamostra, sincroniza e filtra. Retorna DataFrame com t, ml, ap, gy, e brutos."""
    fs = p["fs"]
    t_end = min(L.t.iloc[-1], G.t.iloc[-1] - p["lag"])
    t = np.arange(0, t_end, 1 / fs)
    sig = pd.DataFrame({"t": t})
    sig["ml"] = lowpass(np.interp(t, L.t, L[p["ml_axis"]]) * p["ml_sign"], p["fc_l5"], fs)
    sig["ap"] = lowpass(np.interp(t, L.t, L[p["ap_axis"]]) * p["ap_sign"], p["fc_l5"], fs)
    sig["l5v"] = lowpass(np.interp(t, L.t, L[p["l5_vert_axis"]]), p["fc_l5"], fs)
    sig["gy"] = lowpass(np.interp(t, G.t - p["lag"], G[p["leg_axis"]]), p["fc_leg"], fs)
    sig["gy_raw"] = np.interp(t, G.t - p["lag"], G[p["leg_axis"]])
    return sig


def detect_bursts(sig, p, t_start):
    """Blocos de atividade no vertical da perna (envoltória > limiar)."""
    fs = p["fs"]
    gy = sig.gy.values
    env = lowpass(np.abs(gy - np.median(gy)), 2.0, fs)
    act = env > p["burst_thr"]
    segs, s = [], None
    for i, v in enumerate(act):
        if v and s is None:
            s = i
        if not v and s is not None:
            segs.append([s, i]); s = None
    if s is not None:
        segs.append([s, len(act) - 1])
    merged = []
    for a, b in segs:
        if merged and (a - merged[-1][1]) < p["burst_merge"] * fs:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    t = sig.t.values
    return [(t[a], t[b]) for a, b in merged
            if t[a] > t_start and (t[b] - t[a]) >= p["burst_min_dur"]]


def classify_burst(sig, bs, p):
    """Frente/trás pelo sinal da maior excursão AP de L5 em [bs-0.8, bs+0.3]
    (relativa à mediana do AP no registro todo): + = frente, − = trás."""
    t, ap = sig.t.values, sig.ap.values
    w = (t > bs - 0.8) & (t < bs + 0.3)
    dev = ap[w] - np.median(ap)
    if not w.any() or np.abs(dev).max() < p["class_thr"]:
        return "indefinido"
    return "frente" if dev[np.argmax(np.abs(dev))] > 0 else "trás"


def quiet_baseline(sig, ho, fs, length=0.5, earliest=2.6, latest=0.6):
    """Janela mais quieta de `length` s terminando entre ho-earliest+length e ho-latest."""
    t = sig.t.values
    best = None
    for s0 in np.arange(ho - earliest, ho - latest - length + 1e-9, 0.05):
        w = (t >= s0) & (t < s0 + length)
        if w.sum() < 5:
            continue
        sc = sig.ml.values[w].std() + sig.ap.values[w].std() + sig.gy.values[w].std()
        if best is None or sc < best[0]:
            best = (sc, w)
    return best[1] if best else (t >= ho - 1.5) & (t < ho - 1.0)


def detect_heel_off(sig, bs, p):
    """Âncora: pico AP para frente perto do início do bloco. Heel-off = início da
    subida do vertical da perna até o 1º pico > baseline + ho_thr, buscando a partir
    de (pico AP - 0.3 s)."""
    t, ap, gy = sig.t.values, sig.ap.values, sig.gy.values
    w = np.flatnonzero((t > bs - 1.0) & (t < bs + 1.0))
    t_appk = t[w[np.argmax(ap[w])]]
    pre = (t > t_appk - 2.5) & (t < t_appk - 1.2)
    g0 = np.median(gy[pre]) if pre.any() else np.median(gy)
    seg = np.flatnonzero((t > t_appk - 0.3) & (t < t_appk + 1.0))
    above = gy[seg] - g0 > p["ho_thr"]
    if not above.any():
        return np.nan
    pk = seg[np.argmax(above)]
    while pk + 1 < len(gy) and gy[pk + 1] > gy[pk]:
        pk += 1
    j = pk
    while j > 0 and gy[j - 1] < gy[j]:
        j -= 1
    return t[j]


def apa_metrics(sig, ho, p):
    """Métricas do APA em ML e AP entre o fim da baseline e o heel-off."""
    fs = p["fs"]
    t = sig.t.values
    bw = quiet_baseline(sig, ho, fs, p["base_len"])
    bend = np.flatnonzero(bw)[-1]
    jho = int(np.argmin(np.abs(t - ho)))
    out = {"baseline_ini_s": round(t[bw][0], 2), "baseline_fim_s": round(t[bw][-1], 2)}
    nmin = max(1, int(round(p["min_dur"] * fs)))
    for name in ("ML", "AP"):
        s = sig[name.lower()].values
        b0, sd = s[bw].mean(), max(s[bw].std(), 0.01)
        thr = max(p["k_sd"] * sd, p["min_abs"])
        if name == "AP":
            d = 1
        else:  # sentido dominante: maior desvio em [ho-0.8, ho-0.2]
            ww = (t > ho - 0.8) & (t < ho - 0.2)
            dv = s[ww] - b0
            d = 1 if dv[np.argmax(np.abs(dv))] > 0 else -1
        win = np.arange(bend, jho + 1)
        ob = d * (s[win] - b0) > thr
        onset = None
        for n in range(len(win) - nmin + 1):
            if ob[n:n + nmin].all():
                onset = n; break
        out[f"{name}_sentido"] = "+" if d > 0 else "−"
        out[f"{name}_limiar_m_s2"] = round(thr, 3)
        if onset is None:
            for k in ("inicio_rel_HO_ms", "pico_m_s2", "pico_rel_HO_ms", "dv_m_s"):
                out[f"{name}_{k}"] = np.nan
            continue
        pw = win[onset:]
        pki = pw[np.argmax(d * (s[pw] - b0))]
        out[f"{name}_inicio_rel_HO_ms"] = round((t[win[onset]] - ho) * 1000)
        out[f"{name}_inicio_s"] = round(t[win[onset]], 3)
        out[f"{name}_pico_m_s2"] = round(s[pki] - b0, 3)
        out[f"{name}_pico_rel_HO_ms"] = round((t[pki] - ho) * 1000)
        out[f"{name}_dv_m_s"] = round(np.trapezoid(s[pw] - b0, dx=1 / fs), 3)
    return out, bw


def run_pipeline(L, G, p):
    tj, lag_auto = find_jump_and_lag(L, G, p["fs"], p["jump_until"])
    if p.get("lag") is None:
        p["lag"] = lag_auto
    sig = build_signals(L, G, p)
    bursts = detect_bursts(sig, p, tj + p["post_jump"])
    ev = []
    for i, (bs, be) in enumerate(bursts, 1):
        if p["class_mode"] == "alternada":
            tipo = "frente" if (i - 1) % 2 == 0 else "trás"
        else:
            tipo = classify_burst(sig, bs, p)
        ho = detect_heel_off(sig, bs, p) if tipo == "frente" else np.nan
        ev.append(dict(evento=i, inicio_bloco_s=round(bs, 2), fim_bloco_s=round(be, 2),
                       tipo=tipo, HO_auto_s=round(ho, 3) if not np.isnan(ho) else np.nan))
    return dict(t_jump=tj, lag_auto=lag_auto, sig=sig, events=pd.DataFrame(ev))


# =============================================================================
# Interface Streamlit
# =============================================================================

def main():
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    st.set_page_config(page_title="APA – passo à frente", layout="wide")

    def stretch(fn, *a, **k):
        """Compatível com versões novas (width='stretch') e antigas (use_container_width)."""
        try:
            return fn(*a, width="stretch", **k)
        except TypeError:
            return fn(*a, use_container_width=True, **k)
    st.title("APA do passo à frente — L5 × perna")

    # ---------------- Sidebar: arquivos e parâmetros ----------------
    sb = st.sidebar
    sb.header("Arquivos")
    fL5 = sb.file_uploader("Acelerômetro L5", type=["txt", "csv"])
    fLeg = sb.file_uploader("Acelerômetro perna", type=["txt", "csv"])

    sb.header("Eixos")
    c1, c2 = sb.columns(2)
    ml_axis = c1.selectbox("L5 ML", ["x", "y", "z"], 0)
    ml_sign = c2.selectbox("sinal ML", [1, -1], 0)
    ap_axis = c1.selectbox("L5 AP", ["x", "y", "z"], 2)
    ap_sign = c2.selectbox("sinal AP (+ = frente)", [1, -1], 0)
    l5_vert_axis = c1.selectbox("L5 vertical", ["x", "y", "z"], 1)
    leg_axis = c2.selectbox("Perna vertical", ["x", "y", "z"], 1)

    sb.header("Filtros e sincronização")
    fs = sb.number_input("Reamostragem (Hz)", 50, 500, 100, 10)
    fc_l5 = sb.slider("Passa-baixa L5 (Hz)", 1.0, 20.0, 5.0, 0.5)
    fc_leg = sb.slider("Passa-baixa perna (Hz)", 2.0, 30.0, 10.0, 0.5)
    jump_until = sb.number_input("Procurar salto até (s)", 2.0, 60.0, 20.0, 1.0)
    lag_manual = sb.checkbox("Definir lag manualmente")
    lag_val = sb.number_input("Lag perna vs L5 (s)", -2.0, 2.0, 0.19, 0.01,
                              disabled=not lag_manual)

    sb.header("Detecção de passos")
    burst_thr = sb.slider("Limiar de atividade da perna (m/s²)", 0.05, 1.5, 0.25, 0.05)
    burst_merge = sb.slider("Unir blocos separados por < (s)", 0.1, 2.0, 0.8, 0.1)
    burst_min_dur = sb.slider("Duração mínima do bloco (s)", 0.1, 1.5, 0.3, 0.05)
    post_jump = sb.slider("Ignorar até X s após o salto", 0.0, 3.0, 0.5, 0.1)
    class_mode = sb.radio("Classificação frente/trás",
                          ["alternada", "automática (AP)"], 0,
                          help="Alternada: 1º passo após o salto = frente, depois alterna. "
                               "Automática: sinal da maior excursão AP de L5 antes do passo.")
    class_thr = sb.slider("Limiar AP p/ classificar frente/trás (m/s²)", 0.3, 2.0, 0.8, 0.1)
    ho_thr = sb.slider("Heel-off: pico mínimo na perna (m/s² acima da base)", 0.3, 3.0, 0.8, 0.1)

    sb.header("Início do APA")
    base_len = sb.slider("Janela de baseline (s)", 0.2, 1.0, 0.5, 0.05)
    k_sd = sb.slider("Limiar = k × DP da baseline", 1.0, 6.0, 3.0, 0.5)
    min_abs = sb.slider("Limiar mínimo absoluto (m/s²)", 0.0, 0.5, 0.1, 0.02)
    min_dur = sb.slider("Permanência mínima fora da banda (s)", 0.02, 0.3, 0.1, 0.01)

    if not (fL5 and fLeg):
        st.info("Carregue os dois arquivos na barra lateral (L5 e perna).")
        st.stop()

    L, G = read_acc(fL5), read_acc(fLeg)
    p = dict(fs=fs, fc_l5=fc_l5, fc_leg=fc_leg, ml_axis=ml_axis, ml_sign=ml_sign,
             ap_axis=ap_axis, ap_sign=ap_sign, l5_vert_axis=l5_vert_axis, leg_axis=leg_axis,
             jump_until=jump_until, lag=lag_val if lag_manual else None,
             burst_thr=burst_thr, burst_merge=burst_merge, burst_min_dur=burst_min_dur,
             post_jump=post_jump, class_thr=class_thr,
             class_mode="alternada" if class_mode == "alternada" else "auto", ho_thr=ho_thr,
             base_len=base_len, k_sd=k_sd, min_abs=min_abs, min_dur=min_dur)
    R = run_pipeline(L, G, p)
    sig, t = R["sig"], R["sig"].t.values

    # ---------------- 1. Dados e sincronização ----------------
    st.subheader("1. Dados e sincronização")
    iL, iG = sampling_info(L), sampling_info(G)
    c = st.columns(4)
    c[0].metric("L5: amostras / fs mediana", f"{iL['n']} / {iL['fs_med']:.0f} Hz")
    c[1].metric("Perna: amostras / fs mediana", f"{iG['n']} / {iG['fs_med']:.0f} Hz")
    c[2].metric("Salto (perna)", f"{R['t_jump']:.2f} s")
    c[3].metric("Lag usado (perna atrasada)", f"{p['lag']:.3f} s",
                f"auto = {R['lag_auto']:.3f} s", delta_color="off")

    with st.expander("Ver alinhamento no salto", expanded=False):
        w = (t > R["t_jump"] - p["lag"] - 1.0) & (t < R["t_jump"] - p["lag"] + 1.0)
        mL = np.sqrt(sum(np.interp(t, L.t, L[a]) ** 2 for a in "xyz"))
        mG = np.sqrt(sum(np.interp(t, G.t - p["lag"], G[a]) ** 2 for a in "xyz"))
        fig = go.Figure()
        fig.add_scatter(x=t[w], y=mL[w], name="|a| L5")
        fig.add_scatter(x=t[w], y=mG[w], name="|a| perna (sincronizada)")
        fig.update_layout(height=320, xaxis_title="tempo (s)", yaxis_title="m/s²",
                          margin=dict(t=20, b=40))
        stretch(st.plotly_chart, fig)

    # ---------------- 2. Eventos ----------------
    st.subheader("2. Localização dos eventos")
    st.caption("Edite a coluna **tipo** (frente / trás / ignorar) e, se quiser, "
               "informe um **HO_manual_s** para substituir o heel-off automático.")
    ev = R["events"].copy()
    if ev.empty:
        st.warning("Nenhum passo detectado. Ajuste o limiar de atividade da perna.")
        st.stop()
    ev["HO_manual_s"] = np.nan
    ev = stretch(
        st.data_editor, ev, hide_index=True,
        column_config={
            "tipo": st.column_config.SelectboxColumn(
                options=["frente", "trás", "ignorar", "indefinido"]),
            "HO_manual_s": st.column_config.NumberColumn(format="%.3f"),
        },
        disabled=["evento", "inicio_bloco_s", "fim_bloco_s", "HO_auto_s"], key="ev_editor")

    # recalcula HO para eventos marcados como frente sem HO
    for i, r in ev.iterrows():
        if r.tipo == "frente" and np.isnan(r.HO_auto_s):
            ev.at[i, "HO_auto_s"] = round(detect_heel_off(sig, r.inicio_bloco_s, p), 3)
    ev["HO_s"] = ev.HO_manual_s.fillna(ev.HO_auto_s)
    fwd = ev[(ev.tipo == "frente") & ev.HO_s.notna()].reset_index(drop=True)

    rows, bws = [], {}
    for k, r in fwd.iterrows():
        m, bw = apa_metrics(sig, r.HO_s, p)
        rows.append(dict(tentativa=k + 1, evento=r.evento, HO_s=r.HO_s,
                         HO_origem="manual" if not np.isnan(r.HO_manual_s) else "auto", **m))
        bws[k + 1] = bw
    res = pd.DataFrame(rows)

    # visão geral
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Perna vertical", "L5 ML", "L5 AP (+ frente)"))
    fig.add_scatter(x=t, y=sig.gy, line=dict(color="black", width=1), name="perna vert.", row=1, col=1)
    fig.add_scatter(x=t, y=sig.ml, line=dict(color="#1f77b4", width=1), name="ML", row=2, col=1)
    fig.add_scatter(x=t, y=sig.ap, line=dict(color="#ff7f0e", width=1), name="AP", row=3, col=1)
    colors = {"frente": "rgba(46,160,67,0.15)", "trás": "rgba(214,39,40,0.12)",
              "indefinido": "rgba(128,128,128,0.12)"}
    for _, r in ev.iterrows():
        if r.tipo in colors:
            fig.add_vrect(x0=r.inicio_bloco_s, x1=r.fim_bloco_s, fillcolor=colors[r.tipo],
                          line_width=0, row="all", col=1)
    fig.add_vline(x=R["t_jump"] - p["lag"], line=dict(color="purple", dash="dot"))
    for _, r in res.iterrows():
        fig.add_vline(x=r.HO_s, line=dict(color="red", dash="dash", width=1))
        if not np.isnan(r.get("ML_inicio_s", np.nan)):
            fig.add_vline(x=r.ML_inicio_s, line=dict(color="#1f77b4", dash="dot", width=1), row=2, col=1)
        if not np.isnan(r.get("AP_inicio_s", np.nan)):
            fig.add_vline(x=r.AP_inicio_s, line=dict(color="#ff7f0e", dash="dot", width=1), row=3, col=1)
    fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="tempo (s)", row=3, col=1)
    stretch(st.plotly_chart, fig)
    st.caption("Verde = passo à frente · vermelho = passo para trás · roxo = salto · "
               "tracejado vermelho = heel-off · pontilhado = início do APA (ML azul, AP laranja).")

    if res.empty:
        st.warning("Nenhum passo à frente com heel-off definido.")
        st.stop()

    # ---------------- 3. Tentativa individual ----------------
    st.subheader("3. Tentativa individual")
    k = st.selectbox("Passo à frente nº", res.tentativa.tolist())
    r = res[res.tentativa == k].iloc[0]
    pre, pos = st.slider("Janela relativa ao HO (s)", -3.0, 2.0, (-1.8, 1.0), 0.1)
    w = (t > r.HO_s + pre) & (t < r.HO_s + pos)
    bw = bws[k]
    tt = t[w] - r.HO_s
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Perna vertical (rel. baseline)", "L5 ML (rel. baseline)",
                                        "L5 AP (rel. baseline, + frente)"))
    for i, (col, cor) in enumerate((("gy", "black"), ("ml", "#1f77b4"), ("ap", "#ff7f0e")), 1):
        s = sig[col].values
        b0 = s[bw].mean()
        fig.add_scatter(x=tt, y=s[w] - b0, line=dict(color=cor), row=i, col=1)
        if col != "gy":
            nm = col.upper()
            thr = r[f"{nm}_limiar_m_s2"]
            fig.add_hrect(y0=-thr, y1=thr, fillcolor="rgba(128,128,128,0.15)", line_width=0, row=i, col=1)
            if not np.isnan(r.get(f"{nm}_inicio_rel_HO_ms", np.nan)):
                fig.add_vline(x=r[f"{nm}_inicio_rel_HO_ms"] / 1000, line=dict(color=cor, dash="dot"), row=i, col=1)
                fig.add_scatter(x=[r[f"{nm}_pico_rel_HO_ms"] / 1000], y=[r[f"{nm}_pico_m_s2"]],
                                mode="markers", marker=dict(size=10, color=cor, symbol="x"), row=i, col=1)
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    b_ini, b_fim = t[bw][0] - r.HO_s, t[bw][-1] - r.HO_s
    fig.add_vrect(x0=b_ini, x1=b_fim, fillcolor="rgba(0,128,255,0.08)", line_width=0, row="all", col=1)
    fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="tempo relativo ao heel-off (s)", row=3, col=1)
    stretch(st.plotly_chart, fig)
    st.caption("Faixa azul = baseline · faixa cinza = banda de limiar · pontilhado = início do APA · "
               "× = pico · tracejado vermelho = heel-off.")

    # ---------------- 4. Todas as tentativas alinhadas ----------------
    st.subheader("4. Tentativas alinhadas no heel-off")
    grp_by = st.radio("Agrupar por", ["sentido ML (lado do apoio)", "todas juntas"], horizontal=True)
    tt = np.arange(-1.8, 1.0, 1 / fs)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("L5 ML", "L5 AP (+ frente)"))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    groups = res.groupby("ML_sentido") if grp_by.startswith("sentido") else [("todas", res)]
    for gi, (gname, gdf) in enumerate(groups):
        cor = palette[gi % len(palette)]
        for ci, col in enumerate(("ml", "ap"), 1):
            M = []
            for _, rr in gdf.iterrows():
                s = sig[col].values
                b0 = s[bws[rr.tentativa]].mean()
                y = np.interp(tt + rr.HO_s, t, s) - b0
                M.append(y)
                fig.add_scatter(x=tt, y=y, line=dict(color=cor, width=0.6), opacity=0.35,
                                showlegend=False, row=1, col=ci)
            M = np.array(M)
            fig.add_scatter(x=tt, y=M.mean(0), line=dict(color=cor, width=3),
                            name=f"média ML {gname} (n={len(M)})" if gname != "todas" else f"média (n={len(M)})",
                            showlegend=(ci == 1), row=1, col=ci)
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    fig.update_xaxes(title_text="tempo relativo ao heel-off (s)")
    fig.update_yaxes(title_text="m/s² (rel. baseline)", col=1)
    fig.update_layout(height=420, margin=dict(t=40, b=40))
    stretch(st.plotly_chart, fig)

    # ---------------- 5. Resultados ----------------
    st.subheader("5. Resultados")
    show = ["tentativa", "HO_s", "HO_origem",
            "ML_sentido", "ML_inicio_rel_HO_ms", "ML_pico_m_s2", "ML_pico_rel_HO_ms", "ML_dv_m_s",
            "AP_inicio_rel_HO_ms", "AP_pico_m_s2", "AP_pico_rel_HO_ms", "AP_dv_m_s"]
    show = [c for c in show if c in res.columns]
    stretch(st.dataframe, res[show], hide_index=True)
    num = [c for c in show if c not in ("tentativa", "HO_s", "HO_origem", "ML_sentido")]
    summ = res.groupby("ML_sentido")[num].agg(["mean", "std"]).round(2)
    st.markdown("**Média ± DP por sentido do ML**")
    stretch(st.dataframe, summ)

    buf = io.StringIO()
    res.to_csv(buf, index=False)
    st.download_button("Baixar resultados (CSV)", buf.getvalue(), "apa_resultados.csv", "text/csv")

    with st.expander("Notas de método"):
        st.markdown(
            "- **Sincronização:** correlação cruzada da norma |a| dos dois sensores em torno do "
            "maior impacto da perna (salto).\n"
            "- **Passos:** blocos de atividade no vertical da perna; frente/trás alternados a partir "
            "do salto (ou, no modo automático, pelo sinal da maior excursão AP de L5).\n"
            "- **Heel-off:** a partir do pico AP para frente − 0,3 s, primeiro pico do vertical da "
            "perna acima de base + limiar; HO = início dessa subida.\n"
            "- **Baseline:** janela mais quieta antes do HO (entre HO−2,6 s e HO−0,6 s).\n"
            "- **Início do APA:** 1º instante após a baseline em que o sinal ultrapassa base ± "
            "max(k·DP, mínimo) no sentido dominante e permanece fora pelo tempo mínimo.\n"
            "- **dv:** integral da aceleração do início do APA ao HO (variação de velocidade).\n"
            "- **Atenção:** em L5, a componente AP inclui a projeção da gravidade quando o tronco "
            "inclina (≈0,17 m/s² por grau), além da aceleração linear.")


main()

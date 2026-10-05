"""
Forward-step APA — L5 (ML/AP) before the heel-off detected on the leg vertical axis.

Run:
    pip install -r requirements.txt
    streamlit run app_apa.py
"""
import io
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

FWD, BWD, IGN, UND = "forward", "backward", "ignore", "undefined"

# =============================================================================
# Processing core (no Streamlit dependency)
# =============================================================================

def read_acc(file_or_buf):
    """Reads a 'time(ms), X, Y, Z' file (any header)."""
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
    """Jump = largest leg |a| before `search_until` s. Lag from cross-correlation of the
    acceleration norms in a ±0.9 s window around the jump (lag > 0: leg lags L5)."""
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
    return tj, lags[k][np.argmax(c[k])]


def build_signals(L, G, p):
    """Resample, synchronise and filter."""
    fs = p["fs"]
    t = np.arange(0, min(L.t.iloc[-1], G.t.iloc[-1] - p["lag"]), 1 / fs)
    sig = pd.DataFrame({"t": t})
    sig["ml"] = lowpass(np.interp(t, L.t, L[p["ml_axis"]]) * p["ml_sign"], p["fc_l5"], fs)
    sig["ap"] = lowpass(np.interp(t, L.t, L[p["ap_axis"]]) * p["ap_sign"], p["fc_l5"], fs)
    sig["gy"] = lowpass(np.interp(t, G.t - p["lag"], G[p["leg_axis"]]), p["fc_leg"], fs)
    return sig


def detect_bursts(sig, p, t_start):
    """Activity bursts on the leg vertical axis (envelope above threshold)."""
    fs, gy, t = p["fs"], sig.gy.values, sig.t.values
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
    return [(t[a], t[b]) for a, b in merged
            if t[a] > t_start and (t[b] - t[a]) >= p["burst_min_dur"]]


def classify_burst(sig, bs, p):
    """Forward/backward from the sign of the largest L5 AP excursion in [bs-0.8, bs+0.3]."""
    t, ap = sig.t.values, sig.ap.values
    w = (t > bs - 0.8) & (t < bs + 0.3)
    dev = ap[w] - np.median(ap)
    if not w.any() or np.abs(dev).max() < p["class_thr"]:
        return UND
    return FWD if dev[np.argmax(np.abs(dev))] > 0 else BWD


def quiet_baseline(sig, ho, fs, length=0.5, earliest=2.6, latest=0.6):
    """Quietest `length`-s window ending between ho-earliest+length and ho-latest."""
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
    """Anchor: forward AP peak near the burst. Heel-off = start of the rise of the leg
    vertical signal up to its first peak > baseline + ho_thr, searching from AP peak - 0.3 s."""
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
    """APA metrics for ML and AP between the end of the baseline and heel-off."""
    fs, t = p["fs"], sig.t.values
    bw = quiet_baseline(sig, ho, fs, p["base_len"])
    bend = np.flatnonzero(bw)[-1]
    jho = int(np.argmin(np.abs(t - ho)))
    out = {"baseline_start_s": round(t[bw][0], 2), "baseline_end_s": round(t[bw][-1], 2)}
    nmin = max(1, int(round(p["min_dur"] * fs)))
    for name in ("ML", "AP"):
        s = sig[name.lower()].values
        b0, sd = s[bw].mean(), max(s[bw].std(), 0.01)
        thr = max(p["k_sd"] * sd, p["min_abs"])
        if name == "AP":
            d = 1
        else:  # dominant direction: largest deviation in [ho-0.8, ho-0.2]
            ww = (t > ho - 0.8) & (t < ho - 0.2)
            dv = s[ww] - b0
            d = 1 if dv[np.argmax(np.abs(dv))] > 0 else -1
        win = np.arange(bend, jho + 1)
        dev = d * (s[win] - b0)
        method = p.get("onset_method", "backward")
        onset = None
        if method == "first":
            # first crossing that stays outside the band for >= min_dur
            ob = dev > thr
            for n in range(len(win) - nmin + 1):
                if ob[n:n + nmin].all():
                    onset = n; break
        else:
            # walk back from the peak to the last sample inside the band
            ipk = int(np.argmax(dev))
            lim = thr if method == "backward" else p.get("peak_frac", 0.15) * dev[ipk]
            if dev[ipk] > thr:
                n = ipk
                while n > 0 and dev[n - 1] > lim:
                    n -= 1
                onset = n
        out[f"{name}_direction"] = "+" if d > 0 else "−"
        out[f"{name}_threshold_m_s2"] = round(thr, 3)
        if onset is None:
            for k in ("onset_rel_HO_ms", "onset_s", "peak_m_s2", "peak_rel_HO_ms", "dv_m_s",
                      "onset_threshold_m_s2"):
                out[f"{name}_{k}"] = np.nan
            continue
        pw = win[onset:]
        pki = pw[np.argmax(d * (s[pw] - b0))]
        out[f"{name}_onset_threshold_m_s2"] = round(
            thr if method != "peak_frac" else p.get("peak_frac", 0.15) * d * (s[pki] - b0), 3)
        out[f"{name}_onset_rel_HO_ms"] = round((t[win[onset]] - ho) * 1000)
        out[f"{name}_onset_s"] = round(t[win[onset]], 3)
        out[f"{name}_peak_m_s2"] = round(s[pki] - b0, 3)
        out[f"{name}_peak_rel_HO_ms"] = round((t[pki] - ho) * 1000)
        out[f"{name}_dv_m_s"] = round(np.trapezoid(s[pw] - b0, dx=1 / fs), 3)
    # horizontal resultant sqrt(ML² + AP²) between baseline end and heel-off
    mlr = sig.ml.values - sig.ml.values[bw].mean()
    apr = sig.ap.values - sig.ap.values[bw].mean()
    res_h = np.hypot(mlr, apr)
    win = np.arange(bend, jho + 1)
    ipk = win[np.argmax(res_h[win])]
    out["RES_peak_m_s2"] = round(res_h[ipk], 3)
    out["RES_peak_rel_HO_ms"] = round((t[ipk] - ho) * 1000)
    out["RES_angle_deg"] = round(np.degrees(np.arctan2(mlr[ipk], apr[ipk])), 1)
    return out, bw


def run_pipeline(L, G, p):
    tj, lag_auto = find_jump_and_lag(L, G, p["fs"], p["jump_until"])
    if p.get("lag") is None:
        p["lag"] = lag_auto
    sig = build_signals(L, G, p)
    bursts = detect_bursts(sig, p, tj + p["post_jump"])
    ev = []
    for i, (bs, be) in enumerate(bursts, 1):
        if p["class_mode"] == "alternating":
            first = p.get("first_step", BWD)
            other = FWD if first == BWD else BWD
            kind = first if (i - 1) % 2 == 0 else other
        else:
            kind = classify_burst(sig, bs, p)
        ho = detect_heel_off(sig, bs, p) if kind == FWD else np.nan
        ev.append(dict(event=i, burst_start_s=round(bs, 2), burst_end_s=round(be, 2),
                       type=kind, HO_auto_s=round(ho, 3) if not np.isnan(ho) else np.nan))
    return dict(t_jump=tj, lag_auto=lag_auto, sig=sig, events=pd.DataFrame(ev))


# =============================================================================
# Streamlit interface
# =============================================================================

def main():
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc

    st.set_page_config(page_title="Forward-step APA", layout="wide")

    def stretch(fn, *a, **k):
        """Works with new (width='stretch') and older (use_container_width) Streamlit."""
        try:
            return fn(*a, width="stretch", **k)
        except TypeError:
            return fn(*a, use_container_width=True, **k)

    st.title("Forward-step APA — L5 × leg")

    # ---------------- Sidebar ----------------
    sb = st.sidebar
    sb.header("Files")
    fL5 = sb.file_uploader("L5 accelerometer", type=["txt", "csv"])
    fLeg = sb.file_uploader("Leg accelerometer", type=["txt", "csv"])

    sb.header("Axes")
    c1, c2 = sb.columns(2)
    ml_axis = c1.selectbox("L5 ML axis", ["x", "y", "z"], 0)
    ml_sign = c2.selectbox("ML sign", [1, -1], 0)
    ap_axis = c1.selectbox("L5 AP axis", ["x", "y", "z"], 2)
    ap_sign = c2.selectbox("AP sign (+ = forward)", [1, -1], 1,
                           help="Phone on L5 with the screen facing out: the z axis points "
                                "backwards, so use −1 to make + forward.")
    leg_axis = c1.selectbox("Leg vertical axis", ["x", "y", "z"], 1)

    sb.header("Filters and synchronisation")
    fs = sb.number_input("Resampling rate (Hz)", 50, 500, 100, 10)
    fc_l5 = sb.slider("L5 low-pass cut-off (Hz)", 1.0, 20.0, 5.0, 0.5)
    fc_leg = sb.slider("Leg low-pass cut-off (Hz)", 2.0, 30.0, 10.0, 0.5)
    jump_until = sb.number_input("Search for the jump up to (s)", 2.0, 60.0, 20.0, 1.0)
    lag_manual = sb.checkbox("Set lag manually")
    lag_val = sb.number_input("Leg lag relative to L5 (s)", -2.0, 2.0, 0.19, 0.01,
                              disabled=not lag_manual)

    sb.header("Step detection")
    burst_thr = sb.slider("Leg activity threshold (m/s²)", 0.05, 1.5, 0.25, 0.05)
    burst_merge = sb.slider("Merge bursts closer than (s)", 0.1, 2.0, 0.8, 0.1)
    burst_min_dur = sb.slider("Minimum burst duration (s)", 0.1, 1.5, 0.3, 0.05)
    post_jump = sb.slider("Ignore the first X s after the jump", 0.0, 3.0, 0.5, 0.1)
    class_mode = sb.radio("Forward/backward labelling", ["Alternating", "Automatic (AP)"], 0,
                          help="Alternating: steps alternate starting from the type chosen below. "
                               "Automatic: sign of the largest L5 AP excursion around the step.")
    first_step = sb.selectbox("First step after the jump (alternating mode)", [BWD, FWD], 0)
    class_thr = sb.slider("AP threshold for automatic labelling (m/s²)", 0.3, 2.0, 0.8, 0.1)
    ho_thr = sb.slider("Heel-off: minimum leg peak above baseline (m/s²)", 0.3, 3.0, 0.8, 0.1)

    sb.header("APA onset")
    onset_label = sb.radio(
        "Onset criterion",
        ["Backward from peak (baseline threshold)", "Backward from peak (% of peak)",
         "First sustained crossing"], 0,
        help="Backward from peak: start at the APA peak and go back in time to the last sample "
             "inside the threshold band; earlier oscillations that returned to baseline are ignored. "
             "% of peak: same, but the threshold is a fraction of the peak amplitude. "
             "First sustained crossing: first sample after the baseline that leaves the band and "
             "stays outside for the minimum time.")
    onset_method = {"Backward from peak (baseline threshold)": "backward",
                    "Backward from peak (% of peak)": "peak_frac",
                    "First sustained crossing": "first"}[onset_label]
    peak_frac = sb.slider("% of peak (for the % criterion)", 5, 50, 15, 5,
                          disabled=onset_method != "peak_frac") / 100
    base_len = sb.slider("Baseline window (s)", 0.2, 1.0, 0.5, 0.05)
    k_sd = sb.slider("Threshold = k × baseline SD", 1.0, 10.0, 5.0, 0.5)
    min_abs = sb.slider("Minimum absolute threshold (m/s²)", 0.0, 0.5, 0.1, 0.02)
    min_dur = sb.slider("Minimum time outside the band (s, first-crossing criterion)",
                        0.02, 0.3, 0.1, 0.01, disabled=onset_method != "first")

    if not (fL5 and fLeg):
        st.info("Upload both files (L5 and leg) in the sidebar to start.")
        st.stop()

    L, G = read_acc(fL5), read_acc(fLeg)
    p = dict(fs=fs, fc_l5=fc_l5, fc_leg=fc_leg, ml_axis=ml_axis, ml_sign=ml_sign,
             ap_axis=ap_axis, ap_sign=ap_sign, leg_axis=leg_axis,
             jump_until=jump_until, lag=lag_val if lag_manual else None,
             burst_thr=burst_thr, burst_merge=burst_merge, burst_min_dur=burst_min_dur,
             post_jump=post_jump, class_thr=class_thr, ho_thr=ho_thr,
             class_mode="alternating" if class_mode == "Alternating" else "auto",
             first_step=first_step,
             base_len=base_len, k_sd=k_sd, min_abs=min_abs, min_dur=min_dur,
             onset_method=onset_method, peak_frac=peak_frac)
    R = run_pipeline(L, G, p)
    sig, t = R["sig"], R["sig"].t.values

    # ---------------- 1. Data and synchronisation ----------------
    st.subheader("1. Data and synchronisation")
    iL, iG = sampling_info(L), sampling_info(G)
    c = st.columns(4)
    c[0].metric("L5: samples / median rate", f"{iL['n']} / {iL['fs_med']:.0f} Hz")
    c[1].metric("Leg: samples / median rate", f"{iG['n']} / {iG['fs_med']:.0f} Hz")
    c[2].metric("Jump (leg)", f"{R['t_jump']:.2f} s")
    c[3].metric("Lag used (leg behind L5)", f"{p['lag']:.3f} s",
                f"auto = {R['lag_auto']:.3f} s", delta_color="off")

    with st.expander("Check the alignment at the jump", expanded=False):
        w = (t > R["t_jump"] - p["lag"] - 1.0) & (t < R["t_jump"] - p["lag"] + 1.0)
        mL = np.sqrt(sum(np.interp(t, L.t, L[a]) ** 2 for a in "xyz"))
        mG = np.sqrt(sum(np.interp(t, G.t - p["lag"], G[a]) ** 2 for a in "xyz"))
        fig = go.Figure()
        fig.add_scatter(x=t[w], y=mL[w], name="|a| L5")
        fig.add_scatter(x=t[w], y=mG[w], name="|a| leg (synchronised)")
        fig.update_layout(height=320, xaxis_title="Time (s)", yaxis_title="m/s²",
                          margin=dict(t=20, b=40))
        stretch(st.plotly_chart, fig)

    # ---------------- 2. Events ----------------
    st.subheader("2. Event detection")
    st.caption("Edit the **type** column (forward / backward / ignore) and, if needed, enter "
               "**HO_manual_s** to override the automatic heel-off.")
    ev = R["events"].copy()
    if ev.empty:
        st.warning("No steps detected. Lower the leg activity threshold in the sidebar.")
        st.stop()
    ev["HO_manual_s"] = np.nan
    ev = stretch(
        st.data_editor, ev, hide_index=True,
        column_config={
            "type": st.column_config.SelectboxColumn(options=[FWD, BWD, IGN, UND]),
            "HO_manual_s": st.column_config.NumberColumn(format="%.3f"),
        },
        disabled=["event", "burst_start_s", "burst_end_s", "HO_auto_s"], key="ev_editor")

    for i, r in ev.iterrows():
        if r.type == FWD and np.isnan(r.HO_auto_s):
            ev.at[i, "HO_auto_s"] = round(detect_heel_off(sig, r.burst_start_s, p), 3)
    ev["HO_s"] = ev.HO_manual_s.fillna(ev.HO_auto_s)
    fwd = ev[(ev.type == FWD) & ev.HO_s.notna()].reset_index(drop=True)

    rows, bws = [], {}
    for k, r in fwd.iterrows():
        m, bw = apa_metrics(sig, r.HO_s, p)
        rows.append(dict(step=k + 1, event=r.event, HO_s=r.HO_s,
                         HO_source="manual" if not np.isnan(r.HO_manual_s) else "auto", **m))
        bws[k + 1] = bw
    res = pd.DataFrame(rows)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Leg vertical", "L5 ML", "L5 AP (+ forward)"))
    fig.add_scatter(x=t, y=sig.gy, line=dict(color="black", width=1), name="Leg vertical", row=1, col=1)
    fig.add_scatter(x=t, y=sig.ml, line=dict(color="#1f77b4", width=1), name="ML", row=2, col=1)
    fig.add_scatter(x=t, y=sig.ap, line=dict(color="#ff7f0e", width=1), name="AP", row=3, col=1)
    shade = {FWD: "rgba(46,160,67,0.15)", BWD: "rgba(214,39,40,0.12)", UND: "rgba(128,128,128,0.12)"}
    for _, r in ev.iterrows():
        if r.type in shade:
            fig.add_vrect(x0=r.burst_start_s, x1=r.burst_end_s, fillcolor=shade[r.type],
                          line_width=0, row="all", col=1)
    fig.add_vline(x=R["t_jump"] - p["lag"], line=dict(color="purple", dash="dot"))
    for _, r in res.iterrows():
        fig.add_vline(x=r.HO_s, line=dict(color="red", dash="dash", width=1))
        if not np.isnan(r.get("ML_onset_s", np.nan)):
            fig.add_vline(x=r.ML_onset_s, line=dict(color="#1f77b4", dash="dot", width=1), row=2, col=1)
        if not np.isnan(r.get("AP_onset_s", np.nan)):
            fig.add_vline(x=r.AP_onset_s, line=dict(color="#ff7f0e", dash="dot", width=1), row=3, col=1)
    fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    stretch(st.plotly_chart, fig)
    st.caption("Green = forward step · red = backward step · purple = jump · red dashed = heel-off · "
               "dotted = APA onset (ML blue, AP orange).")

    if res.empty:
        st.warning("No forward step with a heel-off. Check the event table.")
        st.stop()

    # ---------------- 3. Single step ----------------
    st.subheader("3. Single step")
    k = st.selectbox("Forward step no.", res.step.tolist())
    r = res[res.step == k].iloc[0]
    pre, pos = st.slider("Window relative to heel-off (s)", -3.0, 2.0, (-1.8, 1.0), 0.1)
    w = (t > r.HO_s + pre) & (t < r.HO_s + pos)
    bw = bws[k]
    tt = t[w] - r.HO_s
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Leg vertical (re. baseline)", "L5 ML (re. baseline)",
                                        "L5 AP (re. baseline, + forward)"))
    for i, (col, cor) in enumerate((("gy", "black"), ("ml", "#1f77b4"), ("ap", "#ff7f0e")), 1):
        s = sig[col].values
        b0 = s[bw].mean()
        fig.add_scatter(x=tt, y=s[w] - b0, line=dict(color=cor), row=i, col=1)
        if col != "gy":
            nm = col.upper()
            thr = r[f"{nm}_onset_threshold_m_s2"]
            if np.isnan(thr):
                thr = r[f"{nm}_threshold_m_s2"]
            fig.add_hrect(y0=-thr, y1=thr, fillcolor="rgba(128,128,128,0.28)", line_width=0, row=i, col=1)
            if not np.isnan(r.get(f"{nm}_onset_rel_HO_ms", np.nan)):
                x0 = r[f"{nm}_onset_rel_HO_ms"] / 1000
                fig.add_vrect(x0=x0, x1=0, fillcolor="rgba(255,215,0,0.18)", line_width=0, row=i, col=1)
                fig.add_vline(x=x0, line=dict(color=cor, dash="dot"), row=i, col=1)
                fig.add_scatter(x=[r[f"{nm}_peak_rel_HO_ms"] / 1000], y=[r[f"{nm}_peak_m_s2"]],
                                mode="markers", marker=dict(size=10, color=cor, symbol="x"), row=i, col=1)
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    fig.add_vrect(x0=t[bw][0] - r.HO_s, x1=t[bw][-1] - r.HO_s, fillcolor="rgba(0,128,255,0.08)",
                  line_width=0, row="all", col=1)
    fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="Time relative to heel-off (s)", row=3, col=1)
    stretch(st.plotly_chart, fig)
    st.caption("Blue band = baseline · grey band = threshold · yellow = APA (onset → heel-off) · "
               "dotted = APA onset · × = peak · red dashed = heel-off.")

    # ---------------- 4. All steps aligned + mean ----------------
    st.subheader("4. All forward steps aligned to heel-off")
    tt = np.arange(pre, pos, 1 / fs)
    colors = pc.qualitative.Plotly + pc.qualitative.D3
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Leg vertical (re. baseline)", "L5 ML (re. baseline)",
                                        "L5 AP (re. baseline, + forward)"))
    mean_on = {nm: np.nanmean(res[f"{nm}_onset_rel_HO_ms"]) / 1000 for nm in ("ML", "AP")}
    sd_on = {nm: np.nanstd(res[f"{nm}_onset_rel_HO_ms"]) / 1000 for nm in ("ML", "AP")}
    apa_start = {1: min(mean_on.values()), 2: mean_on["ML"], 3: mean_on["AP"]}
    for row in (1, 2, 3):  # APA shading (mean onset -> heel-off)
        if not np.isnan(apa_start[row]):
            fig.add_vrect(x0=apa_start[row], x1=0, fillcolor="rgba(255,215,0,0.20)",
                          line_width=0, row=row, col=1)
    for row, nm in ((2, "ML"), (3, "AP")):
        if not np.isnan(mean_on[nm]):
            fig.add_annotation(x=mean_on[nm] / 2, y=1, yref=f"y{row} domain", showarrow=False,
                               text=f"APA · onset {mean_on[nm]*1000:.0f} ± {sd_on[nm]*1000:.0f} ms",
                               font=dict(size=11), row=row, col=1)
    for row, col in ((1, "gy"), (2, "ml"), (3, "ap")):
        s = sig[col].values
        M = []
        for j, (_, rr) in enumerate(res.iterrows()):
            cor = colors[j % len(colors)]
            y = np.interp(tt + rr.HO_s, t, s) - s[bws[rr.step]].mean()
            M.append(y)
            fig.add_scatter(x=tt, y=y, line=dict(color=cor, width=1.2), opacity=0.75,
                            name=f"Step {rr.step}", legendgroup=f"s{rr.step}",
                            showlegend=(row == 1), row=row, col=1)
            if col in ("ml", "ap"):
                nm = col.upper()
                if not np.isnan(rr[f"{nm}_onset_rel_HO_ms"]):
                    xo = rr[f"{nm}_onset_rel_HO_ms"] / 1000
                    fig.add_scatter(x=[xo], y=[np.interp(xo, tt, y)], mode="markers",
                                    marker=dict(color=cor, size=8, symbol="circle-open", line=dict(width=2)),
                                    name=f"Step {rr.step} onset", legendgroup=f"s{rr.step}",
                                    showlegend=False, row=row, col=1)
                    fig.add_scatter(x=[rr[f"{nm}_peak_rel_HO_ms"] / 1000], y=[rr[f"{nm}_peak_m_s2"]],
                                    mode="markers", marker=dict(color=cor, size=9, symbol="x"),
                                    name=f"Step {rr.step} peak", legendgroup=f"s{rr.step}",
                                    showlegend=False, row=row, col=1)
        M = np.array(M)
        mu, sd = M.mean(0), M.std(0)
        fig.add_scatter(x=np.r_[tt, tt[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself",
                        fillcolor="rgba(0,0,0,0.08)", line=dict(width=0), hoverinfo="skip",
                        name="Mean ± SD", legendgroup="mean", showlegend=(row == 1), row=row, col=1)
        fig.add_scatter(x=tt, y=mu, line=dict(color="black", width=3), name=f"Mean (n={len(M)})",
                        legendgroup="mean", showlegend=(row == 1), row=row, col=1)
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    fig.update_yaxes(title_text="m/s²")
    fig.update_xaxes(title_text="Time relative to heel-off (s)", row=3, col=1)
    fig.update_layout(height=900, margin=dict(t=40, b=40), legend=dict(groupclick="togglegroup"))
    stretch(st.plotly_chart, fig)
    st.caption("Yellow = APA (mean onset → heel-off) · thin lines = individual steps · "
               "○ = step onset · × = step peak · black = mean ± SD · red dashed = heel-off. "
               "Click a step in the legend to hide or show it.")

    # ---------------- 5. ML x AP and resultant ----------------
    st.subheader("5. ML × AP and horizontal resultant")
    c1, c2 = st.columns([1, 1])
    a0, a1 = c1.slider("Time window for the ML × AP plot (s, relative to heel-off)",
                       -2.0, 1.0, (-1.0, 0.0), 0.05)
    show_mean_xy = c2.checkbox("Show mean trajectory", True)
    txy = np.arange(a0, a1 + 1e-9, 1 / fs)
    traj = {}
    fig = go.Figure()
    for j, (_, rr) in enumerate(res.iterrows()):
        cor = colors[j % len(colors)]
        bw_ = bws[rr.step]
        x = np.interp(txy + rr.HO_s, t, sig.ml.values) - sig.ml.values[bw_].mean()
        y = np.interp(txy + rr.HO_s, t, sig.ap.values) - sig.ap.values[bw_].mean()
        traj[rr.step] = (x, y)
        fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=cor, width=1.5),
                        name=f"Step {rr.step}", legendgroup=f"s{rr.step}",
                        customdata=txy * 1000,
                        hovertemplate="t = %{customdata:.0f} ms<br>ML = %{x:.2f}<br>AP = %{y:.2f}")
        # markers: APA onset (AP) and heel-off
        for nm, sym in (("AP", "circle-open"), ("ML", "diamond-open")):
            on = rr[f"{nm}_onset_rel_HO_ms"]
            if not np.isnan(on) and a0 <= on / 1000 <= a1:
                fig.add_scatter(x=[np.interp(on / 1000, txy, x)], y=[np.interp(on / 1000, txy, y)],
                                mode="markers", marker=dict(color=cor, size=9, symbol=sym, line=dict(width=2)),
                                legendgroup=f"s{rr.step}", showlegend=False, hoverinfo="skip")
        if a0 <= 0 <= a1:
            fig.add_scatter(x=[np.interp(0, txy, x)], y=[np.interp(0, txy, y)], mode="markers",
                            marker=dict(color=cor, size=9, symbol="square"),
                            legendgroup=f"s{rr.step}", showlegend=False, hoverinfo="skip")
    if show_mean_xy and traj:
        X = np.mean([v[0] for v in traj.values()], 0)
        Y = np.mean([v[1] for v in traj.values()], 0)
        fig.add_scatter(x=X, y=Y, mode="lines", line=dict(color="black", width=3.5),
                        name=f"Mean (n={len(traj)})", customdata=txy * 1000,
                        hovertemplate="t = %{customdata:.0f} ms<br>ML = %{x:.2f}<br>AP = %{y:.2f}")
    fig.add_hline(y=0, line=dict(color="grey", width=1))
    fig.add_vline(x=0, line=dict(color="grey", width=1))
    fig.update_xaxes(title_text="L5 ML (m/s², re. baseline)", zeroline=False)
    fig.update_yaxes(title_text="L5 AP (m/s², re. baseline, + forward)", zeroline=False,
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(height=650, margin=dict(t=30, b=40), legend=dict(groupclick="togglegroup"))
    stretch(st.plotly_chart, fig)
    st.caption("Each line is the L5 acceleration path in the horizontal plane within the chosen "
               "window · ○ = AP onset · ◇ = ML onset · ■ = heel-off · black = mean trajectory. "
               "Axes use the same scale.")

    # resultant over time
    fig = go.Figure()
    mean_on_all = min(v for v in mean_on.values() if not np.isnan(v)) if not all(np.isnan(v) for v in mean_on.values()) else np.nan
    if not np.isnan(mean_on_all):
        fig.add_vrect(x0=mean_on_all, x1=0, fillcolor="rgba(255,215,0,0.20)", line_width=0)
    M = []
    for j, (_, rr) in enumerate(res.iterrows()):
        cor = colors[j % len(colors)]
        bw_ = bws[rr.step]
        x = np.interp(tt + rr.HO_s, t, sig.ml.values) - sig.ml.values[bw_].mean()
        y = np.interp(tt + rr.HO_s, t, sig.ap.values) - sig.ap.values[bw_].mean()
        rres = np.hypot(x, y)
        M.append(rres)
        fig.add_scatter(x=tt, y=rres, line=dict(color=cor, width=1.2), opacity=0.75,
                        name=f"Step {rr.step}", legendgroup=f"s{rr.step}")
        fig.add_scatter(x=[rr.RES_peak_rel_HO_ms / 1000], y=[rr.RES_peak_m_s2], mode="markers",
                        marker=dict(color=cor, size=9, symbol="x"), legendgroup=f"s{rr.step}",
                        showlegend=False, hoverinfo="skip")
    M = np.array(M); mu, sd = M.mean(0), M.std(0)
    fig.add_scatter(x=np.r_[tt, tt[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself",
                    fillcolor="rgba(0,0,0,0.08)", line=dict(width=0), hoverinfo="skip",
                    name="Mean ± SD", legendgroup="mean")
    fig.add_scatter(x=tt, y=mu, line=dict(color="black", width=3), name=f"Mean (n={len(M)})",
                    legendgroup="mean")
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    fig.update_layout(title="Horizontal resultant √(ML² + AP²)", height=450,
                      margin=dict(t=50, b=40), legend=dict(groupclick="togglegroup"),
                      xaxis_title="Time relative to heel-off (s)", yaxis_title="m/s² (re. baseline)")
    stretch(st.plotly_chart, fig)
    st.caption("Yellow = APA (earliest mean onset → heel-off) · × = peak resultant before heel-off "
               "for each step · red dashed = heel-off.")

    # ---------------- 6. Results ----------------
    st.subheader("6. Results")
    show = ["step", "HO_s", "HO_source",
            "ML_direction", "ML_onset_rel_HO_ms", "ML_peak_m_s2", "ML_peak_rel_HO_ms", "ML_dv_m_s",
            "AP_onset_rel_HO_ms", "AP_peak_m_s2", "AP_peak_rel_HO_ms", "AP_dv_m_s",
            "RES_peak_m_s2", "RES_peak_rel_HO_ms", "RES_angle_deg"]
    show = [c for c in show if c in res.columns]
    stretch(st.dataframe, res[show], hide_index=True)
    num = [c for c in show if c not in ("step", "HO_s", "HO_source", "ML_direction")]
    summ = res[num].agg(["mean", "std", "median", "min", "max"]).round(2).T
    st.markdown("**Summary (all forward steps)**")
    stretch(st.dataframe, summ)

    buf = io.StringIO()
    res.to_csv(buf, index=False)
    st.download_button("Download results (CSV)", buf.getvalue(), "apa_results.csv", "text/csv")

    with st.expander("Method notes"):
        st.markdown(
            "- **Synchronisation:** cross-correlation of the acceleration norm |a| of both sensors "
            "around the largest leg impact (the jump).\n"
            "- **Steps:** activity bursts on the leg vertical axis; forward/backward alternate from "
            "the chosen first step (or, in automatic mode, from the sign of the largest L5 AP excursion).\n"
            "- **Heel-off:** from the forward AP peak − 0.3 s, first leg vertical peak above baseline + "
            "threshold; heel-off = start of that rise.\n"
            "- **Baseline:** quietest window before heel-off (between HO − 2.6 s and HO − 0.6 s).\n"
            "- **APA peak:** largest deviation in the dominant direction between the end of the "
            "baseline and heel-off.\n"
            "- **APA onset (default):** from the peak, walk back in time to the last sample inside "
            "baseline ± max(k·SD, minimum); earlier oscillations that returned to baseline are ignored. "
            "Alternatives: the same with a threshold of a % of the peak amplitude, or the first "
            "crossing that stays outside the band for the minimum time.\n"
            "- **Horizontal resultant:** √(ML² + AP²) of the baseline-corrected signals; peak "
            "between the end of the baseline and heel-off. Angle at the peak: 0° = forward, "
            "+90° = +ML direction.\n"
            "- **dv:** integral of acceleration from APA onset to heel-off (velocity change).\n"
            "- **Caution:** the L5 AP signal includes the gravity projection when the trunk tilts "
            "(≈0.17 m/s² per degree) as well as linear acceleration.")


main()

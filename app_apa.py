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


def find_jump_and_lag(L, G, fs, sync_thr, max_lag=0.5):
    """Sync event = FIRST time the leg acceleration norm departs from its initial resting value
    by more than `sync_thr` m/s². Lag from cross-correlation of the acceleration norms of both
    sensors in a window around that event (lag > 0: leg lags L5)."""
    t = np.arange(0, min(L.t.iloc[-1], G.t.iloc[-1]), 1 / fs)
    mL = np.sqrt(sum(np.interp(t, L.t, L[a]) ** 2 for a in "xyz"))
    mG = np.sqrt(sum(np.interp(t, G.t, G[a]) ** 2 for a in "xyz"))
    rest = np.median(mG[: int(fs)])
    dev = np.abs(lowpass(mG, 20, fs) - rest) > sync_thr
    if not dev.any():
        return np.nan, 0.0, np.nan
    i0 = int(np.argmax(dev))
    t0 = t[i0]
    # end of the sync burst: first 0.4 s of quiet after it
    quiet = np.abs(lowpass(mG, 5, fs) - rest) < 0.5
    j = i0
    while j < len(t) - int(0.4 * fs) and not quiet[j:j + int(0.4 * fs)].all():
        j += 1
    t_end = t[j]
    w = (t > t0 - 0.5) & (t < t_end + 0.3)
    a_, b_ = mL[w] - mL[w].mean(), mG[w] - mG[w].mean()
    c = np.correlate(b_, a_, "full")
    lags = np.arange(-len(a_) + 1, len(a_)) / fs
    k = np.abs(lags) <= max_lag
    return t0, lags[k][np.argmax(c[k])], t_end


def build_signals(L, G, p):
    """Resample, synchronise and filter."""
    fs = p["fs"]
    t = np.arange(0, min(L.t.iloc[-1], G.t.iloc[-1] - p["lag"]), 1 / fs)
    sig = pd.DataFrame({"t": t})
    sig["ml"] = lowpass(np.interp(t, L.t, L[p["ml_axis"]]) * p["ml_sign"], p["fc_l5"], fs)
    sig["ap"] = lowpass(np.interp(t, L.t, L[p["ap_axis"]]) * p["ap_sign"], p["fc_l5"], fs)
    sig["gy"] = lowpass(np.interp(t, G.t - p["lag"], G[p["leg_axis"]]), p["fc_leg"], fs)
    for a in "xyz":  # slow components of the leg sensor (gravity direction) for shank tilt
        sig[f"leg_{a}"] = lowpass(np.interp(t, G.t - p["lag"], G[a]), 3.0, fs)
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


def leg_baseline(sig, bs, length=0.5):
    """Quietest `length`-s window of the leg signals between bs-2.5 s and bs-0.3 s."""
    t = sig.t.values
    V = sig[["leg_x", "leg_y", "leg_z"]].values
    best = None
    for s0 in np.arange(bs - 2.5, bs - 0.3 - length + 1e-9, 0.05):
        w = (t >= s0) & (t < s0 + length)
        if w.sum() < 5:
            continue
        sc = sig.gy.values[w].std() + V[w].std(0).sum()
        if best is None or sc < best[0]:
            best = (sc, w)
    return best[1] if best else (t >= bs - 1.5) & (t < bs - 1.0)


def shank_tilt(sig, bw):
    """Angle (deg) between the leg acceleration vector and its baseline direction."""
    V = sig[["leg_x", "leg_y", "leg_z"]].values
    v0 = V[bw].mean(0)
    c = (V @ v0) / (np.linalg.norm(V, axis=1) * np.linalg.norm(v0))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def heel_off_candidates(sig, bs, p):
    """Leg-only heel-off detectors (independent of L5). Returns dict with the three
    candidate times and the info needed to plot them."""
    fs, t, gy = p["fs"], sig.t.values, sig.gy.values
    bw = leg_baseline(sig, bs)
    g0, gsd = gy[bw].mean(), gy[bw].std()
    n = max(1, int(round(p["ho_dur"] * fs)))
    out = dict(g0=g0, bw=bw)

    # 1) first sustained departure of the leg vertical signal from baseline
    thr_v = max(p["ho_k"] * gsd, p["ho_min_v"])
    seg = np.flatnonzero((t > bs - 1.0) & (t < bs + 1.0))
    ob = np.abs(gy[seg] - g0) > thr_v
    out["depart"], out["thr_v"] = np.nan, thr_v
    for i in range(len(seg) - n + 1):
        if ob[i:i + n].all():
            out["depart"] = t[seg[i]]; break

    # 2) start of the rise to the first leg vertical peak above baseline + ho_thr
    out["rise"] = np.nan
    seg = np.flatnonzero((t > bs - 1.0) & (t < bs + 1.5))
    above = gy[seg] - g0 > p["ho_thr"]
    if above.any():
        pk = seg[np.argmax(above)]
        while pk + 1 < len(gy) and gy[pk + 1] > gy[pk]:
            pk += 1
        j = pk
        while j > 0 and gy[j - 1] < gy[j]:
            j -= 1
        out["rise"] = t[j]

    # 3) shank tilt (3-axis gravity direction) leaves its baseline
    ang = shank_tilt(sig, bw)
    thr_a = max(p["ho_k"] * ang[bw].std(), p["ho_min_deg"])
    seg = np.flatnonzero((t > bs - 1.0) & (t < bs + 1.0))
    ob = ang[seg] > thr_a
    out["tilt"], out["thr_a"], out["ang"] = np.nan, thr_a, ang
    for i in range(len(seg) - n + 1):
        if ob[i:i + n].all():
            out["tilt"] = t[seg[i]]; break
    return out


def detect_heel_off(sig, bs, p):
    return heel_off_candidates(sig, bs, p)[p["ho_method"]]


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
            d = p.get("ap_dir", 1)   # +1 forward step, −1 backward step
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
    tj, lag_auto, tj_end = find_jump_and_lag(L, G, p["fs"], p["sync_thr"])
    if p.get("lag") is None:
        p["lag"] = lag_auto
    sig = build_signals(L, G, p)
    bursts = detect_bursts(sig, p, tj_end - p["lag"] + p["post_jump"])
    ev = []
    for i, (bs, be) in enumerate(bursts, 1):
        if p["class_mode"] == "alternating":
            kind = FWD if i % 2 == 1 else BWD   # first step after the sync event = forward
        else:
            kind = classify_burst(sig, bs, p)
        ho = detect_heel_off(sig, bs, p)
        ev.append(dict(event=i, burst_start_s=round(bs, 2), burst_end_s=round(be, 2),
                       type=kind, HO_auto_s=round(ho, 3) if not np.isnan(ho) else np.nan))
    return dict(t_jump=tj, t_jump_end=tj_end, lag_auto=lag_auto, sig=sig, events=pd.DataFrame(ev))


# =============================================================================
# Streamlit interface
# =============================================================================

def main():
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.colors as pc

    st.set_page_config(page_title="Step APA", layout="wide")

    def stretch(fn, *a, **k):
        """Works with new (width='stretch') and older (use_container_width) Streamlit."""
        try:
            return fn(*a, width="stretch", **k)
        except TypeError:
            return fn(*a, use_container_width=True, **k)

    # ---------------- Sidebar ----------------
    sb = st.sidebar
    sb.header("Files")
    fL5 = sb.file_uploader("L5 accelerometer", type=["txt", "csv"])
    fLeg = sb.file_uploader("Leg accelerometer", type=["txt", "csv"])

    sb.header("Analysis")
    sel_type = sb.radio("Steps to analyse", [FWD, BWD], 0, horizontal=True,
                        format_func=lambda x: x.capitalize(),
                        help="All forward steps together, or all backward steps together. "
                             "For backward steps the AP APA is searched in the backward (−) direction.")
    Sel = sel_type.capitalize()
    st.title(f"{Sel}-step APA — L5 × leg")

    ho_method, onset_method = "rise", "backward"   # fixed: most reliable criteria
    k_sd = sb.slider(
        "APA threshold = k × L5 baseline SD", 1.0, 10.0, 5.0, 0.5,
        help="Width of the grey band around the L5 baseline. The APA onset is the last moment the "
             "signal was inside this band before the APA peak. Lower k = earlier, more sensitive onset "
             "(more affected by noise); higher k = later, more conservative onset. Common choices in "
             "the APA literature are 2–3 SD; 5 SD is conservative. Report the value you use.")

    sb.markdown("---")
    sb.caption("Advanced settings — the defaults fit the current protocol.")

    with sb.expander("Sensor axes"):
        c1, c2 = st.columns(2)
        ml_axis = c1.selectbox("L5 ML axis", ["x", "y", "z"], 0)
        ml_sign = c2.selectbox("ML sign", [1, -1], 0)
        ap_axis = c1.selectbox("L5 AP axis", ["x", "y", "z"], 2)
        ap_sign = c2.selectbox("AP sign (+ = forward)", [1, -1], 1,
                               help="Phone on L5 with the screen facing out: z points backwards, "
                                    "so −1 makes + forward.")
        leg_axis = c1.selectbox("Leg vertical axis", ["x", "y", "z"], 1)

    with sb.expander("Filters and synchronisation"):
        fs = st.number_input("Resampling rate (Hz)", 50, 500, 100, 10)
        fc_l5 = st.slider("L5 low-pass (Hz)", 1.0, 20.0, 5.0, 0.5)
        fc_leg = st.slider("Leg low-pass (Hz)", 2.0, 30.0, 10.0, 0.5)
        sync_thr = st.slider("Sync event threshold (m/s²)", 0.5, 5.0, 1.5, 0.1,
                             help="The sync event is the FIRST leg disturbance larger than this.")
        lag_manual = st.checkbox("Set lag manually")
        lag_val = st.number_input("Leg lag relative to L5 (s)", -2.0, 2.0, 0.25, 0.01,
                                  disabled=not lag_manual)

    with sb.expander("Step detection"):
        burst_thr = st.slider("Leg activity threshold (m/s²)", 0.05, 1.5, 0.25, 0.05)
        burst_merge = st.slider("Merge bursts closer than (s)", 0.1, 2.0, 0.8, 0.1)
        burst_min_dur = st.slider("Minimum burst duration (s)", 0.1, 1.5, 0.3, 0.05)
        post_jump = st.slider("Ignore X s after the sync event", 0.0, 3.0, 0.5, 0.1)
        class_mode = st.radio("Forward/backward labelling", ["Alternating", "Automatic (AP)"], 0,
                              help="Alternating: 1st step after the sync event = forward.")
        class_thr = st.slider("AP threshold, automatic labelling (m/s²)", 0.3, 2.0, 0.8, 0.1,
                              disabled=class_mode == "Alternating")

    with sb.expander("Heel-off and APA thresholds"):
        ho_thr = st.slider("Heel-off: minimum leg peak above baseline (m/s²)", 0.3, 3.0, 0.8, 0.1,
                           help="Heel-off = start of the main rise of the leg vertical signal "
                                "up to its first peak above this value.")
        min_abs = st.slider("APA minimum band width (m/s²)", 0.0, 0.5, 0.1, 0.02)
        base_len = st.slider("L5 baseline window (s)", 0.2, 1.0, 0.5, 0.05)
    ho_k, ho_min_v, ho_min_deg, ho_dur = 5.0, 0.15, 2.0, 0.05
    peak_frac, min_dur = 0.15, 0.1

    if not (fL5 and fLeg):
        st.info("Upload both files (L5 and leg) in the sidebar to start.")
        st.stop()

    L, G = read_acc(fL5), read_acc(fLeg)
    p = dict(fs=fs, fc_l5=fc_l5, fc_leg=fc_leg, ml_axis=ml_axis, ml_sign=ml_sign,
             ap_axis=ap_axis, ap_sign=ap_sign, leg_axis=leg_axis,
             sync_thr=sync_thr, lag=lag_val if lag_manual else None,
             burst_thr=burst_thr, burst_merge=burst_merge, burst_min_dur=burst_min_dur,
             post_jump=post_jump, class_thr=class_thr, ho_thr=ho_thr,
             ho_method=ho_method, ho_k=ho_k, ho_min_v=ho_min_v, ho_min_deg=ho_min_deg,
             ho_dur=ho_dur, ap_dir=1 if sel_type == FWD else -1,
             class_mode="alternating" if class_mode == "Alternating" else "auto",
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
    c[2].metric("Sync event (leg)", f"{R['t_jump']:.2f}–{R['t_jump_end']:.2f} s")
    c[3].metric("Lag used (leg behind L5)", f"{p['lag']:.3f} s",
                f"auto = {R['lag_auto']:.3f} s", delta_color="off")

    with st.expander("Check the alignment at the sync event", expanded=True):
        w = (t > R["t_jump"] - p["lag"] - 1.0) & (t < R["t_jump_end"] - p["lag"] + 1.0)
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
    ev = R["events"].copy()
    if ev.empty:
        st.warning("No steps detected. Lower the leg activity threshold in the sidebar.")
        st.stop()
    ev["HO_manual_s"] = np.nan
    ev_box = st.expander("Event table (edit type or set a manual heel-off)", expanded=False)
    ev_box.caption("Edit the **type** column (forward / backward / ignore) and, if needed, enter "
                   "**HO_manual_s** to override the automatic heel-off.")
    with ev_box:
        ev = stretch(
        st.data_editor, ev, hide_index=True,
        column_config={
            "type": st.column_config.SelectboxColumn(options=[FWD, BWD, IGN, UND]),
            "HO_manual_s": st.column_config.NumberColumn(format="%.3f"),
        },
        disabled=["event", "burst_start_s", "burst_end_s", "HO_auto_s"], key="ev_editor")

    ev["HO_s"] = ev.HO_manual_s.fillna(ev.HO_auto_s)
    fwd = ev[(ev.type == sel_type) & ev.HO_s.notna()].reset_index(drop=True)

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
    fig.add_vrect(x0=R["t_jump"] - p["lag"], x1=R["t_jump_end"] - p["lag"],
                  fillcolor="rgba(148,103,189,0.18)", line_width=0, row="all", col=1)
    for _, r in ev[ev.type.isin([FWD, BWD]) & ev.HO_s.notna()].iterrows():
        fig.add_vline(x=r.HO_s, line=dict(color="red", dash="dash", width=1))
    for _, r in res.iterrows():
        if not np.isnan(r.get("ML_onset_s", np.nan)):
            fig.add_vline(x=r.ML_onset_s, line=dict(color="#1f77b4", dash="dot", width=1), row=2, col=1)
        if not np.isnan(r.get("AP_onset_s", np.nan)):
            fig.add_vline(x=r.AP_onset_s, line=dict(color="#ff7f0e", dash="dot", width=1), row=3, col=1)
    fig.update_layout(height=650, showlegend=False, margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    stretch(st.plotly_chart, fig)
    st.caption("Green = forward step · red = backward step · purple = sync event · red dashed = heel-off · "
               f"dotted = APA onset of the analysed ({sel_type}) steps (ML blue, AP orange).")

    if res.empty:
        st.warning(f"No {sel_type} step with a heel-off. Check the event table.")
        st.stop()

    # ---------------- 3. Single step ----------------
    st.subheader("3. Single step")
    k = st.selectbox(f"{Sel} step no.", res.step.tolist())
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
               "dotted = APA onset · × = peak · red dashed = heel-off (start of the main rise of the "
               "leg vertical signal).")
    # ---------------- 4. All steps aligned + mean ----------------
    st.subheader(f"4. All {sel_type} steps aligned to heel-off")
    res_all = res.copy()
    excl = st.multiselect(
        "Exclude steps from the analysis (sections 4–6, means and summary)",
        res_all.step.tolist(), key=f"excl_{sel_type}",
        format_func=lambda k_: f"Step {k_} (event {int(res_all.loc[res_all.step == k_, 'event'].iloc[0])})")
    res_all["included"] = ~res_all.step.isin(excl)
    res = res_all[res_all.included].reset_index(drop=True)
    if res.empty:
        st.warning("All steps are excluded.")
        st.stop()
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
    st.subheader("5. ML × AP path and horizontal resultant")
    amp_norm = st.radio(
        "Amplitude", ["Absolute (m/s²)", "Normalised to each step's peak (0–1)"], 1, horizontal=True,
        help="Normalised: each step is divided by its own peak horizontal resultant, so steps are "
             "compared by shape and direction. Display only — the results table stays in m/s².").startswith("Normalised")
    st.caption("Time is normalised: each step runs from its APA onset (0 %, earliest of ML/AP onset) "
               "to heel-off (100 %), so steps with APAs of different durations are averaged phase by phase.")

    pct = np.linspace(0, 100, 101)
    ml_d, ap_d = lowpass(sig.ml.values, 3.0, fs), lowpass(sig.ap.values, 3.0, fs)   # display smoothing
    curves, skipped = [], []
    for _, rr in res.iterrows():
        ons = [rr[f"{nm}_onset_s"] for nm in ("ML", "AP") if not np.isnan(rr[f"{nm}_onset_s"])]
        if not ons:
            skipped.append(rr.step); continue
        bw_ = bws[rr.step]
        tq = min(ons) + pct / 100 * (rr.HO_s - min(ons))
        x = np.interp(tq, t, ml_d - ml_d[bw_].mean()); y = np.interp(tq, t, ap_d - ap_d[bw_].mean())
        xr = np.interp(tq, t, sig.ml.values - sig.ml.values[bw_].mean())
        yr = np.interp(tq, t, sig.ap.values - sig.ap.values[bw_].mean())
        sc = np.hypot(xr, yr).max() if amp_norm else 1.0
        curves.append((rr, x / sc, y / sc, xr / sc, yr / sc, (tq - rr.HO_s) * 1000))
    if skipped:
        st.caption(f"Step(s) {', '.join(map(str, skipped))} have no APA onset and are not shown here.")
    unit_lbl = "re. step peak" if amp_norm else "m/s², re. baseline"

    c1, c2 = st.columns([1, 1])
    # --- ML x AP path: individual steps faint, mean bold, SD ellipses at 25/50/75/100 %
    fig = go.Figure()
    for j, (rr, x, y, _, _, tip) in enumerate(curves):
        cor = colors[j % len(colors)]
        fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=cor, width=1.2), opacity=0.45,
                        name=f"Step {rr.step}", legendgroup=f"s{rr.step}", customdata=np.c_[pct, tip],
                        hovertemplate="%{customdata[0]:.0f} % (%{customdata[1]:.0f} ms re. HO)"
                                      "<br>ML = %{x:.2f}<br>AP = %{y:.2f}")
        fig.add_scatter(x=[x[-1]], y=[y[-1]], mode="markers", marker=dict(color=cor, size=7, symbol="square"),
                        opacity=0.6, legendgroup=f"s{rr.step}", showlegend=False, hoverinfo="skip")
    if curves:
        X = np.array([c_[1] for c_ in curves]); Y = np.array([c_[2] for c_ in curves])
        mx, my, sx, sy = X.mean(0), Y.mean(0), X.std(0), Y.std(0)
        th = np.linspace(0, 2 * np.pi, 60)
        for i, ph in enumerate((25, 50, 75, 100)):
            k_ = ph
            fig.add_scatter(x=mx[k_] + sx[k_] * np.cos(th), y=my[k_] + sy[k_] * np.sin(th), fill="toself",
                            fillcolor="rgba(31,119,180,0.12)", line=dict(color="rgba(31,119,180,0.4)", width=1),
                            name="± SD at 25/50/75/100 %", legendgroup="sd", showlegend=(i == 0),
                            hoverinfo="skip")
            fig.add_annotation(x=mx[k_], y=my[k_], text=f"{ph}%", showarrow=False,
                               xshift=18, font=dict(size=10, color="#1f4e79"))
        fig.add_scatter(x=mx, y=my, mode="lines", line=dict(color="black", width=4),
                        name=f"Mean (n={len(curves)})", customdata=pct,
                        hovertemplate="%{customdata:.0f} %<br>ML = %{x:.2f}<br>AP = %{y:.2f}")
        fig.add_scatter(x=[mx[0]], y=[my[0]], mode="markers", name="Onset (0 %)",
                        marker=dict(color="black", size=10, symbol="circle-open", line=dict(width=2)))
        fig.add_scatter(x=[mx[-1]], y=[my[-1]], mode="markers", name="Heel-off (100 %)",
                        marker=dict(color="black", size=10, symbol="square"))
    fig.add_hline(y=0, line=dict(color="grey", width=1))
    fig.add_vline(x=0, line=dict(color="grey", width=1))
    fig.update_xaxes(title_text=f"L5 ML ({unit_lbl})", zeroline=False)
    fig.update_yaxes(title_text=f"L5 AP ({unit_lbl}, + forward)", zeroline=False,
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(title="ML × AP path (onset → heel-off)", height=560,
                      margin=dict(t=50, b=40), legend=dict(groupclick="togglegroup"))
    with c1:
        stretch(st.plotly_chart, fig)

    # --- horizontal resultant over the normalised APA
    fig = go.Figure()
    M = []
    for j, (rr, _, _, xr, yr, tip) in enumerate(curves):
        cor = colors[j % len(colors)]
        rres = np.hypot(xr, yr); M.append(rres)
        fig.add_scatter(x=pct, y=rres, line=dict(color=cor, width=1.2), opacity=0.45,
                        name=f"Step {rr.step}", legendgroup=f"s{rr.step}")
    if M:
        M = np.array(M); mu, sd = M.mean(0), M.std(0)
        fig.add_scatter(x=np.r_[pct, pct[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself",
                        fillcolor="rgba(0,0,0,0.08)", line=dict(width=0), hoverinfo="skip",
                        name="Mean ± SD", legendgroup="mean")
        fig.add_scatter(x=pct, y=mu, line=dict(color="black", width=3), name=f"Mean (n={len(M)})",
                        legendgroup="mean")
    fig.update_layout(title="Horizontal resultant √(ML² + AP²)", height=560,
                      margin=dict(t=50, b=40), legend=dict(groupclick="togglegroup"),
                      xaxis_title="APA phase (% from onset to heel-off)", yaxis_title=unit_lbl)
    with c2:
        stretch(st.plotly_chart, fig)
    st.caption("Left: L5 acceleration path in the horizontal plane (paths smoothed at 3 Hz for display; "
               "equal axis scales) · thin lines = steps · black = mean path · blue ellipses = ± SD of ML "
               "and AP at 25, 50, 75 and 100 % of the APA. Right: size of the horizontal acceleration "
               "over the APA. Click a step in a legend to hide it.")

    # ---------------- 6. Results ----------------
    st.subheader("6. Results")
    show = ["step", "event", "HO_s", "HO_source",
            "ML_direction", "ML_onset_rel_HO_ms", "ML_peak_m_s2", "ML_peak_rel_HO_ms", "ML_dv_m_s",
            "AP_onset_rel_HO_ms", "AP_peak_m_s2", "AP_peak_rel_HO_ms", "AP_dv_m_s",
            "RES_peak_m_s2", "RES_peak_rel_HO_ms", "RES_angle_deg"]
    show = ["included"] + [c for c in show if c in res_all.columns]
    stretch(st.dataframe, res_all[show], hide_index=True)
    num = [c for c in show if c not in ("included", "step", "event", "HO_s", "HO_source", "ML_direction")]
    summ = res[num].agg(["mean", "std", "median", "min", "max"]).round(2).T
    st.markdown(f"**Summary ({sel_type} steps included: n = {len(res)})**")
    stretch(st.dataframe, summ)

    buf = io.StringIO()
    res_all.to_csv(buf, index=False)
    st.download_button("Download results (CSV)", buf.getvalue(), f"apa_results_{sel_type}.csv", "text/csv")

    with st.expander("Method notes"):
        st.markdown(
            "- **Synchronisation:** cross-correlation of the acceleration norm |a| of both sensors "
            "around the FIRST disturbance of the leg signal (the sync jump/tap at the start).\n"
            "- **Steps:** activity bursts on the leg vertical axis, numbered after the sync event; the first "
            "one is forward and they alternate (or, in automatic mode, from the sign of the largest "
            "L5 AP excursion).\n"
            "- **Heel-off (leg sensor only):** leg baseline = quietest 0.5 s between 2.5 and 0.3 s "
            "before the step burst; heel-off = start of the rise of the leg vertical signal to its first "
            "peak above baseline + threshold. Chosen because it does not change with the threshold "
            "(most repeatable of the leg-only criteria tested).\n"
            "- **Forward / backward:** the same pipeline is applied to the chosen step type; for "
            "backward steps the AP APA is searched in the backward (−) direction.\n"
            "- **Baseline:** quietest window before heel-off (between HO − 2.6 s and HO − 0.6 s).\n"
            "- **APA peak:** largest deviation in the dominant direction between the end of the "
            "baseline and heel-off.\n"
            "- **APA onset:** from the APA peak, walk back in time to the last sample inside "
            "baseline ± max(k·SD, minimum); earlier oscillations that returned to baseline are ignored.\n"
            "- **Horizontal resultant:** √(ML² + AP²) of the baseline-corrected signals; peak "
            "between the end of the baseline and heel-off. Angle at the peak: 0° = forward, "
            "+90° = +ML direction. It is the size of the horizontal acceleration vector, always ≥ 0; "
            "baseline noise gives it a small positive floor.\n"
            "- **Section 5:** each step's APA (earliest ML/AP onset → heel-off) resampled to 0–100 %; "
            "optional amplitude normalisation by each step's peak resultant (display only).\n"
            "- **dv:** integral of acceleration from APA onset to heel-off (velocity change).\n"
            "- **Caution:** the L5 AP signal includes the gravity projection when the trunk tilts "
            "(≈0.17 m/s² per degree) as well as linear acceleration.")


main()

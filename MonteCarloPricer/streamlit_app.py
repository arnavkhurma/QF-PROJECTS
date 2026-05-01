from __future__ import annotations

import io
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    # Works when launched from the repo root.
    from MonteCarloPricer.pricer import (
        MCPricerConfig,
        black_scholes_call,
        bump_greeks_mc,
        mc_price_call,
        mc_running_convergence,
        sample_gbm_paths,
    )
except ModuleNotFoundError:
    # Works when launched from inside MonteCarloPricer/ directory.
    from pricer import (
        MCPricerConfig,
        black_scholes_call,
        bump_greeks_mc,
        mc_price_call,
        mc_running_convergence,
        sample_gbm_paths,
    )


st.set_page_config(page_title="Core Pricing Dashboard", layout="wide")


def _indicator_gauge(value: float, title: str, domain_x: tuple[float, float] = (0, 1)) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            domain={"x": list(domain_x), "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "#2563EB"},
                "bgcolor": "white",
            },
        )
    )
    fig.update_layout(height=220, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _badge(text: str, ok: bool) -> str:
    bg = "#DCFCE7" if ok else "#FEE2E2"
    fg = "#166534" if ok else "#991B1B"
    return f"""
    <div style="display:inline-block;padding:6px 10px;border-radius:10px;
                background:{bg};color:{fg};font-weight:600;font-size:12px;">
        {text}
    </div>
    """


def _csv_download(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _coerce_term_structure(value: object, fallback: pd.DataFrame) -> pd.DataFrame:
    """Normalize session value to a clean term-structure DataFrame."""
    try:
        if isinstance(value, pd.DataFrame):
            df = value.copy()
        elif isinstance(value, list):
            df = pd.DataFrame(value)
        elif isinstance(value, dict):
            # Data editor state can be a dict with metadata keys.
            if "data" in value:
                df = pd.DataFrame(value["data"])
            else:
                df = pd.DataFrame(value)
        else:
            return fallback.copy()
    except Exception:
        return fallback.copy()

    for col in ("T", "sigma"):
        if col not in df.columns:
            df[col] = np.nan

    df = df[["T", "sigma"]].copy()
    return df


st.title("Core Pricing Dashboard")

with st.sidebar:
    st.subheader("Parameters")
    S0 = st.slider("S0", 1.0, 300.0, float(st.session_state.get("S0", 100.0)), 0.5)
    K = st.slider("K", 1.0, 300.0, float(st.session_state.get("K", 100.0)), 0.5)
    r = st.slider("r", -0.05, 0.20, float(st.session_state.get("r", 0.05)), 0.001)
    sigma = st.slider("σ", 0.01, 1.50, float(st.session_state.get("sigma", 0.20)), 0.01)
    T = st.slider("T (years)", 1 / 365.0, 10.0, float(st.session_state.get("T", 1.0)), 0.01)

    st.divider()
    st.subheader("Monte Carlo")
    N = st.slider("Paths (N)", 1_000, 1_000_000, int(st.session_state.get("N", 100_000)), step=1_000)
    block_size = st.slider("Block size", 1_000, 200_000, int(st.session_state.get("block_size", 10_000)), step=1_000)
    use_antithetic = st.toggle("Antithetic variates", bool(st.session_state.get("use_antithetic", True)))
    seed = st.number_input("Seed (optional)", value=st.session_state.get("seed", 42), step=1)
    seed_enabled = st.toggle("Use seed", bool(st.session_state.get("seed_enabled", True)))

    st.divider()
    st.subheader("UX / Benchmark")
    benchmark = st.toggle("Benchmark mode (timed BS vs MC)", bool(st.session_state.get("benchmark", True)))
    live_plots = st.toggle("Live diagnostics plots", bool(st.session_state.get("live_plots", True)))

    st.session_state.update(
        {
            "S0": S0,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "N": N,
            "block_size": block_size,
            "use_antithetic": use_antithetic,
            "seed": seed,
            "seed_enabled": seed_enabled,
            "benchmark": benchmark,
            "live_plots": live_plots,
        }
    )


cfg = MCPricerConfig(
    S0=S0,
    K=K,
    r=r,
    sigma=sigma,
    T=T,
    N=N,
    block_size=block_size,
    use_antithetic=use_antithetic,
    seed=int(seed) if seed_enabled else None,
)


colA, colB, colC = st.columns([1.2, 1.2, 1.0], vertical_alignment="top")

with colA:
    st.subheader("Side-by-side: MC vs Black–Scholes")
    t0 = time.perf_counter()
    bs_price = black_scholes_call(S0, K, r, sigma, T)
    bs_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    mc_price, mc_stats = mc_price_call(cfg)
    mc_ms = (time.perf_counter() - t1) * 1000.0

    abs_err = mc_price - bs_price
    rel_err = abs(abs_err) / bs_price if bs_price != 0 else np.nan
    ci_lo, ci_hi = mc_stats["confidence_interval_95"]
    within = (bs_price >= ci_lo) and (bs_price <= ci_hi)

    m1, m2, m3 = st.columns(3)
    m1.metric("MC price", f"{mc_price:.6f}", f"SE {mc_stats['std_error']:.6f}")
    m2.metric("BS price", f"{bs_price:.6f}")
    m3.metric("MC − BS", f"{abs_err:+.6f}", f"{rel_err:.3%}")
    st.caption(f"95% CI: [{ci_lo:.6f}, {ci_hi:.6f}]")
    st.markdown(_badge("BS is within MC 95% CI" if within else "BS is outside MC 95% CI", within), unsafe_allow_html=True)

    if benchmark:
        st.caption(f"Compute cost: BS {bs_ms:.2f} ms · MC {mc_ms:.2f} ms")

with colB:
    st.subheader("Error & CI (gauge / badge)")
    ci_width = max(ci_hi - ci_lo, 0.0)
    # Normalize to something interpretable: target width ~ 1% of BS price
    target = max(0.01 * abs(bs_price), 1e-12)
    ci_score = float(np.clip(1.0 - (ci_width / target), 0.0, 1.0))
    err_score = float(np.clip(1.0 - (abs(abs_err) / target), 0.0, 1.0))

    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=err_score,
            title={"text": "Accuracy vs target (|MC−BS|)"},
            domain={"x": [0, 0.48], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#16A34A"}},
        )
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=ci_score,
            title={"text": "CI tightness vs target"},
            domain={"x": [0.52, 1.0], "y": [0, 1]},
            gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#2563EB"}},
        )
    )
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Targets are heuristic (1% of BS price). Increase N to tighten CI.")

with colC:
    st.subheader("Variance reduction")
    if use_antithetic:
        st.metric("Variance reduction ratio", f"{mc_stats['variance_reduction_ratio']:.2f}×")
    else:
        st.metric("Variance reduction ratio", "1.00×")

    # Live comparison: antithetic vs naive, same N/seed
    cfg_plain = MCPricerConfig(**{**asdict(cfg), "use_antithetic": False})
    cfg_anti = MCPricerConfig(**{**asdict(cfg), "use_antithetic": True})
    _, st_plain = mc_price_call(cfg_plain)
    _, st_anti = mc_price_call(cfg_anti)
    ratio = (st_plain["std_error"] / st_anti["std_error"]) if st_anti["std_error"] > 0 else np.nan
    c1, c2 = st.columns(2)
    c1.metric("SE (naive)", f"{st_plain['std_error']:.6f}")
    c2.metric("SE (antithetic)", f"{st_anti['std_error']:.6f}")
    st.caption(f"SE ratio (naive / antithetic): {ratio:.2f}×")


st.divider()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Convergence & Diagnostics", "Payoff & Distributions", "Sensitivity / Greeks", "Scenario Analysis"]
)

with tab1:
    st.subheader("Live convergence + CI shrinkage")
    if live_plots:
        conv = mc_running_convergence(cfg)
        df_conv = pd.DataFrame({"n": conv["n"], "price": conv["price"], "std_error": conv["std_error"]})

        left, right = st.columns([1.2, 1.0])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_conv["n"], y=df_conv["price"], mode="lines+markers", name="MC estimate"))
            fig.add_hline(y=bs_price, line_dash="dash", line_color="gray", annotation_text="BS")
            fig.update_layout(
                height=320,
                xaxis_title="Paths accumulated",
                yaxis_title="Discounted price",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=df_conv["n"],
                    y=df_conv["std_error"],
                    mode="lines+markers",
                    name="Std error",
                )
            )
            # Reference slope -0.5: c / sqrt(n)
            n0 = float(df_conv["n"].iloc[0])
            se0 = float(df_conv["std_error"].iloc[0])
            ref = se0 * np.sqrt(n0) / np.sqrt(df_conv["n"].values)
            fig2.add_trace(go.Scatter(x=df_conv["n"], y=ref, mode="lines", name="−0.5 slope ref", line=dict(dash="dash")))
            fig2.update_layout(
                height=320,
                xaxis_title="Paths accumulated",
                yaxis_title="Std error",
                yaxis_type="log",
                xaxis_type="log",
                margin=dict(l=10, r=10, t=30, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.caption(r"Std error should scale roughly like $1/\sqrt{N}$ (slope -0.5 on log-log).")

    st.subheader("Accuracy vs compute")
    st.write("Move the **Paths (N)** slider in the sidebar to see CI tighten and compute time change.")


with tab2:
    st.subheader("Terminal price distribution (ST) + log-normal PDF overlay")
    ST = mc_stats["ST"]
    payoffs = mc_stats["payoffs"]

    c1, c2 = st.columns(2)
    with c1:
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=ST, nbinsx=60, name="Simulated ST", histnorm="probability density", opacity=0.65))

        # Lognormal PDF under risk-neutral measure for ST
        mu = np.log(S0) + (r - 0.5 * sigma**2) * T
        v = (sigma**2) * T
        x = np.linspace(np.percentile(ST, 0.5), np.percentile(ST, 99.5), 300)
        pdf = (1.0 / (x * np.sqrt(2 * np.pi * v))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * v))
        hist.add_trace(go.Scatter(x=x, y=pdf, mode="lines", name="Log-normal PDF"))
        hist.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="ST", yaxis_title="Density")
        st.plotly_chart(hist, use_container_width=True)

    with c2:
        st.subheader("Payoff distribution (max(ST−K,0))")
        figp = go.Figure()
        figp.add_trace(go.Histogram(x=payoffs, nbinsx=60, name="Payoffs", opacity=0.75))
        figp.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Payoff", yaxis_title="Count")
        st.plotly_chart(figp, use_container_width=True)

    st.subheader("Sample GBM paths")
    n_paths = st.slider("Number of sample paths", 5, 80, 20, 5, key="sample_paths")
    n_steps = st.slider("Time steps", 20, 400, 120, 20, key="sample_steps")
    t_grid, S_paths = sample_gbm_paths(S0, r, sigma, T, n_steps=n_steps, n_paths=n_paths, seed=int(seed) if seed_enabled else None)
    fig3 = go.Figure()
    for i in range(min(n_paths, S_paths.shape[0])):
        fig3.add_trace(go.Scatter(x=t_grid, y=S_paths[i], mode="lines", line=dict(width=1), showlegend=False))
    fig3.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="t", yaxis_title="S(t)")
    st.plotly_chart(fig3, use_container_width=True)


with tab3:
    st.subheader("Greeks via bump-and-reprice (MC, common random numbers)")
    g1, g2, g3 = st.columns([1.0, 1.0, 1.2])
    with g1:
        dS = st.number_input("ΔS bump", value=float(st.session_state.get("dS", 0.10)), min_value=0.001, step=0.01)
        st.session_state["dS"] = dS
    with g2:
        dV = st.number_input("Δσ bump", value=float(st.session_state.get("dV", 0.01)), min_value=0.0005, step=0.001, format="%.4f")
        st.session_state["dV"] = dV
    with g3:
        greeks = bump_greeks_mc(cfg, dS=dS, dSigma=dV)
        st.metric("Delta", f"{greeks['delta']:.6f}")
        st.metric("Vega", f"{greeks['vega']:.6f}")

    st.subheader("Heatmap: price over (S0, σ) grid (Black–Scholes for speed)")
    h1, h2, h3 = st.columns(3)
    with h1:
        s_min = st.number_input("S0 min", value=max(1.0, S0 * 0.7), step=1.0)
        s_max = st.number_input("S0 max", value=S0 * 1.3, step=1.0)
    with h2:
        v_min = st.number_input("σ min", value=max(0.01, sigma * 0.5), step=0.01)
        v_max = st.number_input("σ max", value=min(1.5, sigma * 1.5), step=0.01)
    with h3:
        grid_n = st.slider("Grid resolution", 10, 80, 35, 5)

    S_grid = np.linspace(float(s_min), float(s_max), grid_n)
    V_grid = np.linspace(float(v_min), float(v_max), grid_n)
    Z = np.zeros((grid_n, grid_n), dtype=float)
    for i, s in enumerate(S_grid):
        for j, v in enumerate(V_grid):
            Z[j, i] = black_scholes_call(s, K, r, v, T)

    heat = go.Figure(data=go.Heatmap(x=S_grid, y=V_grid, z=Z, colorscale="Viridis"))
    heat.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="S0", yaxis_title="σ")
    st.plotly_chart(heat, use_container_width=True)

    st.subheader("P&L surface (contour): price as a function of two parameters")
    contour = go.Figure(data=go.Contour(x=S_grid, y=V_grid, z=Z, contours_coloring="heatmap", colorscale="Turbo"))
    contour.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="S0", yaxis_title="σ")
    st.plotly_chart(contour, use_container_width=True)


with tab4:
    st.subheader("Batch pricer (upload or enter table)")
    st.caption("Columns supported: S0, K, r, sigma, T, N (optional), method (optional: BS/MC).")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
    else:
        batch_df = pd.DataFrame(
            [
                {"S0": S0, "K": K, "r": r, "sigma": sigma, "T": T, "N": min(N, 100_000), "method": "BS"},
                {"S0": S0 * 1.05, "K": K, "r": r, "sigma": sigma, "T": T, "N": min(N, 100_000), "method": "MC"},
            ]
        )

    edited = st.data_editor(batch_df, num_rows="dynamic", use_container_width=True)
    run_batch = st.button("Run batch pricing", type="primary")
    if run_batch and len(edited) > 0:
        rows = []
        for _, row in edited.iterrows():
            s = float(row.get("S0", S0))
            k = float(row.get("K", K))
            rr = float(row.get("r", r))
            vv = float(row.get("sigma", sigma))
            tt = float(row.get("T", T))
            nn = int(row.get("N", min(N, 100_000)))
            m = str(row.get("method", "BS")).upper()

            if m == "MC":
                cfg_row = MCPricerConfig(
                    S0=s,
                    K=k,
                    r=rr,
                    sigma=vv,
                    T=tt,
                    N=nn,
                    block_size=min(block_size, nn),
                    use_antithetic=use_antithetic,
                    seed=int(seed) if seed_enabled else None,
                )
                px, stx = mc_price_call(cfg_row)
                rows.append({**row.to_dict(), "price": px, "std_error": stx["std_error"], "ci95_lo": stx["confidence_interval_95"][0], "ci95_hi": stx["confidence_interval_95"][1]})
            else:
                px = black_scholes_call(s, k, rr, vv, tt)
                rows.append({**row.to_dict(), "price": px})

        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download batch results (CSV)", data=_csv_download(out), file_name="batch_prices.csv", mime="text/csv")

    st.divider()
    st.subheader("Vol term-structure viewer (σ(T) → price(T))")
    st.caption("Enter a simple term structure. Price is computed for the current (S0,K,r) across maturities.")
    default_ts = pd.DataFrame(
        [{"T": 0.25, "sigma": max(0.01, sigma * 0.9)}, {"T": 1.0, "sigma": sigma}, {"T": 2.0, "sigma": min(1.5, sigma * 1.1)}]
    )
    if "term_structure_df" not in st.session_state:
        st.session_state["term_structure_df"] = default_ts.copy()

    source_ts = _coerce_term_structure(st.session_state.get("term_structure_df"), default_ts)
    ts = st.data_editor(
        source_ts,
        num_rows="dynamic",
        use_container_width=True,
        key="term_structure_editor",
    )
    st.session_state["term_structure_df"] = _coerce_term_structure(ts, default_ts)
    if len(ts) >= 2:
        ts = ts.dropna()
        ts = ts.sort_values("T")
        T_vals = ts["T"].astype(float).to_numpy()
        sig_vals = ts["sigma"].astype(float).to_numpy()
        T_plot = np.linspace(float(T_vals.min()), float(T_vals.max()), 80)
        sig_plot = np.interp(T_plot, T_vals, sig_vals)
        px_plot = [black_scholes_call(S0, K, r, float(v), float(t)) for t, v in zip(T_plot, sig_plot)]

        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=T_plot, y=sig_plot, mode="lines", name="σ(T)"))
        fig_ts.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="T", yaxis_title="σ")
        st.plotly_chart(fig_ts, use_container_width=True)

        fig_tp = go.Figure()
        fig_tp.add_trace(go.Scatter(x=T_plot, y=px_plot, mode="lines", name="Price(T)"))
        fig_tp.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="T", yaxis_title="BS Price")
        st.plotly_chart(fig_tp, use_container_width=True)

    st.divider()
    st.subheader("Stress test (±X%)")
    stress = st.slider("Stress magnitude (±X%)", 0.0, 50.0, 10.0, 1.0) / 100.0
    stress_params = ["S0", "K", "sigma", "T"]
    which = st.multiselect("Parameters to stress", stress_params, default=["S0", "sigma"])

    def _apply_stress(name: str, sign: float) -> dict:
        params = {"S0": S0, "K": K, "sigma": sigma, "T": T}
        if name in which:
            params[name] = params[name] * (1.0 + sign * stress)
        return params

    if which:
        lo = _apply_stress(which[0], -1.0)
        hi = _apply_stress(which[0], +1.0)
        # If multiple selected, apply to all
        for nm in which[1:]:
            if nm in lo:
                lo[nm] *= (1.0 - stress)
                hi[nm] *= (1.0 + stress)

        px_lo = black_scholes_call(lo["S0"], lo["K"], r, lo["sigma"], lo["T"])
        px_hi = black_scholes_call(hi["S0"], hi["K"], r, hi["sigma"], hi["T"])
        st.metric("Stressed price range (BS)", f"[{min(px_lo, px_hi):.6f}, {max(px_lo, px_hi):.6f}]")


st.divider()
st.subheader("Export current run")
export_df = pd.DataFrame(
    [
        {
            "S0": S0,
            "K": K,
            "r": r,
            "sigma": sigma,
            "T": T,
            "N": N,
            "use_antithetic": use_antithetic,
            "mc_price": mc_price,
            "mc_std_error": mc_stats["std_error"],
            "mc_ci95_lo": mc_stats["confidence_interval_95"][0],
            "mc_ci95_hi": mc_stats["confidence_interval_95"][1],
            "bs_price": bs_price,
            "abs_error": abs_err,
            "rel_error": rel_err,
            "mc_ms": mc_ms,
            "bs_ms": bs_ms,
        }
    ]
)
st.download_button("Download current results (CSV)", data=_csv_download(export_df), file_name="current_run.csv", mime="text/csv")


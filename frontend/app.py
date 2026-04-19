"""
Blue Lotus Labs — Streamlit Frontend
Client-facing web UI for the stress-testing engine.

Run locally:  streamlit run frontend/app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import time
import io
import os

# ── Config ───────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

BL_DARK  = "#0D1B2A"
BL_NAVY  = "#0D1B2A"
BL_NAVY  = "#0D1B2A"
BL_BLUE  = "#1B4F72"
BL_TEAL  = "#148F77"
BL_GOLD  = "#D4AC0D"
BL_ROSE  = "#C0392B"
BL_LIGHT = "#EAF2FF"

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Blue Lotus Labs",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {BL_DARK}; color: {BL_LIGHT}; }}
    .main-header {{
        background: linear-gradient(135deg, {BL_NAVY} 0%, {BL_BLUE} 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        border: 1px solid {BL_GOLD};
    }}
    .metric-card {{
        background: #111E2D; padding: 1rem; border-radius: 8px;
        border-left: 3px solid {BL_GOLD}; margin-bottom: 0.5rem;
    }}
    .status-badge {{
        display: inline-block; padding: 0.2rem 0.8rem;
        border-radius: 20px; font-size: 0.8rem; font-weight: bold;
    }}
    .badge-completed {{ background: {BL_TEAL}; color: white; }}
    .badge-running   {{ background: {BL_GOLD}; color: black; }}
    .badge-failed    {{ background: {BL_ROSE}; color: white; }}
    .badge-pending   {{ background: #555; color: white; }}
    div[data-testid="metric-container"] {{
        background: #111E2D; border: 1px solid #2E4057;
        border-radius: 8px; padding: 0.5rem;
    }}
    .stButton > button {{
        background: {BL_BLUE}; color: white; border: 1px solid {BL_GOLD};
        border-radius: 6px; font-weight: bold;
    }}
    .stButton > button:hover {{ background: {BL_GOLD}; color: black; }}
</style>
""", unsafe_allow_html=True)




# ── Session state ─────────────────────────────────────────────────
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None


# ── API helpers ───────────────────────────────────────────────────
def api(method, path, **kwargs):
    headers = kwargs.pop("headers", {})
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    try:
        r = getattr(requests, method)(f"{API_URL}{path}", headers=headers, **kwargs)
        return r
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is the backend running?")
        return None


def poll_run(run_id, placeholder, max_wait=120):
    """Poll until run completes or times out."""
    for i in range(max_wait):
        r = api("get", f"/run/{run_id}")
        if r is None:
            return None
        data = r.json()
        status = data.get("status", "pending")
        placeholder.info(f"⏳ Running... ({i+1}s) — status: {status}")
        if status == "completed":
            placeholder.empty()
            return data
        if status == "failed":
            placeholder.error(f"❌ Run failed: {data.get('error_msg', 'Unknown error')}")
            return None
        time.sleep(1)
    placeholder.error("⏱ Timed out waiting for results.")
    return None


# ── Auth pages ────────────────────────────────────────────────────
def page_login():
    st.markdown('<div class="main-header">'
                '<h1 style="color:#D4AC0D;text-align:center;margin:0">🌸 Blue Lotus Labs</h1>'
                '<p style="color:#EAF2FF;text-align:center;margin:0.5rem 0 0">Stress-Testing Engine</p>'
                '</div>', unsafe_allow_html=True)

    tab_login, tab_register = st.tabs(["Login", "Create Account"])

    with tab_login:
        with st.form("login_form"):
            email    = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit   = st.form_submit_button("Login", use_container_width=True)
        if submit:
            r = api("post", "/auth/login",
                    data={"username": email, "password": password})
            if r and r.status_code == 200:
                d = r.json()
                st.session_state.token = d["access_token"]
                st.session_state.user  = d
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab_register:
        with st.form("register_form"):
            name     = st.text_input("Name (optional)")
            email    = st.text_input("Email")
            password = st.text_input("Password (min 8 chars)", type="password")
            submit   = st.form_submit_button("Create Account", use_container_width=True)
        if submit:
            r = api("post", "/auth/register",
                    json={"email": email, "password": password, "name": name or None})
            if r and r.status_code == 200:
                d = r.json()
                st.session_state.token = d["access_token"]
                st.session_state.user  = d
                st.rerun()
            else:
                st.error(r.json().get("detail", "Registration failed.") if r else "Error")


# ── Main app ──────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown(f'<h2 style="color:{BL_GOLD}">🌸 Blue Lotus Labs</h2>', unsafe_allow_html=True)
        user = st.session_state.user
        if user:
            st.markdown(f"**{user.get('email', '')}**")
            st.markdown(f"Plan: `{user.get('plan', 'free').upper()}`")
        st.divider()
        page = st.radio("Navigate", ["🚀 New Run", "📊 Run History", "⚖️ Compare", "🔑 API Keys"],
                        label_visibility="collapsed")
        st.divider()
        if st.button("Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.user  = None
            st.rerun()
    return page


def page_new_run():
    st.markdown("## 🚀 New Stress Test")

    mode = st.radio("Data source", ["Ticker (Yahoo Finance)", "Upload CSV", "Paste Returns"],
                    horizontal=True)

    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown("**Simulation settings**")
        n_paths  = st.select_slider("Paths", [1000, 2000, 5000, 10000, 50000], value=10000)
        horizon  = st.slider("Horizon (days)", 21, 504, 252)
        run_sens = st.checkbox("Fragility Index", value=True)

    with col1:
        if mode == "Ticker (Yahoo Finance)":
            ticker     = st.text_input("Ticker symbol", placeholder="SPY, QQQ, BTC-USD...")
            start_date = st.date_input("From", value=pd.Timestamp("2010-01-01"))
            name       = st.text_input("Strategy name (optional)")

            if st.button("▶ Run Stress Test", use_container_width=True, type="primary"):
                if not ticker:
                    st.warning("Enter a ticker symbol.")
                    return
                placeholder = st.empty()
                r = api("post", "/run/ticker", json={
                    "ticker": ticker.upper(),
                    "start_date": str(start_date),
                    "n_paths": n_paths,
                    "horizon": horizon,
                    "strategy_name": name or None,
                    "run_sensitivity": run_sens,
                })
                if r and r.status_code == 200:
                    run_id = r.json()["run_id"]
                    st.session_state["last_run_id"] = run_id
                    data = poll_run(run_id, placeholder)
                    if data and data.get("result"):
                        render_results(data)
                else:
                    st.error(r.json().get("detail", "Failed to start run.") if r else "API error")

        elif mode == "Upload CSV":
            uploaded = st.file_uploader("CSV with a 'returns' column (decimal, e.g. 0.012)",
                                        type=["csv"])
            name = st.text_input("Strategy name", value="My Strategy")
            if uploaded and st.button("▶ Run", use_container_width=True, type="primary"):
                df = pd.read_csv(uploaded)
                col = st.selectbox("Select returns column", df.columns.tolist())
                returns = df[col].dropna().tolist()
                placeholder = st.empty()
                r = api("post", "/run/custom", json={
                    "returns": returns, "strategy_name": name,
                    "n_paths": n_paths, "horizon": horizon,
                    "run_sensitivity": run_sens,
                })
                if r and r.status_code == 200:
                    run_id = r.json()["run_id"]
                    data   = poll_run(run_id, placeholder)
                    if data and data.get("result"):
                        render_results(data)

        else:  # Paste
            raw  = st.text_area("Paste comma-separated decimal returns",
                                placeholder="0.012, -0.005, 0.003, ...")
            name = st.text_input("Strategy name", value="My Strategy")
            if st.button("▶ Run", use_container_width=True, type="primary"):
                try:
                    returns = [float(x.strip()) for x in raw.split(",") if x.strip()]
                    placeholder = st.empty()
                    r = api("post", "/run/custom", json={
                        "returns": returns, "strategy_name": name,
                        "n_paths": n_paths, "horizon": horizon,
                        "run_sensitivity": run_sens,
                    })
                    if r and r.status_code == 200:
                        run_id = r.json()["run_id"]
                        data   = poll_run(run_id, placeholder)
                        if data and data.get("result"):
                            render_results(data)
                except ValueError:
                    st.error("Could not parse returns. Use comma-separated decimals.")


def render_results(data: dict):
    res    = data.get("result", {})
    dd     = res.get("drawdown", {})
    es     = res.get("expected_shortfall", {})
    rec    = res.get("recovery", {})
    frag   = res.get("fragility", {})
    sim    = res.get("simulation", {})
    regime = res.get("regime", {})

    st.success(f"✅ Completed in {data.get('duration_sec', 0):.1f}s")
    st.markdown("---")
    st.markdown("### Risk Summary")

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mean Max Drawdown", f"{dd.get('mean', 0):.4f}")
    c2.metric("5th Pct Drawdown",  f"{dd.get('p5', 0):.4f}", delta="Severe tail")
    c3.metric("Aggregate ES (5%)", f"{es.get('aggregate', 0):.4f}")
    c4.metric("Never Recover",     f"{rec.get('pct_never', 0):.1%}")
    c5.metric("Fragility Index",   f"{frag.get('index', 0):.4f}" if frag.get('index') else "N/A",
              delta=frag.get("grade", ""))

    # Scenario breakdown
    sc = sim.get("scenario_counts", {})
    st.markdown("**Scenario Distribution**")
    sc_df = pd.DataFrame([{
        "Scenario": k.title(),
        "Paths": v,
        "Share": f"{v/max(sim.get('n_paths',1),1):.1%}"
    } for k, v in sc.items()])
    st.dataframe(sc_df, hide_index=True, use_container_width=True)

    # Regime
    dist = regime.get("stationary_dist", {})
    st.markdown("**Regime Stationary Distribution**")
    r1, r2, r3 = st.columns(3)
    r1.metric("Calm",     f"{dist.get('calm', 0):.1%}")
    r2.metric("Volatile", f"{dist.get('volatile', 0):.1%}")
    r3.metric("Crisis",   f"{dist.get('crisis', 0):.1%}")

    # Histograms from payload
    dd_hist  = dd.get("histogram", [])
    es_hist  = es.get("histogram", [])
    rec_hist = rec.get("histogram", [])

    if dd_hist or es_hist:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.patch.set_facecolor(BL_DARK)
        for ax in axes:
            ax.set_facecolor("#111E2D")
            ax.tick_params(colors="#CBD5E0")
            ax.spines[:].set_color("#2E4057")

        if dd_hist:
            xs = [p["x"] for p in dd_hist]; ys = [p["y"] for p in dd_hist]
            axes[0].bar(xs, ys, width=(xs[1]-xs[0]) if len(xs)>1 else 0.001,
                        color=BL_BLUE, alpha=0.85)
            axes[0].axvline(dd.get("mean", 0),   color=BL_GOLD, lw=1.5, label="Mean")
            axes[0].axvline(dd.get("p5", 0),     color=BL_ROSE, lw=1.5, linestyle="--", label="5th pct")
            axes[0].set_title("Max Drawdown", color=BL_LIGHT)
            axes[0].legend(fontsize=7, framealpha=0.2)

        if es_hist:
            xs = [p["x"] for p in es_hist]; ys = [p["y"] for p in es_hist]
            axes[1].bar(xs, ys, width=(xs[1]-xs[0]) if len(xs)>1 else 0.001,
                        color=BL_ROSE, alpha=0.85)
            axes[1].axvline(es.get("aggregate", 0), color=BL_GOLD, lw=1.5, label="Agg ES")
            axes[1].set_title("Expected Shortfall", color=BL_LIGHT)
            axes[1].legend(fontsize=7, framealpha=0.2)

        if rec_hist:
            xs = [p["x"] for p in rec_hist]; ys = [p["y"] for p in rec_hist]
            axes[2].bar(xs, ys, width=(xs[1]-xs[0]) if len(xs)>1 else 1,
                        color=BL_TEAL, alpha=0.85)
            if rec.get("mean"):
                axes[2].axvline(rec["mean"], color=BL_GOLD, lw=1.5, label=f"Mean: {rec['mean']:.0f}d")
            axes[2].set_title("Time-to-Recovery (days)", color=BL_LIGHT)
            axes[2].legend(fontsize=7, framealpha=0.2)

        plt.tight_layout()
        st.pyplot(fig)

    # PDF download
    st.markdown("---")
    if st.button("📄 Download PDF Report"):
        try:
            from reports.pdf import generate_pdf
            pdf_bytes = generate_pdf(
                result=res,
                strategy_name=data.get("strategy_name", "Strategy"),
                ticker=data.get("ticker"),
                run_id=data.get("run_id"),
            )
            st.download_button(
                "⬇ Download PDF",
                data=pdf_bytes,
                file_name=f"bluelotus_{data.get('ticker','strategy')}_{data.get('run_id','')[:8]}.pdf",
                mime="application/pdf",
            )
        except ImportError:
            st.warning("Install reportlab: pip install reportlab")


def page_history():
    st.markdown("## 📊 Run History")
    r = api("get", "/runs", params={"page": 1, "page_size": 50})
    if r is None or r.status_code != 200:
        st.error("Failed to load runs.")
        return
    data = r.json()
    runs = data.get("runs", [])
    if not runs:
        st.info("No runs yet. Go to 🚀 New Run to get started.")
        return

    df = pd.DataFrame([{
        "Ticker":    r.get("ticker", "Custom"),
        "Strategy":  r.get("strategy_name", "—"),
        "Status":    r.get("status", "—"),
        "N Obs":     r.get("n_observations", "—"),
        "Mean DD":   f"{r['dd_mean']:.4f}" if r.get("dd_mean") else "—",
        "ES (5%)":   f"{r['es_aggregate']:.4f}" if r.get("es_aggregate") else "—",
        "Fragility": f"{r['fragility_index']:.3f} ({r['fragility_grade']})"
                     if r.get("fragility_index") else "—",
        "Run ID":    r["run_id"][:8] + "...",
        "Date":      r["created_at"][:10],
    } for r in runs])

    st.dataframe(df, hide_index=True, use_container_width=True)
    st.markdown(f"*Showing {len(runs)} of {data.get('total', 0)} runs*")


def page_compare():
    st.markdown("## ⚖️ Compare Tickers")
    tickers_raw = st.text_input("Tickers (comma-separated)", placeholder="SPY, QQQ, TLT, GLD")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=pd.Timestamp("2015-01-01"))
    with col2:
        n_paths = st.select_slider("Paths per ticker", [1000, 2000, 5000], value=2000)

    if st.button("▶ Compare", type="primary", use_container_width=True):
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        if len(tickers) < 2:
            st.warning("Enter at least 2 tickers.")
            return
        with st.spinner("Running comparison..."):
            r = api("post", "/compare", json={
                "tickers": tickers,
                "start_date": str(start_date),
                "n_paths": n_paths,
                "horizon": 252,
            })
        if r is None or r.status_code != 200:
            st.error("Comparison failed.")
            return
        rows = r.json().get("rows", [])
        if not rows:
            st.error("No results returned.")
            return
        df = pd.DataFrame([{
            "Ticker":         row["ticker"],
            "N Obs":          row["n_observations"],
            "Ann Vol":        f"{row.get('ann_vol', 0):.2%}" if row.get("ann_vol") else "—",
            "Mean DD":        f"{row.get('dd_mean', 0):.4f}" if row.get("dd_mean") else "—",
            "ES (5%)":        f"{row.get('es_aggregate', 0):.4f}" if row.get("es_aggregate") else "—",
            "Never Recover":  f"{row.get('pct_never_recover', 0):.1%}" if row.get("pct_never_recover") is not None else "—",
            "Med Recovery":   f"{row.get('recovery_median', 0):.0f}d" if row.get("recovery_median") else "—",
        } for row in rows])
        st.dataframe(df, hide_index=True, use_container_width=True)


def page_api_keys():
    st.markdown("## 🔑 API Keys")
    st.markdown("Use API keys to call the Blue Lotus API programmatically.")

    name = st.text_input("Key name (optional)", placeholder="production, backtest-script...")
    if st.button("Generate New API Key", type="primary"):
        r = api("post", "/auth/api-keys", params={"name": name or None})
        if r and r.status_code == 200:
            key = r.json()["key"]
            st.success("API key created — save this now, it won't be shown again.")
            st.code(key)
        else:
            st.error("Failed to create key.")

    st.markdown("---")
    st.markdown("**Your API keys**")
    r = api("get", "/auth/api-keys")
    if r and r.status_code == 200:
        keys = r.json()
        if not keys:
            st.info("No API keys yet.")
        for k in keys:
            st.markdown(f"- **{k.get('name', 'Unnamed')}** — ID: `{k['key_id'][:12]}...` "
                        f"— Created: {k['created_at'][:10]}")

    st.markdown("---")
    st.markdown("**Example API usage:**")
    st.code("""
import requests

API_KEY = "bl_your_key_here"
BASE    = "https://your-app.railway.app"

# Run stress test on SPY
r = requests.post(f"{BASE}/run/ticker",
    headers={"X-API-Key": API_KEY},
    json={"ticker": "SPY", "n_paths": 10000, "horizon": 252}
)
run_id = r.json()["run_id"]

# Poll for result
import time
while True:
    r = requests.get(f"{BASE}/run/{run_id}",
        headers={"X-API-Key": API_KEY})
    if r.json()["status"] == "completed":
        result = r.json()["result"]
        break
    time.sleep(2)

print("Mean Drawdown:", result["drawdown"]["mean"])
print("ES (5%):",       result["expected_shortfall"]["aggregate"])
""", language="python")


# ── Router ────────────────────────────────────────────────────────
def main():
    if not st.session_state.token:
        page_login()
        return

    page = sidebar()

    if   "New Run"  in page: page_new_run()
    elif "History"  in page: page_history()
    elif "Compare"  in page: page_compare()
    elif "API Keys" in page: page_api_keys()


if __name__ == "__main__":
    main()

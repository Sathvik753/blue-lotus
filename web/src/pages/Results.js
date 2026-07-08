import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, API_BASE } from "../utils/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid,
} from "recharts";
import {
  ArrowLeft, Download, FileText, TrendingDown, Shield, Clock,
  AlertTriangle, CheckCircle, XCircle, ChevronDown, ChevronUp,
} from "lucide-react";

const GOLD = "#D4AC0D";
const TEAL = "#148F77";
const ROSE = "#C0392B";
const BLUE = "#1B4F72";
const MUTED = "#5D7A99";


const pct = (v, d = 2) =>
  v == null || isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`;

const num = (v, d = 4) =>
  v == null || isNaN(v) ? "—" : v.toFixed(d);

const scaleHist = (data) =>
  data?.map(d => ({ ...d, x: parseFloat((d.x * 100).toFixed(4)) })) ?? [];

const scaleRef = (r) => ({ ...r, value: r.value != null ? r.value * 100 : r.value });

// Bootstrap CI formatted as "90% CI: [X, Y]"
function CiBadge({ lo, hi, isDay = false }) {
  if (lo == null || hi == null || isNaN(lo) || isNaN(hi)) return null;
  const fmt = isDay ? v => `${v.toFixed(0)}d` : v => pct(v);
  return (
    <div style={{ fontSize: 10, color: MUTED, marginTop: 5, fontFamily: "DM Mono, monospace" }}>
      90% CI: [{fmt(lo)}, {fmt(hi)}]
    </div>
  );
}


function Collapsible({ title, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: "100%", background: "none", border: "none", cursor: "pointer",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          padding: 0, color: "var(--light)",
        }}
      >
        <div className="section-title" style={{ fontSize: 14, marginBottom: 0 }}>{title}</div>
        {open ? <ChevronUp size={14} color={MUTED} /> : <ChevronDown size={14} color={MUTED} />}
      </button>
      {open && <div style={{ marginTop: 16 }}>{children}</div>}
    </div>
  );
}


function ChartTooltip({ active, payload, label, isPercent }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0D1B2A", border: "1px solid #1E3048",
      borderRadius: 6, padding: "8px 12px", fontSize: 12,
    }}>
      <div style={{ color: MUTED, marginBottom: 2 }}>
        {isPercent ? `${label?.toFixed(2)}%` : label?.toFixed(2)}
      </div>
      <div style={{ color: GOLD, fontWeight: 600 }}>
        Density: {payload[0]?.value?.toFixed(4)}
      </div>
    </div>
  );
}


function HistChart({ data, color, refLines = [], title, subtitle, isPercent = false }) {
  if (!data?.length) return <div style={{ color: MUTED, fontSize: 12 }}>No data</div>;
  return (
    <div>
      <div style={{ fontSize: 12, color: MUTED, marginBottom: 2, letterSpacing: "0.06em", textTransform: "uppercase" }}>
        {title}
      </div>
      {subtitle && (
        <div style={{ fontSize: 11, color: MUTED, marginBottom: 8, opacity: 0.7 }}>{subtitle}</div>
      )}
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1E3048" vertical={false} />
          <XAxis dataKey="x" tick={{ fill: MUTED, fontSize: 9 }}
            tickFormatter={v => isPercent ? `${v.toFixed(1)}%` : v.toFixed(2)}
            interval="preserveStartEnd" />
          <YAxis tick={{ fill: MUTED, fontSize: 9 }} width={32} />
          <Tooltip content={<ChartTooltip isPercent={isPercent} />} />
          {refLines.map((r, i) => (
            <ReferenceLine key={i} x={r.value} stroke={r.color} strokeWidth={1.5}
              strokeDasharray={r.dash ? "4 4" : "0"}
              label={{ value: r.label, fill: r.color, fontSize: 9 }} />
          ))}
          <Bar dataKey="y" fill={color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}


function StatRow({ label, value, description, ci }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: MUTED, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 2 }}>{label}</div>
      <div style={{ fontFamily: "DM Mono, monospace", fontSize: 15, color: "var(--white)", marginBottom: 2 }}>{value}</div>
      {ci && <CiBadge lo={ci.lo} hi={ci.hi} />}
      {description && <div style={{ fontSize: 11, color: MUTED, opacity: 0.8, marginTop: 4 }}>{description}</div>}
    </div>
  );
}


function SectionHeader({ title, description }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <div className="section-title" style={{ fontSize: 14, marginBottom: 4 }}>{title}</div>
      {description && <div style={{ fontSize: 12, color: MUTED, lineHeight: 1.5 }}>{description}</div>}
    </div>
  );
}


function WassersteinRow({ name, value, max }) {
  const labels = {
    "mean_+10pct":  "Avg returns +10%",
    "mean_-10pct":  "Avg returns −10%",
    "vol_+20pct":   "Volatility +20%",
    "vol_-20pct":   "Volatility −20%",
    "trans_±15pct": "Regime transitions ±15%",
  };
  const w = max > 0 ? (value / max) * 100 : 0;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3, fontSize: 11 }}>
        <span style={{ color: "var(--muted)" }}>{labels[name] || name}</span>
        <span style={{ fontFamily: "DM Mono, monospace", color: GOLD, fontSize: 11 }}>
          W₁ = {value?.toFixed(6)}
        </span>
      </div>
      <div style={{ height: 4, background: "var(--border)", borderRadius: 2 }}>
        <div style={{ height: "100%", width: `${w}%`, background: GOLD, borderRadius: 2, opacity: 0.7, transition: "width 0.5s" }} />
      </div>
    </div>
  );
}


function BacktestRow({ br }) {
  const covered = br.dd_covered;
  return (
    <div style={{
      display: "grid", gridTemplateColumns: "1fr auto auto auto auto",
      gap: 12, alignItems: "center", padding: "12px 0",
      borderBottom: "1px solid var(--border)", fontSize: 12,
    }}>
      <div>
        <div style={{ color: "var(--light)", fontWeight: 500, marginBottom: 2 }}>{br.period_label}</div>
        <div style={{ color: MUTED, fontSize: 11 }}>{br.n_observations} trading days</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div style={{ fontSize: 10, color: MUTED, marginBottom: 2 }}>Realized DD</div>
        <div style={{ fontFamily: "DM Mono, monospace", color: ROSE }}>{pct(br.realized_max_dd)}</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div style={{ fontSize: 10, color: MUTED, marginBottom: 2 }}>Predicted P5</div>
        <div style={{ fontFamily: "DM Mono, monospace", color: GOLD }}>{pct(br.predicted_dd_p5)}</div>
      </div>
      <div style={{ textAlign: "right" }}>
        <div style={{ fontSize: 10, color: MUTED, marginBottom: 2 }}>Ratio</div>
        <div style={{ fontFamily: "DM Mono, monospace", color: br.coverage_ratio > 1 ? ROSE : TEAL }}>
          {br.coverage_ratio?.toFixed(2)}×
        </div>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        {covered
          ? <CheckCircle size={16} color={TEAL} />
          : <XCircle size={16} color={ROSE} />}
        <span style={{ fontSize: 11, color: covered ? TEAL : ROSE }}>
          {covered ? "Covered" : "Missed"}
        </span>
      </div>
    </div>
  );
}


export default function Results() {
  const { runId }  = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [exporting, setExporting] = useState("");

  async function handleExport(kind) {
    if (!data) return;
    setExporting(kind);
    try {
      const path = kind === "pdf" ? `/run/${runId}/pdf` : `/run/${runId}/export`;
      await api.download(path, `bluelotus_${runId.slice(0, 8)}.${kind}`);
    } catch (e) {
      setError(`Export failed: ${e.message}`);
    } finally {
      setExporting("");
    }
  }

  useEffect(() => {
    async function load() {
      const res = await api.get(`/run/${runId}`);
      if (!res?.ok)                        { setError("Could not load results."); setLoading(false); return; }
      if (res.data.status === "failed")    { setError(res.data.error_msg || "Run failed."); setLoading(false); return; }
      if (res.data.status !== "completed") { setError("Run not completed yet."); setLoading(false); return; }
      setData(res.data);
      setLoading(false);
    }
    load();
  }, [runId]);

  if (loading) return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, color: "var(--muted)", paddingTop: 60 }}>
      <span className="spinner" /> Loading results...
    </div>
  );

  if (error) return (
    <div className="fade-in">
      <button onClick={() => navigate(-1)} className="btn btn-secondary"
        style={{ marginBottom: 24, display: "flex", alignItems: "center", gap: 8 }}>
        <ArrowLeft size={14} /> Back
      </button>
      <div style={{ color: "var(--rose)", background: "rgba(192,57,43,0.1)", border: "1px solid rgba(192,57,43,0.3)", borderRadius: 8, padding: 16 }}>
        {error}
      </div>
    </div>
  );

  const r = data.result;
  const dd = r.drawdown;
  const es = r.expected_shortfall;
  const rec = r.recovery;
  const frag = r.fragility;
  const sim = r.simulation;
  const regime = r.regime?.stationary_dist || {};
  const hmm = r.regime?.fit_quality || {};
  const gpd = r.tail_constraints?.gpd || {};
  const tail = r.tail_constraints || {};
  const sc = sim?.scenario_counts || {};
  const backtest = r.backtest || [];
  const ddBci = dd?.bootstrap_ci || {};
  const esBci = es?.bootstrap_ci || {};
  const recBci = rec?.bootstrap_ci || {};

  const fragGrade = frag?.grade || "—";
  const fragColor = fragGrade === "Robust" ? TEAL
    : fragGrade === "Moderate" ? GOLD
    : fragGrade === "Fragile" ? ROSE
    : MUTED;
  const fragDesc = fragGrade === "Robust"
    ? "Estimates stay consistent when model inputs are nudged."
    : fragGrade === "Moderate"
    ? "Some sensitivity to input parameters."
    : fragGrade === "Fragile"
    ? "Risk estimates shift noticeably with small input changes. Interpret with caution."
    : "The stability diagnostic did not complete for this run.";

  // Percentage shares that are guaranteed to sum to exactly 100.0 —
  // independent rounding of each share can drift to 100.1 / 99.9.
  const shareTotal = (sc.normal || 0) + (sc.stress || 0) + (sc.crisis || 0);
  const rawShares = ["normal", "stress", "crisis"].map(k =>
    shareTotal > 0 ? ((sc[k] || 0) / shareTotal) * 1000 : 0);
  const floored = rawShares.map(Math.floor);
  let leftover = 1000 - floored.reduce((a, b) => a + b, 0);
  const order = rawShares.map((v, i) => [v - floored[i], i]).sort((a, b) => b[0] - a[0]);
  for (let j = 0; j < order.length && leftover > 0; j++, leftover--) floored[order[j][1]] += 1;
  const shares = {
    normal: (floored[0] / 10).toFixed(1),
    stress: (floored[1] / 10).toFixed(1),
    crisis: (floored[2] / 10).toFixed(1),
  };

  const wassDetails = frag?.details || {};
  const wassMax = Object.values(wassDetails).length > 0
    ? Math.max(...Object.values(wassDetails).filter(v => v != null))
    : 0;

  return (
    <div className="fade-in">

      {/* ── Header ── */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 32 }}>
        <div>
          <button onClick={() => navigate(-1)} className="btn btn-secondary"
            style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 8, fontSize: 11 }}>
            <ArrowLeft size={12} /> Back
          </button>
          <div className="accent-line" />
          <h1 style={{ fontSize: 30 }} className="gradient-text">{data.strategy_name || data.ticker || "Results"}</h1>
          <p style={{ color: "var(--muted)", fontSize: 12, marginTop: 4 }}>
            {sim?.n_paths?.toLocaleString()} simulated scenarios &nbsp;·&nbsp;
            {sim?.horizon}-day forward horizon &nbsp;·&nbsp;
            {data.completed_at ? new Date(data.completed_at).toLocaleString() : ""}
          </p>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button
            onClick={() => handleExport("json")}
            disabled={!!exporting}
            className="btn btn-secondary"
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            <Download size={14} /> {exporting === "json" ? "Downloading…" : "JSON"}
          </button>
          <button
            onClick={() => handleExport("pdf")}
            disabled={!!exporting}
            className="btn btn-primary"
            style={{ display: "flex", alignItems: "center", gap: 8 }}
          >
            <FileText size={14} /> {exporting === "pdf" ? "Building…" : "PDF Report"}
          </button>
        </div>
      </div>

      {/* ── Key metric cards ── */}
      <div className="grid-4" style={{ marginBottom: 24 }}>

        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <TrendingDown size={11} /> Typical Peak-to-Trough Loss
          </div>
          <div className="metric-value" style={{ color: ROSE }}>{pct(dd?.mean)}</div>
          <CiBadge lo={ddBci?.mean?.lo} hi={ddBci?.mean?.hi} />
          <span className="metric-badge badge-red" style={{ marginTop: 8 }}>
            Worst 5%: {pct(dd?.p5)}
          </span>
          <div style={{ fontSize: 11, color: MUTED, marginTop: 6, lineHeight: 1.4 }}>
            How far the strategy typically falls from its peak.
          </div>
        </div>

        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <AlertTriangle size={11} /> Average Tail Loss (CVaR)
          </div>
          <div className="metric-value" style={{ color: GOLD }}>{pct(es?.aggregate)}</div>
          <CiBadge lo={esBci?.aggregate?.lo} hi={esBci?.aggregate?.hi} />
          <span className="metric-badge badge-gold" style={{ marginTop: 8 }}>
            Worst {(es?.alpha * 100).toFixed(0)}% of days
          </span>
          <div style={{ fontSize: 11, color: MUTED, marginTop: 6, lineHeight: 1.4 }}>
            Average daily return on the worst simulated days.
          </div>
        </div>

        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Clock size={11} /> Recovery Time
          </div>
          <div className="metric-value">
            {rec?.mean ? `${Math.round(rec.mean)} days` : "—"}
          </div>
          <CiBadge lo={recBci?.lo} hi={recBci?.hi} isDay={true} />
          <span className="metric-badge badge-gold" style={{ marginTop: 8 }}>
            {(rec?.pct_never * 100).toFixed(1)}% still below peak at day {sim?.horizon}
          </span>
          <div style={{ fontSize: 11, color: MUTED, marginTop: 6, lineHeight: 1.4 }}>
            Time to reclaim the previous high, among paths that drew down.
            Paths still underwater when the simulation ends are cut off by the
            horizon — not predicted to be permanent losses.
          </div>
        </div>

        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Shield size={11} /> Model Stability (MFI)
          </div>
          <div className="metric-value" style={{ color: fragColor }}>
            {frag?.index != null ? frag.index.toFixed(4) : "—"}
          </div>
          <span className="metric-badge" style={{ background: `${fragColor}22`, color: fragColor, marginTop: 8 }}>
            {fragGrade}
          </span>
          <div style={{ fontSize: 11, color: MUTED, marginTop: 6, lineHeight: 1.4 }}>
            {fragDesc}
          </div>
        </div>

      </div>

      {/* ── Scenarios + Regime ── */}
      <div className="grid-2" style={{ marginBottom: 24 }}>

        <div className="card">
          <SectionHeader
            title="Simulation Outcomes"
            description="How the simulated paths split across calm, stressed, and crisis conditions."
          />
          <table>
            <thead>
              <tr>
                <th>Outcome</th><th>Paths</th><th>Share</th>
                <th style={{ color: MUTED, fontWeight: 400 }}>What it means</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Normal", sc.normal, shares.normal, TEAL, "Drawdown stays within a typical range"],
                ["Stress", sc.stress, shares.stress, GOLD, "Moderate drawdown — elevated but manageable"],
                ["Crisis", sc.crisis, shares.crisis, ROSE, "Severe drawdown — significant loss scenario"],
              ].map(([name, count, share, color, desc]) => (
                <tr key={name}>
                  <td><span style={{ color, fontWeight: 600 }}>● </span>{name}</td>
                  <td className="mono">{count?.toLocaleString() || 0}</td>
                  <td className="mono">{shareTotal > 0 ? share + "%" : "—"}</td>
                  <td style={{ fontSize: 11, color: MUTED }}>{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <SectionHeader
            title="Market Regime Breakdown"
            description="Long-run time spent in each market state, classified from rolling realized volatility in the historical data."
          />
          <div style={{ display: "flex", gap: 12, flexDirection: "column" }}>
            {[
              ["Calm",     regime.calm,     TEAL, "Low volatility, steady returns"],
              ["Volatile", regime.volatile, GOLD, "Elevated volatility, larger swings"],
              ["Crisis",   regime.crisis,   ROSE, "Severe stress, large drawdowns"],
            ].map(([name, val, color, desc]) => (
              <div key={name}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12 }}>
                  <div>
                    <span style={{ color: "var(--light)" }}>{name}</span>
                    <span style={{ color: MUTED, fontSize: 11, marginLeft: 8 }}>{desc}</span>
                  </div>
                  <span style={{ color, fontFamily: "DM Mono, monospace", fontWeight: 600 }}>
                    {val ? (val * 100).toFixed(1) + "%" : "—"}
                  </span>
                </div>
                <div style={{ height: 5, background: "var(--border)", borderRadius: 3 }}>
                  <div style={{ height: "100%", width: `${(val || 0) * 100}%`, background: color, borderRadius: 3, transition: "width 0.5s" }} />
                </div>
              </div>
            ))}
          </div>
          {(hmm?.aic != null || hmm?.bic != null) && (
            <div style={{ marginTop: 16, paddingTop: 12, borderTop: "1px solid var(--border)", display: "flex", gap: 20 }}>
              {[["AIC", hmm.aic], ["BIC", hmm.bic], ["Log-likelihood", hmm.log_likelihood]].map(([label, val]) => (
                <div key={label}>
                  <div style={{ fontSize: 10, color: MUTED, marginBottom: 2 }}>{label}</div>
                  <div style={{ fontFamily: "DM Mono, monospace", fontSize: 11, color: "var(--white)" }}>
                    {val != null ? val.toFixed(0) : "—"}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

      </div>

      {/* ── Charts ── */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <div className="card">
          <HistChart
            title="Max Drawdown Distribution"
            subtitle="Spread of worst losses across all simulated scenarios"
            data={scaleHist(dd?.histogram)} color={BLUE} isPercent={true}
            refLines={[
              scaleRef({ value: dd?.mean, color: GOLD, label: "Avg" }),
              scaleRef({ value: dd?.p5,   color: ROSE, dash: true, label: "5th" }),
            ]}
          />
        </div>
        <div className="card">
          <HistChart
            title="Tail Loss Distribution (CVaR)"
            subtitle="Daily loss on the worst trading days simulated"
            data={scaleHist(es?.histogram)} color={ROSE} isPercent={true}
            refLines={[scaleRef({ value: es?.aggregate, color: GOLD, label: "Avg" })]}
          />
        </div>
        <div className="card">
          <HistChart
            title="Time to Recover (days)"
            subtitle="How long before the strategy gets back to its previous high"
            data={rec?.histogram} color={TEAL} isPercent={false}
            refLines={rec?.mean ? [{ value: rec.mean, color: GOLD, label: "Avg" }] : []}
          />
        </div>
      </div>

      {/* ── Drawdown details ── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <SectionHeader
          title="Drawdown Details"
          description="Full breakdown of peak-to-trough losses. Bootstrap confidence intervals show how stable each estimate is across repeated simulations."
        />
        <div className="grid-4">
          <StatRow label="Mean Max Drawdown" value={pct(dd?.mean)}
            ci={ddBci?.mean} description="Average worst loss across all scenarios" />
          <StatRow label="Median Max Drawdown" value={pct(dd?.median)}
            description="Half of scenarios have a worse loss than this" />
          <StatRow label="5th Percentile" value={pct(dd?.p5)}
            ci={ddBci?.p5} description="Only 5% of scenarios produce a worse loss" />
          <StatRow label="90% Confidence Interval"
            value={`[${pct(dd?.ci_90_low, 1)}, ${pct(dd?.ci_90_high, 1)}]`}
            description="Range that captures 90% of all outcomes" />
        </div>
        {dd?.by_scenario && (
          <div style={{ marginTop: 20, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
            <div style={{ fontSize: 11, color: MUTED, marginBottom: 12, textTransform: "uppercase", letterSpacing: "0.06em" }}>
              Average Loss by Scenario Type
            </div>
            <div style={{ display: "flex", gap: 24 }}>
              {[["Normal", dd.by_scenario.normal, TEAL], ["Stress", dd.by_scenario.stress, GOLD], ["Crisis", dd.by_scenario.crisis, ROSE]].map(([name, val, color]) => (
                <div key={name}>
                  <span style={{ color, fontWeight: 600, marginRight: 6 }}>●</span>
                  <span style={{ color: MUTED, fontSize: 12, marginRight: 4 }}>{name}:</span>
                  <span style={{ fontFamily: "DM Mono, monospace", fontSize: 12, color: "var(--white)" }}>
                    {val != null && !isNaN(val) ? pct(val) : "—"}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── ES details ── */}
      <div className="card" style={{ marginBottom: 16 }}>
        <SectionHeader
          title="Tail Loss Details (Expected Shortfall / CVaR)"
          description="Expected Shortfall is the average loss on the worst days — a stricter measure than VaR because it considers the full tail."
        />
        <div className="grid-4">
          <StatRow label="Aggregate ES" value={pct(es?.aggregate)}
            ci={esBci?.aggregate} description="Average return across all tail scenarios combined" />
          <StatRow label="Mean Per-Scenario ES" value={pct(es?.mean)}
            ci={esBci?.mean} description="Average of each scenario's own tail loss" />
          <StatRow label="90% CI Lower" value={pct(es?.ci_90_low)}
            description="Lower bound of the 90% confidence range" />
          <StatRow label="90% CI Upper" value={pct(es?.ci_90_high)}
            description="Upper bound of the 90% confidence range" />
        </div>
      </div>

      {/* ── Model Stability details (Wasserstein) ── */}
      {Object.keys(wassDetails).length > 0 && (
        <Collapsible title="Model Stability — Sensitivity Breakdown">
          <p style={{ fontSize: 12, color: MUTED, marginBottom: 16, lineHeight: 1.6 }}>
            The Model Fragility Index measures how much the drawdown distribution shifts when
            key model parameters are nudged. Each bar shows the Wasserstein distance (W₁)
            between the baseline and perturbed distribution — a larger bar means that
            parameter has more influence on the results.
          </p>
          {Object.entries(wassDetails).map(([name, val]) => (
            <WassersteinRow key={name} name={name} value={val} max={wassMax} />
          ))}
          <div style={{ marginTop: 12, fontSize: 11, color: MUTED }}>
            MFI = average W₁ / σ(baseline drawdown) = {frag?.index != null ? frag.index.toFixed(4) : "unavailable"} ({fragGrade})
          </div>
        </Collapsible>
      )}

      {/* ── Historical backtest ── */}
      {backtest.length > 0 && (
        <Collapsible title="Historical Backtest Validation" defaultOpen={true}>
          <p style={{ fontSize: 12, color: MUTED, marginBottom: 16, lineHeight: 1.6 }}>
            How did the engine's predictions hold up against real market crashes?
            "Covered" means the realized loss fell within the predicted range.
            A ratio below 1× means the model was conservative (predicted a worse outcome than what happened).
          </p>
          {backtest.map((br, i) => <BacktestRow key={i} br={br} />)}
          <div style={{ marginTop: 12, fontSize: 11, color: MUTED, fontStyle: "italic" }}>
            Note: this is an in-sample validation — the model was fitted on the full history
            including these periods. Out-of-sample testing would require separate model fits.
          </div>
        </Collapsible>
      )}

      {/* ── Technical model details ── */}
      <Collapsible title="Technical Model Details">
        <div className="grid-3" style={{ gap: 24 }}>
          <div>
            <div style={{ fontSize: 11, color: MUTED, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
              Tail Model (GPD)
            </div>
            {[
              ["Method",    tail.method === "gpd_evt" ? "GPD / EVT" : "Empirical fallback"],
              ["Shape ξ",   num(gpd.xi, 4)],
              ["Scale β",   num(gpd.beta, 6)],
              ["Threshold", pct(gpd.threshold)],
              ["Tail obs",  gpd.n_tail != null ? gpd.n_tail.toString() : "—"],
            ].map(([label, val]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 12 }}>
                <span style={{ color: MUTED }}>{label}</span>
                <span style={{ fontFamily: "DM Mono, monospace", color: "var(--white)" }}>{val}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{ fontSize: 11, color: MUTED, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
              Regime Model
            </div>
            {[
              ["Type",           "3-state RV-Percentile"],
              ["Estimation",     "Deterministic (auditable)"],
              ["Calm threshold",  "< 70th vol percentile"],
              ["Crisis threshold","≥ 90th vol + neg. trend"],
              ["Min crisis run",  "5 consecutive days"],
            ].map(([label, val]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 12 }}>
                <span style={{ color: MUTED }}>{label}</span>
                <span style={{ fontFamily: "DM Mono, monospace", color: "var(--white)", fontSize: 11 }}>{val}</span>
              </div>
            ))}
          </div>
          <div>
            <div style={{ fontSize: 11, color: MUTED, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 10 }}>
              Simulation
            </div>
            {[
              ["Paths",          sim?.n_paths?.toLocaleString()],
              ["Horizon",        `${sim?.horizon} days`],
              ["Rejection rate", pct(sim?.rejection_rate, 1)],
              ["Esscher λ₁",     num(sim?.esscher_lam1, 4)],
              ["Esscher λ₂",     num(sim?.esscher_lam2, 4)],
            ].map(([label, val]) => (
              <div key={label} style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 12 }}>
                <span style={{ color: MUTED }}>{label}</span>
                <span style={{ fontFamily: "DM Mono, monospace", color: "var(--white)" }}>{val || "—"}</span>
              </div>
            ))}
          </div>
        </div>
      </Collapsible>

      {/* ── Disclaimer ── */}
      <div style={{
        marginTop: 8, padding: "12px 16px",
        background: "rgba(212,172,13,0.05)", border: "1px solid rgba(212,172,13,0.15)",
        borderRadius: 8, fontSize: 11, color: MUTED, lineHeight: 1.6,
      }}>
        <strong style={{ color: GOLD }}>⚠ Important:</strong> These results are estimated risk
        distributions based on historical patterns and statistical modelling. They are not
        predictions of future performance. This output is for informational purposes only
        and does not constitute investment advice.
      </div>

    </div>
  );
}

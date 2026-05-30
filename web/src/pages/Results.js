import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { api, API_BASE } from "../utils/api";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, CartesianGrid,
} from "recharts";
import { ArrowLeft, Download, TrendingDown, Shield, Clock, AlertTriangle } from "lucide-react";

const GOLD = "#D4AC0D";
const TEAL = "#148F77";
const ROSE = "#C0392B";
const BLUE = "#1B4F72";
const MUTED = "#5D7A99";

function CustomTooltip({ active, payload, label, fmt }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#0D1B2A", border: "1px solid #1E3048",
      borderRadius: 6, padding: "8px 12px", fontSize: 12,
    }}>
      <div style={{ color: MUTED, marginBottom: 2 }}>{fmt ? fmt(label) : label}</div>
      <div style={{ color: GOLD, fontWeight: 600 }}>{payload[0]?.value?.toFixed(4)}</div>
    </div>
  );
}

function HistChart({ data, color, refLines = [], title, xLabel }) {
  if (!data?.length) return <div style={{ color: MUTED, fontSize: 12 }}>No data</div>;
  return (
    <div>
      <div style={{ fontSize: 12, color: MUTED, marginBottom: 8, letterSpacing: "0.06em", textTransform: "uppercase" }}>
        {title}
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 4, right: 4, bottom: 4, left: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1E3048" vertical={false} />
          <XAxis dataKey="x" tick={{ fill: MUTED, fontSize: 9 }}
            tickFormatter={v => v.toFixed(2)} interval="preserveStartEnd" />
          <YAxis tick={{ fill: MUTED, fontSize: 9 }} width={32} />
          <Tooltip content={<CustomTooltip />} />
          {refLines.map((r, i) => (
            <ReferenceLine key={i} x={r.value} stroke={r.color} strokeWidth={1.5}
              strokeDasharray={r.dash ? "4 4" : "0"} label={{ value: r.label, fill: r.color, fontSize: 9 }} />
          ))}
          <Bar dataKey="y" fill={color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Results() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      const res = await api.get(`/run/${runId}`);
      if (!res?.ok) { setError("Could not load results."); setLoading(false); return; }
      if (res.data.status === "failed") { setError(res.data.error_msg || "Run failed."); setLoading(false); return; }
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
      <button onClick={() => navigate(-1)} className="btn btn-secondary" style={{ marginBottom: 24, display: "flex", alignItems: "center", gap: 8 }}>
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
  const sc = sim?.scenario_counts || {};

  const fragGrade = frag?.grade || "—";
  const fragColor = fragGrade === "Robust" ? TEAL : fragGrade === "Moderate" ? GOLD : ROSE;

  return (
    <div className="fade-in">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 32 }}>
        <div>
          <button onClick={() => navigate(-1)} className="btn btn-secondary"
            style={{ marginBottom: 16, display: "flex", alignItems: "center", gap: 8, fontSize: 11 }}>
            <ArrowLeft size={12} /> Back
          </button>
          <div className="accent-line" />
          <h1 style={{ fontSize: 28 }}>{data.strategy_name || data.ticker || "Results"}</h1>
          <p style={{ color: "var(--muted)", fontSize: 12, marginTop: 4 }}>
            {sim?.n_paths?.toLocaleString()} paths · {sim?.horizon} day horizon ·{" "}
            {data.completed_at ? new Date(data.completed_at).toLocaleString() : ""}
          </p>
        </div>
        <a
          href={`${API_BASE}/run/${runId}`}
          target="_blank" rel="noreferrer"
          className="btn btn-secondary"
          style={{ display: "flex", alignItems: "center", gap: 8 }}
        >
          <Download size={14} /> Export JSON
        </a>
      </div>

      {/* Key metrics */}
      <div className="grid-4" style={{ marginBottom: 24 }}>
        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <TrendingDown size={11} /> Mean Max Drawdown
          </div>
          <div className="metric-value" style={{ color: ROSE }}>{dd?.mean?.toFixed(4)}</div>
          <span className="metric-badge badge-red">5th pct: {dd?.p5?.toFixed(4)}</span>
        </div>
        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <AlertTriangle size={11} /> Expected Shortfall
          </div>
          <div className="metric-value" style={{ color: GOLD }}>{es?.aggregate?.toFixed(4)}</div>
          <span className="metric-badge badge-gold">α = {(es?.alpha * 100).toFixed(0)}%</span>
        </div>
        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Clock size={11} /> Recovery
          </div>
          <div className="metric-value">{rec?.mean ? `${rec.mean.toFixed(0)}d` : "—"}</div>
          <span className="metric-badge badge-red">{(rec?.pct_never * 100).toFixed(1)}% never recover</span>
        </div>
        <div className="metric">
          <div className="metric-label" style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Shield size={11} /> Fragility Index™
          </div>
          <div className="metric-value" style={{ color: fragColor }}>{frag?.index?.toFixed(4) || "—"}</div>
          <span className="metric-badge" style={{ background: `${fragColor}22`, color: fragColor }}>
            {fragGrade}
          </span>
        </div>
      </div>

      {/* Scenario + Regime */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="section-title" style={{ fontSize: 14 }}>Scenario Distribution</div>
          <table>
            <thead><tr><th>Scenario</th><th>Paths</th><th>Share</th></tr></thead>
            <tbody>
              {[["Normal", sc.normal, TEAL], ["Stress", sc.stress, GOLD], ["Crisis", sc.crisis, ROSE]].map(([name, count, color]) => (
                <tr key={name}>
                  <td><span style={{ color, fontWeight: 600 }}>● </span>{name}</td>
                  <td className="mono">{count?.toLocaleString() || 0}</td>
                  <td className="mono">{count && sim?.n_paths ? ((count / sim.n_paths) * 100).toFixed(1) + "%" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <div className="section-title" style={{ fontSize: 14 }}>Regime Stationary Distribution</div>
          <div style={{ display: "flex", gap: 8, flexDirection: "column" }}>
            {[["Calm", regime.calm, TEAL], ["Volatile", regime.volatile, GOLD], ["Crisis", regime.crisis, ROSE]].map(([name, val, color]) => (
              <div key={name}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12 }}>
                  <span style={{ color: "var(--muted)" }}>{name}</span>
                  <span style={{ color, fontFamily: "DM Mono, monospace", fontWeight: 600 }}>
                    {val ? (val * 100).toFixed(1) + "%" : "—"}
                  </span>
                </div>
                <div style={{ height: 4, background: "var(--border)", borderRadius: 2 }}>
                  <div style={{ height: "100%", width: `${(val || 0) * 100}%`, background: color, borderRadius: 2 }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <div className="card">
          <HistChart
            title="Max Drawdown Distribution"
            data={dd?.histogram}
            color={BLUE}
            refLines={[
              { value: dd?.mean, color: GOLD, label: "Mean" },
              { value: dd?.p5, color: ROSE, dash: true, label: "5th" },
            ]}
          />
        </div>
        <div className="card">
          <HistChart
            title="Expected Shortfall"
            data={es?.histogram}
            color={ROSE}
            refLines={[
              { value: es?.aggregate, color: GOLD, label: "Agg" },
            ]}
          />
        </div>
        <div className="card">
          <HistChart
            title="Time-to-Recovery (days)"
            data={rec?.histogram}
            color={TEAL}
            refLines={rec?.mean ? [{ value: rec.mean, color: GOLD, label: "Mean" }] : []}
          />
        </div>
      </div>

      {/* Drawdown CI */}
      <div className="card">
        <div className="section-title" style={{ fontSize: 14, marginBottom: 16 }}>Drawdown Details</div>
        <div className="grid-4">
          {[
            ["Mean", dd?.mean?.toFixed(6)],
            ["Median", dd?.median?.toFixed(6)],
            ["5th Percentile", dd?.p5?.toFixed(6)],
            ["90% CI", `[${dd?.ci_90_low?.toFixed(4)}, ${dd?.ci_90_high?.toFixed(4)}]`],
          ].map(([label, val]) => (
            <div key={label}>
              <div style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 4 }}>{label}</div>
              <div style={{ fontFamily: "DM Mono, monospace", fontSize: 13, color: "var(--white)" }}>{val}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 16, fontSize: 11, color: "var(--muted)", textAlign: "center" }}>
        ⚠ This report estimates risk distributions only. No return predictions are made or implied.
      </div>
    </div>
  );
}

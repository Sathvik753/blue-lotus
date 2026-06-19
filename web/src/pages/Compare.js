import React, { useState } from "react";
import { api } from "../utils/api";
import { Plus, X } from "lucide-react";

export default function Compare() {
  const [tickers, setTickers] = useState(["SPY", "QQQ"]);
  const [input, setInput] = useState("");
  const [startDate, setStartDate] = useState("2015-01-01");
  const [nPaths, setNPaths] = useState(2000);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function addTicker() {
    const t = input.trim().toUpperCase();
    if (t && !tickers.includes(t) && tickers.length < 8) {
      setTickers([...tickers, t]);
      setInput("");
    }
  }

  async function run() {
    setError(""); setLoading(true); setResults(null);
    try {
      const res = await api.post("/compare", {
        tickers, start_date: startDate, n_paths: nPaths, horizon: 252,
      });
      if (res?.ok) {
        setResults(res.data.rows);
      } else {
        const detail = res?.data?.detail;
        setError(Array.isArray(detail) ? detail.map(e => e.msg).join("; ") : (detail || "Comparison failed."));
      }
    } catch {
      setError("Network error. Please try again.");
    }
    setLoading(false);
  }

  const COLS = [
    { key: "ticker", label: "Ticker" },
    { key: "n_observations", label: "N Obs" },
    { key: "ann_vol", label: "Ann Vol", fmt: v => v != null ? (v * 100).toFixed(2) + "%" : "—" },
    { key: "dd_mean", label: "Mean DD", fmt: v => v?.toFixed(4) },
    { key: "es_aggregate", label: "ES (5%)", fmt: v => v?.toFixed(4) },
    { key: "pct_never_recover", label: "No Recovery", fmt: v => v != null ? (v * 100).toFixed(1) + "%" : "—" },
    { key: "recovery_median", label: "Med Recovery", fmt: v => v != null ? v.toFixed(0) + "d" : "—" },
  ];

  return (
    <div className="fade-in">
      <div className="accent-line" />
      <h1 style={{ fontSize: 34, marginBottom: 8 }} className="gradient-text">Compare</h1>
      <p style={{ color: "var(--muted)", marginBottom: 32 }}>
        Run the engine across multiple tickers and rank their risk side by side.
      </p>

      <div className="card" style={{ marginBottom: 16 }}>
        <div style={{ marginBottom: 16 }}>
          <label>Tickers</label>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBottom: 10 }}>
            {tickers.map(t => (
              <div key={t} style={{
                display: "flex", alignItems: "center", gap: 6,
                background: "var(--dark)", border: "1px solid var(--border)",
                borderRadius: 6, padding: "4px 10px", fontSize: 13, fontWeight: 600,
              }}>
                {t}
                <X size={12} style={{ cursor: "pointer", color: "var(--muted)" }}
                  onClick={() => setTickers(tickers.filter(x => x !== t))} />
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              value={input} onChange={e => setInput(e.target.value.toUpperCase())}
              onKeyDown={e => e.key === "Enter" && addTicker()}
              placeholder="Add ticker (e.g. TLT, GLD...)"
              style={{ flex: 1 }}
            />
            <button type="button" onClick={addTicker} className="btn btn-secondary"
              style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <Plus size={14} /> Add
            </button>
          </div>
        </div>

        <div className="grid-2">
          <div>
            <label>Start Date</label>
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
          </div>
          <div>
            <label>Paths per ticker: {nPaths.toLocaleString()}</label>
            <input
              type="range" min={1000} max={5000} step={500}
              value={nPaths} onChange={e => setNPaths(+e.target.value)}
              style={{ padding: 0, background: "transparent", border: "none" }}
            />
          </div>
        </div>
      </div>

      {error && (
        <div style={{
          background: "rgba(192,57,43,0.1)", border: "1px solid rgba(192,57,43,0.3)",
          borderRadius: 8, padding: "12px 16px", color: "var(--rose)", fontSize: 13, marginBottom: 16,
        }}>{error}</div>
      )}

      <button
        onClick={run} disabled={loading || tickers.length < 2}
        className="btn btn-primary" style={{ marginBottom: 24, padding: "12px 32px" }}
      >
        {loading ? <><span className="spinner" style={{ marginRight: 8 }} />Comparing...</> : "▶  Run Comparison"}
      </button>

      {results && (
        <div className="card fade-in">
          <table>
            <thead>
              <tr>{COLS.map(c => <th key={c.key}>{c.label}</th>)}</tr>
            </thead>
            <tbody>
              {results.map(row => (
                <tr key={row.ticker}>
                  {COLS.map(c => (
                    <td key={c.key} className={c.key === "ticker" ? "mono" : ""} style={{
                      fontWeight: c.key === "ticker" ? 700 : 400,
                      color: c.key === "ticker" ? "var(--gold)" : "var(--white)",
                    }}>
                      {c.fmt ? c.fmt(row[c.key]) : (row[c.key] ?? "—")}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

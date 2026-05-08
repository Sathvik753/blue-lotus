import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../utils/api";
import { TrendingUp, AlignLeft, Upload, FileText, X, Play } from "lucide-react";

// ── CSV parser ────────────────────────────────────────────────────

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) throw new Error("CSV must have a header row and at least one data row.");

  const headers = lines[0].split(",").map(h => h.trim().toLowerCase());
  const col = headers.indexOf("return");
  if (col === -1) throw new Error('CSV must contain a "return" column.');

  const returns = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(",");
    const raw = cells[col]?.trim();
    if (!raw) continue;
    const val = parseFloat(raw);
    if (isNaN(val)) throw new Error(`Row ${i + 1}: "${raw}" is not a number.`);
    returns.push(val);
  }

  if (returns.length < 30) throw new Error("Need at least 30 return observations.");
  return returns;
}

// ── Paste parser ──────────────────────────────────────────────────

function parsePaste(text) {
  const tokens = text.trim().split(/[\s,]+/).filter(Boolean);
  if (tokens.length === 0) return null;
  const returns = tokens.map((v, i) => {
    const n = parseFloat(v);
    if (isNaN(n)) throw new Error(`Value ${i + 1}: "${v}" is not a number.`);
    return n;
  });
  if (returns.length < 30) throw new Error(`Need at least 30 observations — got ${returns.length}.`);
  return returns;
}

// ── Shared sub-components ─────────────────────────────────────────

function ErrorBox({ msg }) {
  if (!msg) return null;
  return (
    <div style={{
      background: "rgba(192,57,43,0.1)", border: "1px solid rgba(192,57,43,0.3)",
      borderRadius: 8, padding: "10px 14px", color: "var(--rose)", fontSize: 13,
    }}>
      {msg}
    </div>
  );
}

function formatApiError(data) {
  if (!data) return "Request failed.";
  const d = data.detail;
  if (Array.isArray(d)) return d.map(e => e.msg).join("; ");
  if (typeof d === "string") return d;
  return "Something went wrong.";
}

// ── Main component ────────────────────────────────────────────────

const MODES = [
  { id: "ticker", label: "Yahoo Finance", icon: TrendingUp },
  { id: "paste",  label: "Paste Returns", icon: AlignLeft  },
  { id: "csv",    label: "Upload CSV",    icon: Upload      },
];

export default function NewRun() {
  const navigate = useNavigate();
  const fileRef  = useRef();

  const [mode, setMode] = useState("ticker");

  // Ticker mode
  const [ticker,    setTicker]    = useState("");
  const [startDate, setStartDate] = useState("2010-01-01");

  // Paste mode
  const [pasteText,    setPasteText]    = useState("");
  const [pasteReturns, setPasteReturns] = useState(null);
  const [pasteError,   setPasteError]   = useState("");

  // CSV mode
  const [fileName,   setFileName]   = useState("");
  const [csvReturns, setCsvReturns] = useState(null);
  const [parseError, setParseError] = useState("");
  const [dragging,   setDragging]   = useState(false);

  // Shared params
  const [strategyName, setStrategyName] = useState("");
  const [nPaths,       setNPaths]       = useState("1000");
  const [horizon,      setHorizon]      = useState("252");

  // Run state
  const [running, setRunning] = useState(false);
  const [status,  setStatus]  = useState("");
  const [error,   setError]   = useState("");

  // ── Paste handling ──────────────────────────────────────────────

  function onPasteChange(text) {
    setPasteText(text);
    setPasteError("");
    setPasteReturns(null);
    if (!text.trim()) return;
    try {
      setPasteReturns(parsePaste(text));
    } catch (err) {
      setPasteError(err.message);
    }
  }

  // ── CSV handling ────────────────────────────────────────────────

  function handleFile(file) {
    if (!file) return;
    setParseError("");
    setCsvReturns(null);
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const parsed = parseCSV(e.target.result);
        setFileName(file.name);
        setCsvReturns(parsed);
      } catch (err) {
        setParseError(err.message);
        setFileName("");
      }
    };
    reader.readAsText(file);
  }

  function onFileInput(e) {
    handleFile(e.target.files[0]);
    e.target.value = "";
  }

  function onDrop(e) {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }

  function clearFile() {
    setFileName("");
    setCsvReturns(null);
    setParseError("");
  }

  // ── Submit ──────────────────────────────────────────────────────

  async function submit(e) {
    e.preventDefault();
    setError("");
    setStatus("Submitting…");
    setRunning(true);

    try {
      let res;

      if (mode === "ticker") {
        res = await api.post("/run/ticker", {
          ticker:        ticker.trim().toUpperCase(),
          start_date:    startDate,
          strategy_name: strategyName.trim() || undefined,
          n_paths:       parseInt(nPaths, 10)  || 1000,
          horizon:       parseInt(horizon, 10) || 252,
        });
      } else {
        const returns = mode === "csv" ? csvReturns : pasteReturns;
        res = await api.post("/run/custom", {
          returns,
          strategy_name: strategyName.trim() || "Custom",
          n_paths:       parseInt(nPaths, 10)  || 1000,
          horizon:       parseInt(horizon, 10) || 252,
        });
      }

      if (!res?.ok) throw new Error(formatApiError(res?.data));

      const runId = res.data.run_id;
      setStatus("Running simulation…");

      while (true) {
        await new Promise(r => setTimeout(r, 2000));
        const poll = await api.get(`/run/${runId}`);
        if (!poll) throw new Error("Lost connection while polling.");
        const s = poll.data.status;
        setStatus(`Running simulation… (${s})`);
        if (s === "completed") break;
        if (s === "failed") throw new Error(poll.data.error_msg || "Run failed.");
      }

      navigate(`/results/${runId}`);
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setRunning(false);
      setStatus("");
    }
  }

  const canSubmit = !running && (
    (mode === "ticker" && ticker.trim() !== "") ||
    (mode === "paste"  && pasteReturns !== null) ||
    (mode === "csv"    && csvReturns   !== null)
  );

  // ── Render ──────────────────────────────────────────────────────

  return (
    <div className="fade-in">
      <div className="accent-line" />
      <h1 style={{ fontSize: 32, marginBottom: 8 }}>New Stress Test</h1>
      <p style={{ color: "var(--muted)", marginBottom: 32 }}>
        Run a Monte Carlo stress test on any return series.
      </p>

      <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 24 }}>

        {/* ── Mode selector ── */}
        <div style={{
          display: "flex", background: "var(--dark)", borderRadius: 10,
          padding: 4, gap: 4,
        }}>
          {MODES.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => { setMode(id); setError(""); }}
              style={{
                flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
                gap: 8, padding: "10px 12px", borderRadius: 7, border: "none",
                background: mode === id ? "var(--card)" : "transparent",
                color: mode === id ? "var(--gold)" : "var(--muted)",
                fontFamily: "Syne, sans-serif", fontWeight: 600,
                fontSize: 12, letterSpacing: "0.05em", textTransform: "uppercase",
                cursor: "pointer", transition: "all 0.15s",
                boxShadow: mode === id ? "0 1px 3px rgba(0,0,0,0.3)" : "none",
              }}
            >
              <Icon size={13} />
              {label}
            </button>
          ))}
        </div>

        {/* ── Data source panel ── */}
        <div className="card">

          {/* Yahoo Finance */}
          {mode === "ticker" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div className="section-title" style={{ fontSize: 16, marginBottom: 0 }}>
                Ticker Lookup
              </div>
              <p style={{ color: "var(--muted)", fontSize: 13, marginTop: -8 }}>
                Fetches daily close prices from Yahoo Finance and computes returns automatically.
              </p>
              <div className="grid-2" style={{ gap: 16 }}>
                <div>
                  <label>Ticker Symbol</label>
                  <input
                    placeholder="SPY, AAPL, BTC-USD…"
                    value={ticker}
                    onChange={e => setTicker(e.target.value.toUpperCase())}
                    style={{ fontFamily: "DM Mono, monospace", letterSpacing: "0.05em" }}
                  />
                </div>
                <div>
                  <label>History Start Date</label>
                  <input
                    type="date"
                    value={startDate}
                    onChange={e => setStartDate(e.target.value)}
                  />
                </div>
              </div>
            </div>
          )}

          {/* Paste Returns */}
          {mode === "paste" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div className="section-title" style={{ fontSize: 16, marginBottom: 0 }}>
                Paste Return Series
              </div>
              <p style={{ color: "var(--muted)", fontSize: 13, marginTop: -4 }}>
                Paste decimal daily returns separated by commas or newlines
                (e.g. <code style={{ color: "var(--gold)" }}>0.012, -0.005, 0.003…</code>).
              </p>
              <textarea
                rows={6}
                placeholder={"0.012, -0.005, 0.003, 0.008, -0.011…"}
                value={pasteText}
                onChange={e => onPasteChange(e.target.value)}
                style={{
                  resize: "vertical", fontFamily: "DM Mono, monospace",
                  fontSize: 12, lineHeight: 1.7,
                }}
              />
              {pasteReturns && (
                <div style={{
                  display: "flex", alignItems: "center", gap: 8,
                  color: "var(--teal)", fontSize: 12,
                }}>
                  <FileText size={13} />
                  {pasteReturns.length.toLocaleString()} observations · min{" "}
                  {Math.min(...pasteReturns).toFixed(4)} · max{" "}
                  {Math.max(...pasteReturns).toFixed(4)}
                </div>
              )}
              <ErrorBox msg={pasteError} />
            </div>
          )}

          {/* Upload CSV */}
          {mode === "csv" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              <div className="section-title" style={{ fontSize: 16, marginBottom: 0 }}>
                Upload CSV
              </div>
              <p style={{ color: "var(--muted)", fontSize: 13, marginTop: -4 }}>
                Upload a CSV with a <code style={{ color: "var(--gold)" }}>return</code> column
                containing decimal daily returns.
              </p>

              {!csvReturns ? (
                <div
                  onClick={() => fileRef.current.click()}
                  onDragOver={e => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={onDrop}
                  style={{
                    border: `2px dashed ${dragging ? "var(--gold)" : "var(--border)"}`,
                    borderRadius: 10, padding: "36px 24px", textAlign: "center",
                    cursor: "pointer", transition: "border-color 0.2s",
                    background: dragging ? "rgba(212,172,13,0.04)" : "transparent",
                  }}
                >
                  <Upload size={28} color="var(--muted)" style={{ marginBottom: 10 }} />
                  <div style={{ color: "var(--light)", fontWeight: 500, marginBottom: 4 }}>
                    Drop a CSV file here, or click to browse
                  </div>
                  <div style={{ color: "var(--muted)", fontSize: 12 }}>
                    Must include a <code style={{ color: "var(--gold)" }}>return</code> column · ≥ 30 rows
                  </div>
                  <input ref={fileRef} type="file" accept=".csv,text/csv"
                    onChange={onFileInput} style={{ display: "none" }} />
                </div>
              ) : (
                <div style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  background: "var(--dark)", borderRadius: 8, padding: "12px 16px",
                  border: "1px solid var(--border)",
                }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                    <FileText size={18} color="var(--teal)" />
                    <div>
                      <div style={{ fontWeight: 500, fontSize: 13 }}>{fileName}</div>
                      <div style={{ color: "var(--muted)", fontSize: 11 }}>
                        {csvReturns.length.toLocaleString()} observations parsed ·{" "}
                        min {Math.min(...csvReturns).toFixed(4)} · max {Math.max(...csvReturns).toFixed(4)}
                      </div>
                    </div>
                  </div>
                  <button type="button" onClick={clearFile} style={{
                    background: "none", border: "none", cursor: "pointer",
                    color: "var(--muted)", padding: 4,
                  }}>
                    <X size={16} />
                  </button>
                </div>
              )}

              <ErrorBox msg={parseError} />
            </div>
          )}
        </div>

        {/* ── Parameters ── */}
        <div className="card">
          <div className="section-title" style={{ fontSize: 16, marginBottom: 16 }}>Parameters</div>
          <div className="grid-3" style={{ gap: 20 }}>
            <div>
              <label>Strategy Name</label>
              <input
                placeholder={mode === "ticker" ? ticker || "My Strategy" : "My Portfolio"}
                value={strategyName}
                onChange={e => setStrategyName(e.target.value)}
              />
            </div>
            <div>
              <label>Simulation Paths</label>
              <input
                type="number" min={1000} max={10000}
                value={nPaths}
                onChange={e => setNPaths(e.target.value)}
              />
            </div>
            <div>
              <label>Horizon (days)</label>
              <input
                type="number" min={21} max={1260}
                value={horizon}
                onChange={e => setHorizon(e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* ── Error ── */}
        <ErrorBox msg={error} />

        {/* ── Status ── */}
        {running && status && (
          <div style={{ display: "flex", alignItems: "center", gap: 12, color: "var(--gold)", fontSize: 13 }}>
            <span className="spinner" /> {status}
          </div>
        )}

        <button
          type="submit"
          className="btn btn-primary"
          disabled={!canSubmit}
          style={{ alignSelf: "flex-start", display: "flex", alignItems: "center", gap: 8, padding: "12px 28px" }}
        >
          <Play size={14} /> Run Stress Test
        </button>
      </form>
    </div>
  );
}

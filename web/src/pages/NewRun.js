import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../utils/api";
import { Upload, FileText, X, Play } from "lucide-react";

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

export default function NewRun() {
  const navigate = useNavigate();
  const fileRef = useRef();

  const [fileName, setFileName] = useState("");
  const [returns, setReturns] = useState(null);
  const [parseError, setParseError] = useState("");
  const [dragging, setDragging] = useState(false);

  const [strategyName, setStrategyName] = useState("");
  const [nPaths, setNPaths] = useState("1000");
  const [horizon, setHorizon] = useState("252");

  const [status, setStatus] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  function handleFile(file) {
    if (!file) return;
    setParseError("");
    setReturns(null);
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const parsed = parseCSV(e.target.result);
        setFileName(file.name);
        setReturns(parsed);
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
    setReturns(null);
    setParseError("");
  }

  async function submit(e) {
    e.preventDefault();
    if (!returns) return;
    setError("");
    setStatus("Submitting…");
    setRunning(true);

    try {
      const body = {
        returns,
        strategy_name: strategyName.trim() || "Custom",
        n_paths: parseInt(nPaths, 10) || 1000,
        horizon: parseInt(horizon, 10) || 252,
      };

      const res = await api.post("/run/custom", body);
      if (!res?.ok) {
        const detail = res?.data?.detail;
        const msg = Array.isArray(detail)
          ? detail.map(e => e.msg).join("; ")
          : (typeof detail === "string" ? detail : "Failed to start run.");
        throw new Error(msg);
      }

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

  const canSubmit = returns && !running;

  return (
    <div className="fade-in">
      <div className="accent-line" />
      <h1 style={{ fontSize: 32, marginBottom: 8 }}>New Stress Test</h1>
      <p style={{ color: "var(--muted)", marginBottom: 32 }}>
        Upload a CSV with a <code style={{ color: "var(--gold)" }}>return</code> column to run a Monte Carlo stress test.
      </p>

      <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 24 }}>

        {/* CSV Upload */}
        <div className="card">
          <div className="section-title" style={{ fontSize: 16, marginBottom: 16 }}>Upload Returns</div>

          {!returns ? (
            <div
              onClick={() => fileRef.current.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              style={{
                border: `2px dashed ${dragging ? "var(--gold)" : "var(--border)"}`,
                borderRadius: 10,
                padding: "40px 24px",
                textAlign: "center",
                cursor: "pointer",
                transition: "border-color 0.2s",
                background: dragging ? "rgba(212,172,13,0.04)" : "transparent",
              }}
            >
              <Upload size={32} color="var(--muted)" style={{ marginBottom: 12 }} />
              <div style={{ color: "var(--light)", fontWeight: 500, marginBottom: 6 }}>
                Drop a CSV file here, or click to browse
              </div>
              <div style={{ color: "var(--muted)", fontSize: 12 }}>
                Must include a <code style={{ color: "var(--gold)" }}>return</code> column · e.g. daily log-returns
              </div>
              <input ref={fileRef} type="file" accept=".csv,text/csv" onChange={onFileInput}
                style={{ display: "none" }} />
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
                    {returns.length.toLocaleString()} observations parsed ·{" "}
                    min {Math.min(...returns).toFixed(4)} · max {Math.max(...returns).toFixed(4)}
                  </div>
                </div>
              </div>
              <button type="button" onClick={clearFile} style={{
                background: "none", border: "none", cursor: "pointer", color: "var(--muted)",
                padding: 4,
              }}>
                <X size={16} />
              </button>
            </div>
          )}

          {parseError && (
            <div style={{
              marginTop: 12, background: "rgba(192,57,43,0.1)",
              border: "1px solid rgba(192,57,43,0.3)", borderRadius: 8,
              padding: "10px 14px", color: "var(--rose)", fontSize: 13,
            }}>
              {parseError}
            </div>
          )}
        </div>

        {/* Parameters */}
        <div className="card">
          <div className="section-title" style={{ fontSize: 16, marginBottom: 16 }}>Parameters</div>
          <div className="grid-3" style={{ gap: 20 }}>
            <div>
              <label>Strategy Name</label>
              <input
                placeholder="My Portfolio"
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
                type="number" min={1} max={1260}
                value={horizon}
                onChange={e => setHorizon(e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            background: "rgba(192,57,43,0.1)", border: "1px solid rgba(192,57,43,0.3)",
            borderRadius: 8, padding: "12px 16px", color: "var(--rose)", fontSize: 13,
          }}>
            {error}
          </div>
        )}

        {/* Status */}
        {running && status && (
          <div style={{
            display: "flex", alignItems: "center", gap: 12,
            color: "var(--gold)", fontSize: 13,
          }}>
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

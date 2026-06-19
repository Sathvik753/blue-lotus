import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../utils/api";
import { ArrowRight, Inbox, RefreshCw } from "lucide-react";

const pct = (v, d = 2) => (v == null || isNaN(v) ? "—" : `${(v * 100).toFixed(d)}%`);

const gradeColor = (g) =>
  g === "Robust" ? "var(--teal-2)" : g === "Moderate" ? "var(--gold)" : "var(--rose)";

export function History() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  function load() {
    setLoading(true);
    api.get("/runs?page=1&page_size=50").then(res => {
      if (res?.ok) setRuns(res.data.runs || []);
      setLoading(false);
    });
  }

  useEffect(load, []);

  return (
    <div className="fade-in">
      <div className="accent-line" />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 34 }} className="gradient-text">Run History</h1>
          <p style={{ color: "var(--muted)", marginTop: 6 }}>
            Every stress test you've run, newest first.
          </p>
        </div>
        <button onClick={load} className="btn btn-secondary"
          style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      <div className="card">
        {loading ? (
          <div style={{ display: "flex", alignItems: "center", gap: 12, color: "var(--muted)", padding: "20px 0" }}>
            <span className="spinner" /> Loading runs…
          </div>
        ) : runs.length === 0 ? (
          <div style={{ textAlign: "center", color: "var(--muted)", padding: "56px 0" }}>
            <Inbox size={34} style={{ marginBottom: 14, opacity: 0.6 }} />
            <div style={{ marginBottom: 6, color: "var(--light)", fontSize: 15 }}>No runs yet</div>
            <div style={{ marginBottom: 18 }}>Your stress tests will appear here.</div>
            <button onClick={() => navigate("/run")} className="btn btn-primary"
              style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
              Run your first stress test <ArrowRight size={14} />
            </button>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Status</th>
                <th>Typical DD</th>
                <th>Tail Loss</th>
                <th>Stability</th>
                <th>Date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map(run => {
                const done = run.status === "completed";
                return (
                  <tr key={run.run_id} style={{ cursor: done ? "pointer" : "default" }}
                    onClick={() => done && navigate(`/results/${run.run_id}`)}>
                    <td>
                      <div style={{ fontWeight: 600, color: "var(--white)" }}>
                        {run.ticker || run.strategy_name || "Custom"}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--muted)" }}>
                        {run.n_observations ? `${run.n_observations.toLocaleString()} observations` : "—"}
                      </div>
                    </td>
                    <td>
                      <span className={`status-pill status-${run.status}`}>{run.status}</span>
                    </td>
                    <td className="mono" style={{ color: run.dd_mean != null ? "var(--rose)" : "var(--muted)" }}>
                      {pct(run.dd_mean)}
                    </td>
                    <td className="mono" style={{ color: run.es_aggregate != null ? "var(--gold)" : "var(--muted)" }}>
                      {pct(run.es_aggregate)}
                    </td>
                    <td>
                      {run.fragility_grade ? (
                        <span style={{ color: gradeColor(run.fragility_grade), fontSize: 12, fontWeight: 600 }}>
                          {run.fragility_grade}
                        </span>
                      ) : <span style={{ color: "var(--muted)" }}>—</span>}
                    </td>
                    <td style={{ color: "var(--muted)", fontSize: 12 }}>
                      {new Date(run.created_at).toLocaleDateString(undefined,
                        { month: "short", day: "numeric", year: "numeric" })}
                    </td>
                    <td>
                      {done && <ArrowRight size={15} color="var(--muted)" />}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export default History;

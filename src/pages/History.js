// History.js
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../utils/api";
import { ExternalLink } from "lucide-react";

export function History() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get("/runs?page=1&page_size=50").then(res => {
      if (res?.ok) setRuns(res.data.runs || []);
      setLoading(false);
    });
  }, []);

  return (
    <div className="fade-in">
      <div className="accent-line" />
      <h1 style={{ fontSize: 32, marginBottom: 8 }}>Run History</h1>
      <p style={{ color: "var(--muted)", marginBottom: 32 }}>All your past stress test runs.</p>

      <div className="card">
        {loading ? (
          <div style={{ display: "flex", alignItems: "center", gap: 12, color: "var(--muted)" }}>
            <span className="spinner" /> Loading...
          </div>
        ) : runs.length === 0 ? (
          <div style={{ textAlign: "center", color: "var(--muted)", padding: "40px 0" }}>
            No runs yet. <span
              onClick={() => navigate("/run")}
              style={{ color: "var(--gold)", cursor: "pointer" }}
            >Run your first stress test →</span>
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Strategy</th>
                <th>Status</th>
                <th>Mean DD</th>
                <th>ES (5%)</th>
                <th>Fragility</th>
                <th>Date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {runs.map(run => (
                <tr key={run.run_id} style={{ cursor: "pointer" }}
                  onClick={() => run.status === "completed" && navigate(`/results/${run.run_id}`)}>
                  <td>
                    <div style={{ fontWeight: 500 }}>{run.ticker || run.strategy_name || "Custom"}</div>
                    <div style={{ fontSize: 11, color: "var(--muted)" }}>{run.n_observations ? `${run.n_observations} obs` : ""}</div>
                  </td>
                  <td>
                    <span className={`status-${run.status}`} style={{ fontWeight: 600, fontSize: 12 }}>
                      {run.status}
                    </span>
                  </td>
                  <td className="mono">{run.dd_mean != null ? run.dd_mean.toFixed(4) : "—"}</td>
                  <td className="mono">{run.es_aggregate != null ? run.es_aggregate.toFixed(4) : "—"}</td>
                  <td>
                    {run.fragility_grade ? (
                      <span style={{
                        color: run.fragility_grade === "Robust" ? "var(--teal)" : run.fragility_grade === "Moderate" ? "var(--gold)" : "var(--rose)",
                        fontSize: 12, fontWeight: 600,
                      }}>
                        {run.fragility_grade}
                      </span>
                    ) : "—"}
                  </td>
                  <td style={{ color: "var(--muted)", fontSize: 12 }}>
                    {new Date(run.created_at).toLocaleDateString()}
                  </td>
                  <td>
                    {run.status === "completed" && (
                      <ExternalLink size={14} color="var(--muted)" />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

export default History;

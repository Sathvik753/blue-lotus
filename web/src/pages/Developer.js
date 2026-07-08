import React, { useEffect, useState } from "react";
import { api } from "../utils/api";
import { Terminal } from "lucide-react";

function Stat({ label, value }) {
  return (
    <div className="card metric" style={{ padding: "18px 20px" }}>
      <div style={{ fontSize: 12, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 800, marginTop: 6 }}>{value}</div>
    </div>
  );
}

export default function Developer() {
  const [stats, setStats] = useState(null);
  const [orgs, setOrgs] = useState([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    api.get("/dev/stats").then(r => r.ok ? setStats(r.data) : setErr(r.data?.detail || "Failed"));
    api.get("/dev/organizations").then(r => { if (r.ok) setOrgs(r.data); });
  }, []);

  return (
    <div className="fade-in">
      <div className="accent-line" />
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
        <Terminal size={22} color="var(--gold)" />
        <h1 style={{ fontSize: 34 }} className="gradient-text">Developer Console</h1>
      </div>
      <p style={{ color: "var(--muted)", marginBottom: 28 }}>
        Restricted view — visible only to allowlisted developer accounts.
      </p>

      {err && <p style={{ color: "var(--rose)" }}>{err}</p>}

      {stats && (
        <>
          <div className="grid-3" style={{ gap: 16, marginBottom: 16 }}>
            <Stat label="Organizations" value={stats.organizations} />
            <Stat label="Users" value={stats.users} />
            <Stat label="Runs (total)" value={stats.runs_total} />
            <Stat label="Runs (24h)" value={stats.runs_24h} />
            <Stat label="Stripe" value={stats.stripe_enabled ? "Live" : "Mock"} />
            <Stat label="Environment" value={stats.env} />
          </div>

          <div className="card" style={{ marginBottom: 24 }}>
            <div className="section-title" style={{ marginBottom: 12 }}>Runs by status</div>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap" }}>
              {Object.entries(stats.runs_by_status).map(([k, v]) => (
                <div key={k}>
                  <span style={{ color: "var(--muted)", fontSize: 12, textTransform: "capitalize" }}>{k}: </span>
                  <span style={{ fontWeight: 700 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      {orgs.length > 0 && (
        <div className="card">
          <div className="section-title" style={{ marginBottom: 12 }}>Organizations</div>
          <table>
            <thead><tr><th>Name</th><th>Plan</th><th>Status</th><th>Created</th></tr></thead>
            <tbody>
              {orgs.map(o => (
                <tr key={o.id}>
                  <td style={{ fontWeight: 600 }}>{o.name}</td>
                  <td>{o.plan}</td>
                  <td>{o.subscription_status}</td>
                  <td style={{ color: "var(--muted)" }}>{new Date(o.created_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

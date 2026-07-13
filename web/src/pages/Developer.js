import React, { useEffect, useState } from "react";
import { api, setToken } from "../utils/api";
import { useAuth } from "../context/Auth";
import { Terminal, Lock } from "lucide-react";

function Stat({ label, value }) {
  return (
    <div className="card metric" style={{ padding: "18px 20px" }}>
      <div style={{ fontSize: 12, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.08em" }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 800, marginTop: 6 }}>{value}</div>
    </div>
  );
}


function UnlockGate() {
  const { refresh } = useAuth();
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setError(""); setBusy(true);
    const res = await api.post("/auth/developer/unlock", { code: code.trim() });
    setBusy(false);
    if (!res.ok) {
      setError(res.data?.detail || "Unlock failed.");
      return;
    }
    // Fresh token carries the developer claim (rate-limit + quota exempt).
    setToken(res.data.access_token);
    await refresh();
  }

  return (
    <div className="fade-in" style={{ maxWidth: 420, margin: "60px auto" }}>
      <div className="card" style={{ textAlign: "center" }}>
        <Lock size={26} color="var(--gold)" style={{ marginBottom: 12 }} />
        <h1 style={{ fontSize: 20, marginBottom: 6 }}>Developer access</h1>
        <p style={{ color: "var(--muted)", fontSize: 13, marginBottom: 20 }}>
          This area is restricted. Enter the access code to unlock the
          developer console for your account.
        </p>
        <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <input
            type="password" value={code} onChange={e => setCode(e.target.value)}
            placeholder="Access code" autoFocus
            style={{ textAlign: "center", fontFamily: "DM Mono, monospace", letterSpacing: "0.2em" }}
          />
          {error && (
            <div style={{
              background: "rgba(224,86,63,0.1)", border: "1px solid rgba(224,86,63,0.3)",
              borderRadius: 8, padding: "9px 12px", color: "var(--rose)", fontSize: 12.5,
            }}>{error}</div>
          )}
          <button type="submit" className="btn btn-primary" disabled={busy || !code.trim()}>
            {busy ? "Checking…" : "Unlock"}
          </button>
        </form>
      </div>
    </div>
  );
}

export default function Developer() {
  const { user } = useAuth();
  const isDev = !!user?.is_developer;
  const [stats, setStats] = useState(null);
  const [orgs, setOrgs] = useState([]);
  const [err, setErr] = useState("");

  useEffect(() => {
    if (!isDev) return;
    api.get("/dev/stats").then(r => r.ok ? setStats(r.data) : setErr(r.data?.detail || "Failed"));
    api.get("/dev/organizations").then(r => { if (r.ok) setOrgs(r.data); });
  }, [isDev]);

  if (!isDev) return <UnlockGate />;

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

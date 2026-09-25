import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Copy, Check, Trash2, Plus, ExternalLink, Lock } from "lucide-react";
import { api, API_BASE } from "../utils/api";
import { useAuth } from "../context/Auth";

const MONO = "ui-monospace, SF Mono, Menlo, monospace";

function CopyButton({ text, label = "Copy" }) {
  const [done, setDone] = useState(false);
  return (
    <button
      className="btn btn-secondary"
      style={{ padding: "8px 14px", display: "inline-flex", alignItems: "center", gap: 6 }}
      onClick={async () => {
        try { await navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); } catch { /* clipboard blocked */ }
      }}
    >
      {done ? <Check size={14} /> : <Copy size={14} />} {done ? "Copied" : label}
    </button>
  );
}

export default function ApiKeys() {
  const { user } = useAuth();
  const [plan, setPlan] = useState(null);
  const [keys, setKeys] = useState([]);
  const [name, setName] = useState("");
  const [newKey, setNewKey] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const canUseApi = !!user?.is_developer || ["pro", "institutional"].includes(plan);

  async function load() {
    const s = await api.get("/billing/status");
    if (s.ok) setPlan(s.data.plan);
    const k = await api.get("/auth/api-keys");
    if (k.ok && Array.isArray(k.data)) setKeys(k.data);
  }
  useEffect(() => { load(); }, []);

  async function generate() {
    setBusy(true); setErr(""); setNewKey(null);
    const q = name.trim() ? `?name=${encodeURIComponent(name.trim())}` : "";
    const res = await api.post(`/auth/api-keys${q}`);
    setBusy(false);
    if (!res.ok) { setErr(res.data?.detail || "Could not create key."); return; }
    setNewKey(res.data.key);
    setName("");
    load();
  }

  async function revoke(id) {
    const res = await api.delete(`/auth/api-keys/${id}`);
    if (res.ok) setKeys((ks) => ks.filter((k) => k.key_id !== id));
  }

  const curl = `curl -X POST ${API_BASE}/run/ticker \\
  -H "X-API-Key: YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"ticker": "SPY"}'`;

  return (
    <div className="fade-in" style={{ maxWidth: 880 }}>
      <div className="accent-line" />
      <h1 className="gradient-text" style={{ fontSize: 34, marginBottom: 8 }}>API access</h1>
      <p style={{ color: "var(--muted)", marginBottom: 28 }}>
        Run stress tests programmatically — from Python, a cron job, or your trading bot.
      </p>

      {!canUseApi ? (
        <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", gap: 16 }}>
          <div style={{ width: 48, height: 48, borderRadius: 13, display: "grid", placeItems: "center", background: "rgba(242,193,78,0.12)", border: "1px solid rgba(242,193,78,0.3)" }}>
            <Lock size={20} color="var(--gold)" />
          </div>
          <div>
            <div style={{ fontSize: 19, fontWeight: 600, marginBottom: 6 }}>API access is on Algo Pro</div>
            <p style={{ color: "var(--muted)", fontSize: 14, lineHeight: 1.6, maxWidth: 540 }}>
              Generate API keys and call the engine programmatically on the <b style={{ color: "var(--light)" }}>Algo Pro</b> plan
              and above. You're currently on the <b style={{ color: "var(--light)" }}>{plan || "current"}</b> plan.
            </p>
          </div>
          <Link to="/billing" className="btn btn-primary">Upgrade to Algo Pro →</Link>
        </div>
      ) : (
        <>
          {/* Quick start */}
          <div className="card" style={{ marginBottom: 18 }}>
            <div className="section-title">Quick start</div>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
              <div>
                <div style={{ color: "var(--muted)", fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 4 }}>Base URL</div>
                <code style={{ fontFamily: MONO, fontSize: 13, color: "var(--white)" }}>{API_BASE}</code>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <CopyButton text={API_BASE} label="Copy URL" />
                <a href={`${API_BASE}/docs`} target="_blank" rel="noopener noreferrer" className="btn btn-secondary" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  <ExternalLink size={14} /> Full docs
                </a>
              </div>
            </div>
            <pre style={{ background: "rgba(6,11,20,0.7)", border: "1px solid var(--border)", borderRadius: 10, padding: "14px 16px", overflowX: "auto", fontFamily: MONO, fontSize: 12.5, lineHeight: 1.6, color: "var(--light)", margin: 0 }}>{curl}</pre>
          </div>

          {/* One-time reveal of a newly created key */}
          {newKey && (
            <div className="card" style={{ marginBottom: 18, borderColor: "rgba(242,193,78,0.45)", boxShadow: "0 0 0 1px rgba(242,193,78,0.2)" }}>
              <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>Your new API key</div>
              <p style={{ color: "var(--muted)", fontSize: 13, marginBottom: 12 }}>
                Copy it now — we only store a hash, so it <b style={{ color: "var(--light)" }}>won't be shown again.</b>
              </p>
              <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                <code style={{ flex: 1, minWidth: 240, fontFamily: MONO, fontSize: 13, color: "var(--gold)", background: "rgba(6,11,20,0.7)", border: "1px solid var(--border)", borderRadius: 8, padding: "11px 13px", wordBreak: "break-all" }}>{newKey}</code>
                <CopyButton text={newKey} label="Copy key" />
              </div>
            </div>
          )}

          {/* Keys */}
          <div className="card">
            <div className="section-title">Your API keys</div>
            <div style={{ display: "flex", gap: 10, marginBottom: 18, flexWrap: "wrap" }}>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Key name (optional) — e.g. Trading bot" style={{ flex: 1, minWidth: 220 }} />
              <button className="btn btn-primary" onClick={generate} disabled={busy} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                <Plus size={15} /> {busy ? "Generating…" : "Generate key"}
              </button>
            </div>
            {err && <div style={{ color: "var(--rose)", fontSize: 13, marginBottom: 12 }}>{err}</div>}

            {keys.length === 0 ? (
              <div style={{ color: "var(--muted)", fontSize: 13, padding: "8px 0" }}>No API keys yet. Generate one to get started.</div>
            ) : (
              <table>
                <thead><tr><th>Name</th><th>Key</th><th>Last used</th><th></th></tr></thead>
                <tbody>
                  {keys.map((k) => (
                    <tr key={k.key_id}>
                      <td>{k.name || <span style={{ color: "var(--muted)" }}>—</span>}</td>
                      <td style={{ fontFamily: MONO, fontSize: 12 }}>{k.prefix}••••••••</td>
                      <td style={{ color: "var(--muted)", fontSize: 12 }}>{k.last_used ? new Date(k.last_used).toLocaleDateString() : "never"}</td>
                      <td style={{ textAlign: "right" }}>
                        <button className="btn btn-secondary" onClick={() => revoke(k.key_id)} style={{ padding: "6px 12px", display: "inline-flex", alignItems: "center", gap: 5 }}>
                          <Trash2 size={13} /> Revoke
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}

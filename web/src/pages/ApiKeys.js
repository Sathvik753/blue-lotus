import React, { useEffect, useState } from "react";
import { api } from "../utils/api";
import { Copy, Key, Check } from "lucide-react";

export default function ApiKeys() {
  const [keys, setKeys] = useState([]);
  const [newKey, setNewKey] = useState(null);
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => { loadKeys(); }, []);

  async function loadKeys() {
    const res = await api.get("/auth/api-keys");
    if (res?.ok) setKeys(res.data);
  }

  async function generate() {
    setLoading(true);
    const res = await api.post(`/auth/api-keys${name ? `?name=${encodeURIComponent(name)}` : ""}`);
    if (res?.ok) {
      setNewKey(res.data.key);
      setName("");
      await loadKeys();
    }
    setLoading(false);
  }

  function copy(text) {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="fade-in" style={{ maxWidth: 680 }}>
      <div className="accent-line" />
      <h1 style={{ fontSize: 32, marginBottom: 8 }}>API Keys</h1>
      <p style={{ color: "var(--muted)", marginBottom: 32 }}>
        Use API keys to call the Blue Lotus API programmatically.
      </p>

      {/* Generate new key */}
      <div className="card" style={{ marginBottom: 24 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Generate New Key</h3>
        <div style={{ display: "flex", gap: 10 }}>
          <input
            value={name} onChange={e => setName(e.target.value)}
            placeholder="Key name (e.g. production, backtest)"
            style={{ flex: 1 }}
          />
          <button onClick={generate} disabled={loading} className="btn btn-primary"
            style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Key size={14} />
            {loading ? "Generating..." : "Generate"}
          </button>
        </div>

        {newKey && (
          <div style={{
            marginTop: 16, background: "rgba(20,143,119,0.1)",
            border: "1px solid rgba(20,143,119,0.3)", borderRadius: 8, padding: 16,
          }}>
            <div style={{ fontSize: 11, color: "var(--teal)", marginBottom: 8, fontWeight: 600, letterSpacing: "0.06em" }}>
              ✓ KEY CREATED — COPY NOW, WON'T BE SHOWN AGAIN
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <code style={{
                flex: 1, background: "var(--dark)", padding: "8px 12px",
                borderRadius: 6, fontSize: 12, color: "var(--gold)",
                fontFamily: "DM Mono, monospace", wordBreak: "break-all",
              }}>
                {newKey}
              </code>
              <button onClick={() => copy(newKey)} className="btn btn-secondary"
                style={{ display: "flex", alignItems: "center", gap: 6, whiteSpace: "nowrap" }}>
                {copied ? <Check size={14} color="var(--teal)" /> : <Copy size={14} />}
                {copied ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Existing keys */}
      <div className="card" style={{ marginBottom: 32 }}>
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Your Keys</h3>
        {keys.length === 0 ? (
          <div style={{ color: "var(--muted)", fontSize: 13 }}>No API keys yet.</div>
        ) : (
          <table>
            <thead><tr><th>Name</th><th>ID</th><th>Last Used</th><th>Created</th></tr></thead>
            <tbody>
              {keys.map(k => (
                <tr key={k.key_id}>
                  <td style={{ fontWeight: 500 }}>{k.name || "Unnamed"}</td>
                  <td className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>
                    {k.key_id?.slice(0, 12)}...
                  </td>
                  <td style={{ color: "var(--muted)", fontSize: 12 }}>
                    {k.last_used ? new Date(k.last_used).toLocaleDateString() : "Never"}
                  </td>
                  <td style={{ color: "var(--muted)", fontSize: 12 }}>
                    {new Date(k.created_at).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Code example */}
      <div className="card">
        <h3 style={{ fontSize: 14, marginBottom: 16 }}>Example Usage</h3>
        <pre style={{
          background: "var(--dark)", border: "1px solid var(--border)",
          borderRadius: 8, padding: 16, fontSize: 12,
          fontFamily: "DM Mono, monospace", color: "var(--light)",
          overflowX: "auto", lineHeight: 1.8,
        }}>
{`import requests, time

API_KEY = "bl_your_key_here"
BASE    = "https://blue-lotus-production.up.railway.app"

# Submit a stress test
r = requests.post(f"{BASE}/run/custom",
    headers={"X-API-Key": API_KEY},
    json={"returns": [0.012, -0.005, 0.003, ...],
          "n_paths": 5000, "horizon": 252}
)
run_id = r.json()["run_id"]

# Poll for results
while True:
    r = requests.get(f"{BASE}/run/{run_id}",
        headers={"X-API-Key": API_KEY})
    data = r.json()
    if data["status"] == "completed":
        print("DD:", data["result"]["drawdown"]["mean"])
        print("ES:", data["result"]["expected_shortfall"]["aggregate"])
        break
    time.sleep(2)`}
        </pre>
      </div>
    </div>
  );
}

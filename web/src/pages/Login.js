import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../utils/api";
import { Activity } from "lucide-react";

export default function Login() {
  const navigate = useNavigate();
  const [tab, setTab] = useState("login");
  const [form, setForm] = useState({ email: "", password: "", name: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  if (api.isLoggedIn()) { navigate("/run"); return null; }

  async function submit(e) {
    e.preventDefault();
    setError(""); setLoading(true);
    try {
      let res;
      if (tab === "login") {
        res = await api.login(form.email, form.password);
      } else {
        res = await api.register(form.email, form.password, form.name);
      }
      if (res?.ok) {
        navigate("/run");
      } else {
        setError(res?.data?.detail || "Something went wrong.");
      }
    } catch {
      setError("Network error.");
    }
    setLoading(false);
  }

  return (
    <div style={{
      minHeight: "100vh", background: "var(--navy)",
      display: "flex", alignItems: "center", justifyContent: "center",
      padding: 20,
    }}>
      {/* Background grid */}
      <div style={{
        position: "fixed", inset: 0, opacity: 0.03,
        backgroundImage: "linear-gradient(var(--border) 1px, transparent 1px), linear-gradient(90deg, var(--border) 1px, transparent 1px)",
        backgroundSize: "40px 40px", pointerEvents: "none",
      }} />

      <div className="fade-in" style={{ width: "100%", maxWidth: 400 }}>
        {/* Logo */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{
            display: "inline-flex", alignItems: "center", justifyContent: "center",
            width: 56, height: 56, background: "rgba(212,172,13,0.1)",
            borderRadius: 16, border: "1px solid rgba(212,172,13,0.3)",
            marginBottom: 16,
          }}>
            <Activity size={24} color="var(--gold)" />
          </div>
          <h1 style={{ fontSize: 28, color: "var(--white)", marginBottom: 4 }}>Blue Lotus Labs</h1>
          <p style={{ color: "var(--muted)", fontSize: 13 }}>Stress-Testing Engine</p>
        </div>

        {/* Card */}
        <div className="card">
          {/* Tabs */}
          <div style={{
            display: "flex", background: "var(--dark)", borderRadius: 8,
            padding: 4, marginBottom: 24, gap: 4,
          }}>
            {["login", "register"].map(t => (
              <button key={t} onClick={() => setTab(t)} style={{
                flex: 1, padding: "8px", borderRadius: 6, border: "none",
                background: tab === t ? "var(--card)" : "transparent",
                color: tab === t ? "var(--gold)" : "var(--muted)",
                fontFamily: "Syne, sans-serif", fontWeight: 600,
                fontSize: 12, letterSpacing: "0.06em", textTransform: "uppercase",
                cursor: "pointer", transition: "all 0.15s",
                boxShadow: tab === t ? "0 1px 3px rgba(0,0,0,0.3)" : "none",
              }}>
                {t === "login" ? "Sign In" : "Create Account"}
              </button>
            ))}
          </div>

          <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            {tab === "register" && (
              <div>
                <label>Name (optional)</label>
                <input
                  placeholder="Your name"
                  value={form.name}
                  onChange={e => setForm({ ...form, name: e.target.value })}
                />
              </div>
            )}
            <div>
              <label>Email</label>
              <input
                type="email" required
                placeholder="you@example.com"
                value={form.email}
                onChange={e => setForm({ ...form, email: e.target.value })}
              />
            </div>
            <div>
              <label>Password</label>
              <input
                type="password" required minLength={8}
                placeholder="••••••••"
                value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
              />
            </div>

            {error && (
              <div style={{
                background: "rgba(192,57,43,0.1)", border: "1px solid rgba(192,57,43,0.3)",
                borderRadius: 8, padding: "10px 14px", color: "var(--rose)", fontSize: 13,
              }}>
                {error}
              </div>
            )}

            <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: "100%", padding: "12px" }}>
              {loading ? <span className="spinner" /> : tab === "login" ? "Sign In" : "Create Account"}
            </button>
          </form>
        </div>

        <p style={{ textAlign: "center", color: "var(--muted)", fontSize: 11, marginTop: 20, letterSpacing: "0.04em" }}>
          BLUE LOTUS LABS · STRESS-TESTING ENGINE · v1.0
        </p>
      </div>
    </div>
  );
}

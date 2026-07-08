import React, { useState } from "react";
import { useNavigate, Link, useLocation } from "react-router-dom";
import { useAuth } from "../context/Auth";
import Logo from "../components/Logo";

export default function Login() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const dest = location.state?.from || "/run";

  async function submit(e) {
    e.preventDefault();
    setError(""); setBusy(true);
    try {
      await login(email.trim(), password);
      navigate(dest, { replace: true });
    } catch (err) {
      setError(err.message || "Login failed.");
      setBusy(false);
    }
  }

  return (
    <AuthShell title="Welcome back" subtitle="Sign in to your workspace.">
      <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        <div>
          <label>Email</label>
          <input type="email" value={email} onChange={e => setEmail(e.target.value)}
            placeholder="you@firm.com" required autoFocus />
        </div>
        <div>
          <label>Password</label>
          <input type="password" value={password} onChange={e => setPassword(e.target.value)}
            placeholder="••••••••" required />
        </div>
        {error && <ErrorBox msg={error} />}
        <button type="submit" className="btn btn-primary" disabled={busy}
          style={{ padding: "12px", marginTop: 4 }}>
          {busy ? <><span className="spinner" style={{ marginRight: 8 }} />Signing in…</> : "Sign in"}
        </button>
      </form>
      <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 20, textAlign: "center" }}>
        No account? <Link to="/register" style={{ color: "var(--gold)" }}>Create one</Link>
      </p>
    </AuthShell>
  );
}

export function AuthShell({ title, subtitle, children }) {
  return (
    <div className="fade-in" style={{
      minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
    }}>
      <div style={{ width: "100%", maxWidth: 400 }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 28 }}>
          <Logo size={54} />
          <div className="gradient-text" style={{
            fontFamily: "Syne, sans-serif", fontWeight: 800, fontSize: 22, marginTop: 12,
          }}>Blue Lotus</div>
        </div>
        <div className="card">
          <h1 style={{ fontSize: 22, marginBottom: 4 }}>{title}</h1>
          <p style={{ color: "var(--muted)", fontSize: 13, marginBottom: 24 }}>{subtitle}</p>
          {children}
        </div>
      </div>
    </div>
  );
}

export function ErrorBox({ msg }) {
  if (!msg) return null;
  return (
    <div style={{
      background: "rgba(224,86,63,0.1)", border: "1px solid rgba(224,86,63,0.3)",
      borderRadius: 8, padding: "10px 14px", color: "var(--rose)", fontSize: 13,
    }}>{msg}</div>
  );
}

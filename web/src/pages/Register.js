import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../context/Auth";
import { AuthShell, ErrorBox } from "./Login";

export default function Register() {
  const navigate = useNavigate();
  const { register } = useAuth();
  const [form, setForm] = useState({ name: "", org_name: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });

  async function submit(e) {
    e.preventDefault();
    setError("");
    if (form.password.length < 8) { setError("Password must be at least 8 characters."); return; }
    setBusy(true);
    try {
      await register({
        email: form.email.trim(),
        password: form.password,
        name: form.name.trim() || undefined,
        org_name: form.org_name.trim() || undefined,
      });
      navigate("/run", { replace: true });
    } catch (err) {
      setError(err.message || "Registration failed.");
      setBusy(false);
    }
  }

  return (
    <AuthShell title="Create your workspace" subtitle="Start with 25 free stress runs a month.">
      <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div className="grid-2" style={{ gap: 12 }}>
          <div>
            <label>Your name</label>
            <input value={form.name} onChange={set("name")} placeholder="Jane Doe" />
          </div>
          <div>
            <label>Firm / org</label>
            <input value={form.org_name} onChange={set("org_name")} placeholder="Acme Capital" />
          </div>
        </div>
        <div>
          <label>Work email</label>
          <input type="email" value={form.email} onChange={set("email")} placeholder="you@firm.com" required />
        </div>
        <div>
          <label>Password</label>
          <input type="password" value={form.password} onChange={set("password")}
            placeholder="At least 8 characters" required />
        </div>
        {error && <ErrorBox msg={error} />}
        <button type="submit" className="btn btn-primary" disabled={busy} style={{ padding: "12px", marginTop: 4 }}>
          {busy ? <><span className="spinner" style={{ marginRight: 8 }} />Creating…</> : "Create workspace"}
        </button>
        <p style={{ color: "var(--muted)", fontSize: 11.5, lineHeight: 1.5, textAlign: "center", margin: "2px 4px 0" }}>
          By creating an account you agree to our{" "}
          <Link to="/terms" style={{ color: "var(--gold)" }}>Terms</Link> and{" "}
          <Link to="/privacy" style={{ color: "var(--gold)" }}>Privacy Policy</Link>, and understand that
          Blue Lotus is a risk-analytics tool, <Link to="/disclaimer" style={{ color: "var(--gold)" }}>not investment advice</Link>.
        </p>
      </form>
      <p style={{ color: "var(--muted)", fontSize: 13, marginTop: 20, textAlign: "center" }}>
        Already have an account? <Link to="/login" style={{ color: "var(--gold)" }}>Sign in</Link>
      </p>
    </AuthShell>
  );
}

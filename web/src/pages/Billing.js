import React, { useEffect, useState } from "react";
import { api } from "../utils/api";
import { useAuth } from "../context/Auth";
import { Check } from "lucide-react";

export default function Billing() {
  const { refresh } = useAuth();
  const [status, setStatus] = useState(null);
  const [plans, setPlans] = useState([]);
  const [busy, setBusy] = useState("");
  const [msg, setMsg] = useState("");

  async function load() {
    const [s, p] = await Promise.all([api.get("/billing/status"), api.get("/billing/plans")]);
    if (s.ok) setStatus(s.data);
    if (p.ok) setPlans(p.data);
  }
  useEffect(() => { load(); }, []);

  async function choose(tier) {
    setBusy(tier); setMsg("");
    const res = await api.post("/billing/checkout", { tier });
    setBusy("");
    if (!res.ok) { setMsg(res.data?.detail || "Checkout failed."); return; }
    if (res.data.mode === "live" && res.data.checkout_url) {
      window.location.href = res.data.checkout_url;   // to Stripe
      return;
    }
    // Mock mode: plan already switched server-side.
    setMsg(res.data.message || "Plan updated.");
    await load();
    await refresh();
  }

  const pct = status && status.runs_limit
    ? Math.min(100, Math.round((status.runs_used / status.runs_limit) * 100))
    : 0;

  return (
    <div className="fade-in">
      <div className="accent-line" />
      <h1 style={{ fontSize: 34, marginBottom: 8 }} className="gradient-text">Billing &amp; Usage</h1>
      <p style={{ color: "var(--muted)", marginBottom: 28 }}>
        Manage your plan and track this month's stress-run usage.
      </p>

      {status && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
            <div>
              <div style={{ fontSize: 12, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em" }}>Current plan</div>
              <div style={{ fontSize: 24, fontFamily: "Syne, sans-serif", fontWeight: 800 }}>
                {status.plan_name}
                <span className="status-pill" style={{
                  fontSize: 11, marginLeft: 10, padding: "3px 10px", borderRadius: 999,
                  background: "rgba(33,208,173,0.12)", color: "var(--teal-2)", verticalAlign: "middle",
                }}>{status.subscription_status}</span>
              </div>
            </div>
            {!status.stripe_enabled && (
              <div style={{ fontSize: 11, color: "var(--muted)", maxWidth: 220, textAlign: "right" }}>
                Billing is in mock mode — set Stripe keys to take live payments.
              </div>
            )}
          </div>

          <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 6 }}>
            {status.runs_used} / {status.runs_limit == null ? "∞" : status.runs_limit} runs used · {status.period}
          </div>
          <div style={{ height: 8, borderRadius: 999, background: "var(--dark)", overflow: "hidden" }}>
            <div style={{
              width: `${pct}%`, height: "100%",
              background: pct > 90 ? "var(--rose)" : "linear-gradient(90deg, var(--teal-2), var(--gold))",
              transition: "width 0.4s",
            }} />
          </div>
        </div>
      )}

      {msg && (
        <div style={{
          background: "rgba(33,208,173,0.1)", border: "1px solid rgba(33,208,173,0.3)",
          borderRadius: 8, padding: "10px 14px", color: "var(--teal-2)", fontSize: 13, marginBottom: 20,
        }}>{msg}</div>
      )}

      <div className="grid-3" style={{ gap: 18 }}>
        {plans.map(p => {
          const current = status && status.plan === p.tier;
          return (
            <div key={p.tier} className="card" style={{
              borderColor: current ? "rgba(33,208,173,0.4)" : "var(--border-soft)",
            }}>
              <div style={{ fontFamily: "Syne, sans-serif", fontWeight: 700, fontSize: 16 }}>{p.name}</div>
              <div style={{ margin: "10px 0 4px" }}>
                <span style={{ fontSize: 28, fontWeight: 800 }}>
                  {p.price_usd == null ? "Custom" : p.price_usd === 0 ? "$0" : `$${p.price_usd.toLocaleString()}`}
                </span>
                {p.price_usd ? <span style={{ color: "var(--muted)", fontSize: 12 }}> /mo</span> : null}
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 7, margin: "14px 0 18px" }}>
                {p.features.map(f => (
                  <div key={f} style={{ display: "flex", gap: 8, fontSize: 12.5 }}>
                    <Check size={14} color="var(--teal-2)" style={{ marginTop: 2, flexShrink: 0 }} />
                    <span style={{ color: "var(--light)" }}>{f}</span>
                  </div>
                ))}
              </div>
              {current ? (
                <button className="btn btn-secondary" disabled style={{ width: "100%", opacity: 0.6 }}>Current plan</button>
              ) : p.tier === "free" ? (
                <button className="btn btn-secondary" disabled style={{ width: "100%", opacity: 0.5 }}>—</button>
              ) : (
                <button className="btn btn-primary" style={{ width: "100%" }}
                  disabled={busy === p.tier} onClick={() => choose(p.tier)}>
                  {busy === p.tier ? "Working…" : p.tier === "enterprise" ? "Upgrade" : "Upgrade to Pro"}
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

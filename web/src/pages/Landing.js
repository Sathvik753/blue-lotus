import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Check, ArrowRight, ShieldCheck, Activity, GitBranch, FileText } from "lucide-react";
import Logo from "../components/Logo";
import { api } from "../utils/api";
import { useAuth } from "../context/Auth";


const HIGHLIGHTS = [
  { icon: Activity, title: "Regime-aware Monte Carlo", body: "Volatility regimes, EVT tails, and bootstrap intervals on every metric — not a single-distribution toy." },
  { icon: ShieldCheck, title: "Honest about its limits", body: "Out-of-sample validated on 9 crises and 213 calm windows. It reports the tail gap instead of hiding it." },
  { icon: GitBranch, title: "API-first", body: "Every run is reproducible from a seed and reachable over a clean REST API with per-org keys." },
];

export default function Landing() {
  const { user } = useAuth();
  const [plans, setPlans] = useState([]);

  useEffect(() => {
    api.get("/billing/plans").then(r => { if (r.ok) setPlans(r.data); });
  }, []);

  return (
    <div className="fade-in" style={{ maxWidth: 1080, margin: "0 auto", padding: "0 24px 80px" }}>
      {/* Nav */}
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "26px 0" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <Logo size={38} />
          <span className="gradient-text" style={{ fontFamily: "Syne, sans-serif", fontWeight: 800, fontSize: 18 }}>
            Blue Lotus
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
          <Link to="/status" style={{ color: "var(--muted)", fontSize: 13 }}>Status</Link>
          {user ? (
            <Link to="/run" className="btn btn-primary" style={{ padding: "9px 18px" }}>Open app</Link>
          ) : (
            <>
              <Link to="/login" style={{ color: "var(--light)", fontSize: 13, fontWeight: 600 }}>Sign in</Link>
              <Link to="/register" className="btn btn-primary" style={{ padding: "9px 18px" }}>Get started</Link>
            </>
          )}
        </div>
      </header>

      {/* Hero */}
      <section style={{ textAlign: "center", padding: "70px 0 50px" }}>
        <div style={{
          display: "inline-block", fontSize: 12, letterSpacing: "0.14em", textTransform: "uppercase",
          color: "var(--teal-2)", border: "1px solid var(--border-soft)", borderRadius: 999,
          padding: "6px 14px", marginBottom: 26,
        }}>
          Institutional stress-testing, on demand
        </div>
        <h1 style={{ fontSize: 52, lineHeight: 1.05, letterSpacing: "-0.03em", marginBottom: 20 }}>
          Know how bad it can get<br />
          <span className="gradient-text">before it does.</span>
        </h1>
        <p style={{ color: "var(--muted)", fontSize: 17, maxWidth: 600, margin: "0 auto 34px", lineHeight: 1.6 }}>
          Blue Lotus turns any return series into a forward distribution of drawdown, tail-loss,
          and recovery — with confidence intervals and a model-fragility score on every run.
        </p>
        <div style={{ display: "flex", gap: 14, justifyContent: "center" }}>
          <Link to="/register" className="btn btn-primary" style={{ padding: "13px 30px", display: "flex", alignItems: "center", gap: 8 }}>
            Start free <ArrowRight size={16} />
          </Link>
        </div>
      </section>

      {/* Highlights */}
      <section className="grid-3" style={{ gap: 18, marginBottom: 70 }}>
        {HIGHLIGHTS.map(({ icon: Icon, title, body }) => (
          <div key={title} className="card">
            <Icon size={22} color="var(--gold)" style={{ marginBottom: 12 }} />
            <div style={{ fontFamily: "Syne, sans-serif", fontWeight: 700, fontSize: 16, marginBottom: 8 }}>{title}</div>
            <p style={{ color: "var(--muted)", fontSize: 13.5, lineHeight: 1.6 }}>{body}</p>
          </div>
        ))}
      </section>

      {/* Research */}
      <section style={{ marginBottom: 70 }}>
        <div className="card" style={{
          display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between",
          gap: 24, padding: "26px 28px",
        }}>
          <div style={{ maxWidth: 560 }}>
            <div style={{
              fontFamily: "DM Mono, monospace", fontSize: 11, letterSpacing: "0.16em",
              textTransform: "uppercase", color: "var(--teal-2)", marginBottom: 8,
            }}>Research</div>
            <h2 style={{ fontSize: 24, marginBottom: 8 }}>The engine, in full detail</h2>
            <p style={{ color: "var(--muted)", fontSize: 14, lineHeight: 1.6 }}>
              Read the methodology and the out-of-sample evidence behind Blue Lotus —
              regime modeling, Extreme Value tails, and a walk-forward validation across
              744 asset-years that benchmarks the engine against naive baselines.
            </p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, minWidth: 220 }}>
            <a href="/engine-paper.pdf" target="_blank" rel="noopener noreferrer"
              className="btn btn-primary" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
              <FileText size={15} /> Engine paper
            </a>
            <a href="/validation-paper.pdf" target="_blank" rel="noopener noreferrer"
              className="btn btn-secondary" style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 8 }}>
              <FileText size={15} /> Validation study
            </a>
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" style={{ textAlign: "center", marginBottom: 40 }}>
        <h2 style={{ fontSize: 32, marginBottom: 8 }}>Simple, usage-based pricing</h2>
        <p style={{ color: "var(--muted)", marginBottom: 40 }}>Start free. Upgrade when the desk relies on it.</p>

        <div className="grid-3" style={{ gap: 18, textAlign: "left" }}>
          {plans.map(p => (
            <div key={p.tier} className="card" style={{
              position: "relative",
              borderColor: p.tier === "pro" ? "rgba(212,172,13,0.4)" : "var(--border-soft)",
              boxShadow: p.tier === "pro" ? "0 0 0 1px rgba(212,172,13,0.25)" : "none",
            }}>
              {p.tier === "pro" && (
                <div style={{
                  position: "absolute", top: -11, right: 18, fontSize: 10, fontWeight: 700,
                  letterSpacing: "0.1em", textTransform: "uppercase", color: "var(--navy)",
                  background: "linear-gradient(var(--gold), var(--gold-2))", borderRadius: 999, padding: "3px 10px",
                }}>Most popular</div>
              )}
              <div style={{ fontFamily: "Syne, sans-serif", fontWeight: 700, fontSize: 17 }}>{p.name}</div>
              <div style={{ margin: "12px 0 4px" }}>
                <span style={{ fontSize: 34, fontWeight: 800 }}>
                  {p.price_usd == null ? "Custom" : p.price_usd === 0 ? "$0" : `$${p.price_usd.toLocaleString()}`}
                </span>
                {p.price_usd != null && p.price_usd > 0 && (
                  <span style={{ color: "var(--muted)", fontSize: 13 }}> /mo</span>
                )}
              </div>
              <p style={{ color: "var(--muted)", fontSize: 13, minHeight: 38, marginBottom: 16 }}>{p.blurb}</p>
              <div style={{ display: "flex", flexDirection: "column", gap: 9, marginBottom: 22 }}>
                {p.features.map(f => (
                  <div key={f} style={{ display: "flex", alignItems: "flex-start", gap: 8, fontSize: 13 }}>
                    <Check size={15} color="var(--teal-2)" style={{ marginTop: 2, flexShrink: 0 }} />
                    <span style={{ color: "var(--light)" }}>{f}</span>
                  </div>
                ))}
              </div>
              <Link to="/register" className={`btn ${p.tier === "pro" ? "btn-primary" : "btn-secondary"}`}
                style={{ width: "100%", textAlign: "center", display: "block" }}>
                {p.tier === "free" ? "Start free" : p.tier === "enterprise" ? "Get started" : "Choose Pro"}
              </Link>
            </div>
          ))}
        </div>
      </section>

      <footer style={{
        borderTop: "1px solid var(--border-soft)", marginTop: 60, paddingTop: 26,
        display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 12,
        color: "var(--muted)", fontSize: 12,
      }}>
        <span>© {new Date().getFullYear()} Blue Lotus Labs · Risk analytics, not investment advice.</span>
        <span style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
          <Link to="/terms" style={{ color: "var(--muted)" }}>Terms</Link>
          <Link to="/privacy" style={{ color: "var(--muted)" }}>Privacy</Link>
          <Link to="/disclaimer" style={{ color: "var(--muted)" }}>Disclaimer</Link>
          <Link to="/status" style={{ color: "var(--muted)" }}>Status</Link>
        </span>
      </footer>
    </div>
  );
}

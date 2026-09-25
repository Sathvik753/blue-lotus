import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Check, ShieldCheck, Activity, GitBranch, FileText } from "lucide-react";
import Logo from "../components/Logo";
import Mandala from "../components/Mandala";
import { api } from "../utils/api";
import { useAuth } from "../context/Auth";
import { PAYMENTS_ENABLED } from "../config";

const TILES = [
  { icon: Activity, v: "a", title: "Regime-aware Monte Carlo", body: "Volatility regimes, EVT tails, and bootstrap intervals on every metric — not a single-distribution toy." },
  { icon: ShieldCheck, v: "b", title: "Honest about its limits", body: "Out-of-sample validated on 9 crises and 213 calm windows. It reports the tail gap instead of hiding it." },
  { icon: GitBranch, v: "c", title: "API-first", body: "Reproducible from a seed and reachable over a clean REST API with per-org keys." },
  { icon: FileText, v: "a", title: "Every run, a full report", body: "Drawdown, tail-loss, recovery, and a model-fragility score — exportable as JSON or PDF." },
];

const STATS = [
  { num: "744", cap: "asset-years of walk-forward, out-of-sample validation" },
  { num: <>9<span className="u"> crises</span></>, cap: "of market history the engine was stress-tested against" },
  { num: <>4.0<span className="u">%</span></>, cap: "calm-market p5 breach rate — statistically calibrated" },
];

function useReveal(dep) {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll(".reveal:not(.in)"));
    if (!("IntersectionObserver" in window) || els.length === 0) {
      els.forEach((e) => e.classList.add("in"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => entries.forEach((en) => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } }),
      { threshold: 0.12 }
    );
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, [dep]);
}

function priceLabel(p) {
  if (p.price_usd == null) return { amt: "Custom", per: null };
  if (p.price_usd === 0) return { amt: "$0", per: null };
  return { amt: `$${p.price_usd.toLocaleString()}`, per: "/mo" };
}

export default function Landing() {
  const { user } = useAuth();
  const [plans, setPlans] = useState([]);
  const [payNotice, setPayNotice] = useState(false);

  useEffect(() => {
    api.get("/billing/plans").then((r) => { if (r.ok && Array.isArray(r.data)) setPlans(r.data); });
  }, []);
  useReveal(plans.length);

  return (
    <div className="bl-page">
      {/* Geometric lotus mandala behind the whole page — rotates on scroll */}
      <Mandala />
      <div className="site-scrim" aria-hidden="true" />

      {/* Nav */}
      <nav className="bl-nav">
        <div className="bl-nav-inner">
          <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
            <Logo size={34} />
            <span className="gradient-text" style={{ fontWeight: 600, fontSize: 19, letterSpacing: "-0.02em" }}>Blue Lotus Labs</span>
          </div>
          <div className="bl-nav-links">
            <a href="#pricing">Pricing</a>
            <a href="#research">Research</a>
            <Link to="/status">Status</Link>
            {user ? (
              <Link to="/run" className="btn btn-primary" style={{ padding: "9px 18px" }}>Open app</Link>
            ) : (
              <>
                <Link to="/login">Sign in</Link>
                <Link to="/register" className="btn btn-primary" style={{ padding: "9px 18px" }}>Get started</Link>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Hero — oversized editorial title over the rotating mandala */}
      <header className="hero">
        <div className="hero-inner">
          <h1 className="hero-title">Blue Lotus</h1>
          <div className="hero-links">
            <Link to="/register" className="link-chevron">Start free</Link>
            <a href="#pricing" className="link-chevron">See pricing</a>
          </div>
          <div className="hero-pills">
            <span className="pill">Institutional-grade</span>
            <span className="pill">744 asset-years validated</span>
            <span className="pill">9 crises stress-tested</span>
          </div>
          <p className="hero-sub">
            Institutional trading software, made accessible — stress-test your
            strategies and investments with bleeding-edge financial mathematics.
          </p>
        </div>
        <a href="#pricing" className="scroll-cue" aria-label="Scroll down">
          SCROLL<span aria-hidden="true">›</span>
        </a>
      </header>

      {/* Highlights — borderless feature columns over the page-wide graph */}
      <section className="sec feat-sec">
        <div className="wrap center reveal">
          <span className="eyebrow">Get the highlights</span>
          <h2 className="h-lg">Built like a risk desk, not a demo.</h2>
        </div>
        <div className="feat-row">
          {TILES.map(({ icon: Icon, title, body }, i) => (
            <div key={title} className="feat reveal" style={{ transitionDelay: `${i * 80}ms` }}>
              <Icon className="feat-ic" size={26} />
              <h3>{title}</h3>
              <p>{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Deep feature + specs */}
      <section className="sec feature">
        <div className="wrap reveal">
          <span className="eyebrow">The math, in the open</span>
          <h2 className="display">Every run tells you how much to trust it.</h2>
          <p className="lead lead-narrow">
            Most tools hand you one number and a false sense of certainty. Blue Lotus
            gives you a distribution, confidence intervals, and a fragility score — so
            you know when the model is on solid ground and when it isn't.
          </p>
          <div className="specs">
            {STATS.map((s, i) => (
              <div key={i} className="spec reveal" style={{ transitionDelay: `${i * 90}ms` }}>
                <div className="spec-num">{s.num}</div>
                <div className="spec-cap">{s.cap}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Research */}
      <section id="research" className="sec sec--band center">
        <div className="wrap--narrow reveal">
          <span className="eyebrow">Research</span>
          <h2 className="h-lg">The engine, in full detail.</h2>
          <p className="lead lead-narrow" style={{ marginTop: 16 }}>
            Read the methodology and the out-of-sample evidence behind Blue Lotus —
            regime modeling, Extreme Value tails, and a walk-forward validation across
            744 asset-years that benchmarks the engine against naive baselines.
          </p>
          <div className="paper-row">
            <a href="/engine-paper.pdf" target="_blank" rel="noopener noreferrer"
              className="btn btn-primary btn-lg"><FileText size={16} /> Engine paper</a>
            <a href="/validation-paper.pdf" target="_blank" rel="noopener noreferrer"
              className="btn btn-secondary btn-lg"><FileText size={16} /> Validation study</a>
          </div>
        </div>
      </section>

      {/* Pricing */}
      <section id="pricing" className="sec center">
        <div className="wrap">
          <div className="reveal">
            <h2 className="h-lg">Pricing that scales with the desk.</h2>
            <p className="lead lead-narrow" style={{ marginTop: 14 }}>Start free. Move up when the book depends on it.</p>
          </div>
          <div className="plans">
            {plans.map((p, i) => {
              const pop = p.tier === "plus";
              const { amt, per } = priceLabel(p);
              const isPaid = p.tier === "plus" || p.tier === "pro";
              const disabled = isPaid && !PAYMENTS_ENABLED;
              const label = p.tier === "free" ? "Start free"
                : p.tier === "institutional" ? "Contact us" : `Choose ${p.name}`;
              return (
                <div key={p.tier} className={`plan reveal ${pop ? "plan--pop" : ""}`} style={{ transitionDelay: `${i * 60}ms` }}>
                  {pop && <div className="plan-tag">Most popular</div>}
                  <div className="plan-name">{p.name}</div>
                  <div className="plan-price">
                    <span className="amt">{amt}</span>
                    {per && <span className="per">{per}</span>}
                  </div>
                  <div className="plan-blurb">{p.blurb}</div>
                  <div className="plan-feats">
                    {p.features.map((f) => (
                      <div key={f} className="plan-feat">
                        <Check size={14} color="var(--teal-2)" style={{ marginTop: 1, flexShrink: 0 }} />
                        <span>{f}</span>
                      </div>
                    ))}
                  </div>
                  {disabled
                    ? <button type="button" className="btn btn-secondary plan-cta" style={{ opacity: 0.8 }} onClick={() => setPayNotice(true)}>Temporarily unavailable</button>
                    : pop
                      ? <Link to="/register" className="btn btn-primary plan-cta">{label}</Link>
                      : <Link to="/register" className="link-chevron plan-cta">{label}</Link>}
                </div>
              );
            })}
          </div>
          {payNotice && !PAYMENTS_ENABLED && (
            <p style={{ marginTop: 22, color: "var(--muted)", fontSize: 14, maxWidth: 620, marginInline: "auto" }}>
              Paid plans are temporarily disabled while we finalize payment processing.
              The <b style={{ color: "var(--light)" }}>Sandbox</b> plan is fully available to explore right now.
            </p>
          )}
        </div>
      </section>

      {/* Footer */}
      <footer className="bl-foot">
        <span>© {new Date().getFullYear()} Blue Lotus Labs · Risk analytics, not investment advice.</span>
        <span className="bl-foot-links">
          <Link to="/terms">Terms</Link>
          <Link to="/privacy">Privacy</Link>
          <Link to="/disclaimer">Disclaimer</Link>
          <Link to="/status">Status</Link>
        </span>
      </footer>
    </div>
  );
}

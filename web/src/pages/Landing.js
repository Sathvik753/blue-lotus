import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Check, ArrowRight, ShieldCheck, Activity, GitBranch, FileText } from "lucide-react";
import Logo from "../components/Logo";
import HeroCanvas from "../components/HeroCanvas";
import { api } from "../utils/api";
import { useAuth } from "../context/Auth";

const HIGHLIGHTS = [
  { icon: Activity, title: "Regime-aware Monte Carlo", body: "Volatility regimes, EVT tails, and bootstrap intervals on every metric — not a single-distribution toy." },
  { icon: ShieldCheck, title: "Honest about its limits", body: "Out-of-sample validated on 9 crises and 213 calm windows. It reports the tail gap instead of hiding it." },
  { icon: GitBranch, title: "API-first", body: "Every run is reproducible from a seed and reachable over a clean REST API with per-org keys." },
];

const STATS = [
  { num: "744", cap: "asset-years of walk-forward, out-of-sample validation" },
  { num: <>9 <span className="u">crises</span></>, cap: "of history the engine was stress-tested against" },
  { num: <>4.0<span className="u">%</span></>, cap: "calm-market p5 breach rate — statistically calibrated" },
];

// Reveal-on-scroll: attaches the .in class when an element enters the viewport.
function useReveal() {
  useEffect(() => {
    const els = Array.from(document.querySelectorAll(".reveal"));
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
  }, []);
}

function priceLabel(p) {
  if (p.price_usd == null) return { amt: "Custom", per: null };
  if (p.price_usd === 0) return { amt: "$0", per: null };
  return { amt: `$${p.price_usd.toLocaleString()}`, per: "/mo" };
}

export default function Landing() {
  const { user } = useAuth();
  const [plans, setPlans] = useState([]);
  const revealReady = useRef(false);

  useEffect(() => {
    api.get("/billing/plans").then((r) => { if (r.ok && Array.isArray(r.data)) setPlans(r.data); });
  }, []);

  useReveal();
  // Re-scan reveals once plans render in (they arrive async).
  useEffect(() => {
    if (plans.length && !revealReady.current) {
      revealReady.current = true;
      const els = document.querySelectorAll(".bl-price-grid .reveal");
      const io = new IntersectionObserver(
        (es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }),
        { threshold: 0.05 }
      );
      els.forEach((e) => io.observe(e));
    }
  }, [plans]);

  return (
    <div className="bl-page fade-in">
      {/* Nav */}
      <nav className="bl-nav">
        <div className="bl-nav-inner">
          <div style={{ display: "flex", alignItems: "center", gap: 11 }}>
            <Logo size={34} />
            <span className="gradient-text" style={{ fontWeight: 600, fontSize: 19, letterSpacing: "-0.02em" }}>
              Blue Lotus
            </span>
          </div>
          <div className="bl-nav-links">
            <a href="#pricing">Pricing</a>
            <a href="#research">Research</a>
            <Link to="/status">Status</Link>
            {user ? (
              <Link to="/run" className="btn btn-primary" style={{ padding: "9px 18px" }}>Open app</Link>
            ) : (
              <>
                <Link to="/login" style={{ color: "var(--light)", fontWeight: 600 }}>Sign in</Link>
                <Link to="/register" className="btn btn-primary" style={{ padding: "9px 18px" }}>Get started</Link>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Hero */}
      <header className="bl-hero">
        <span className="bl-eyebrow">Institutional stress-testing, on demand</span>
        <h1>Know how bad it can get <span className="gradient-text">before it does.</span></h1>
        <p className="bl-hero-sub">
          Blue Lotus turns any return series into a forward distribution of drawdown,
          tail-loss, and recovery — with confidence intervals and a model-fragility
          score on every run.
        </p>
        <div className="bl-cta-row">
          <Link to="/register" className="btn btn-primary bl-btn-lg">
            Start free <ArrowRight size={17} />
          </Link>
          <a href="#pricing" className="link-chevron" style={{ fontSize: 17 }}>See pricing</a>
        </div>
        <div className="bl-fine">Free to start · <b>Plus $25/mo</b> · <b>Pro $100/mo</b> · no card required</div>

        <HeroCanvas />
      </header>

      {/* Highlights */}
      <section className="bl-section">
        <div className="reveal" style={{ textAlign: "center", marginBottom: 44 }}>
          <div className="bl-kicker">Get the highlights</div>
          <h2 style={{ fontSize: "clamp(28px,4vw,42px)", letterSpacing: "-0.03em", marginTop: 12 }}>
            Built like a risk desk, not a demo.
          </h2>
        </div>
        <div className="bl-panels">
          {HIGHLIGHTS.map(({ icon: Icon, title, body }, i) => (
            <div key={title} className="bl-panel reveal" style={{ transitionDelay: `${i * 90}ms` }}>
              <div className="bl-panel-icn"><Icon size={22} color="var(--teal-2)" /></div>
              <h3>{title}</h3>
              <p>{body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Big feature line + stats */}
      <section className="bl-section bl-feature reveal">
        <div className="bl-kicker">The math, in the open</div>
        <h2>Every run tells you how much to trust it.</h2>
        <p>
          Most tools hand you one number and a false sense of certainty. Blue Lotus
          gives you a distribution, confidence intervals, and a fragility score — so
          you know when the model is on solid ground and when it isn't.
        </p>
        <div className="bl-stat-row">
          {STATS.map((s, i) => (
            <div key={i} className="bl-stat reveal" style={{ transitionDelay: `${i * 90}ms` }}>
              <div className="bl-stat-num">{s.num}</div>
              <div className="bl-stat-cap">{s.cap}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Research */}
      <section id="research" className="bl-section bl-section--tight">
        <div className="bl-research reveal">
          <div>
            <div className="bl-kicker">Research</div>
            <h2>The engine, in full detail.</h2>
            <p>
              Read the methodology and the out-of-sample evidence behind Blue Lotus —
              regime modeling, Extreme Value tails, and a walk-forward validation across
              744 asset-years that benchmarks the engine against naive baselines.
            </p>
          </div>
          <div className="bl-research-btns">
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
      <section id="pricing" className="bl-section">
        <div className="bl-price-head reveal">
          <h2>Pricing that scales with the desk.</h2>
          <p>Start free. Move up when the book depends on it.</p>
        </div>
        <div className="bl-price-grid">
          {plans.map((p, i) => {
            const pop = p.tier === "pro";
            const { amt, per } = priceLabel(p);
            const label = p.tier === "free" ? "Start free"
              : p.tier === "custom" ? "Get started"
              : `Choose ${p.name}`;
            return (
              <div key={p.tier} className={`bl-plan reveal ${pop ? "bl-plan--pop" : ""}`} style={{ transitionDelay: `${i * 70}ms` }}>
                {pop && <div className="bl-plan-tag">Most popular</div>}
                <div className="bl-plan-name">{p.name}</div>
                <div className="bl-plan-price">
                  <span className="amt">{amt}</span>
                  {per && <span className="per">{per}</span>}
                </div>
                <div className="bl-plan-blurb">{p.blurb}</div>
                <div className="bl-plan-feats">
                  {p.features.map((f) => (
                    <div key={f} className="bl-plan-feat">
                      <Check size={14} color="var(--teal-2)" style={{ marginTop: 2, flexShrink: 0 }} />
                      <span>{f}</span>
                    </div>
                  ))}
                </div>
                <Link to="/register" className={`btn ${pop ? "btn-primary" : "btn-secondary"}`}>{label}</Link>
              </div>
            );
          })}
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

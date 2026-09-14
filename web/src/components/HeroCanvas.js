import React, { useEffect, useRef } from "react";

/**
 * Interactive Monte-Carlo risk surface.
 *
 * A field of simulated equity paths fans out from a common origin; the tail
 * paths dive into drawdown. A shaded p5–p95 band sits behind them and a sweep
 * line reveals the simulation left-to-right, looping gently. The whole plane
 * tilts in 3D toward the pointer (perspective on the wrapper) so it reads like
 * a physical object floating in space — Apple's "product in the void", rebuilt
 * for a risk engine. No external libraries; pure canvas + CSS transforms.
 */

const N_PATHS = 64;
const STEPS = 180;

// Deterministic PRNG so the fan is identical on every load (mulberry32).
function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(r) {
  // Box–Muller
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Build the paths once. Each is an array of cumulative log-wealth in [0,1]-ish.
function buildPaths() {
  const r = rng(20260914);
  const paths = [];
  for (let i = 0; i < N_PATHS; i++) {
    // A few paths get a fat negative drift → deep drawdowns (the tail).
    const tail = i % 11 === 0;
    const drift = tail ? -0.0016 : 0.0004;
    const vol = tail ? 0.017 : 0.010;
    const pts = new Array(STEPS);
    let w = 0;
    for (let s = 0; s < STEPS; s++) {
      w += drift + vol * gauss(r);
      pts[s] = w;
    }
    paths.push({ pts, tail });
  }
  return paths;
}

export default function HeroCanvas() {
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const rafRef = useRef(0);
  const tilt = useRef({ tx: 0, ty: 0, cx: 0, cy: 0 }); // target + current

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    const paths = buildPaths();

    // Precompute per-step quantile band for the shaded envelope.
    const lo = new Array(STEPS), hi = new Array(STEPS);
    for (let s = 0; s < STEPS; s++) {
      const col = paths.map((p) => p.pts[s]).sort((a, b) => a - b);
      lo[s] = col[Math.floor(col.length * 0.05)];
      hi[s] = col[Math.floor(col.length * 0.95)];
    }
    let vmin = Infinity, vmax = -Infinity;
    for (let s = 0; s < STEPS; s++) {
      for (const p of paths) { if (p.pts[s] < vmin) vmin = p.pts[s]; if (p.pts[s] > vmax) vmax = p.pts[s]; }
    }

    let W = 0, H = 0, dpr = 1;
    function resize() {
      const rect = wrap.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = rect.width; H = rect.height;
      canvas.width = W * dpr; canvas.height = H * dpr;
      canvas.style.width = W + "px"; canvas.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    resize();
    window.addEventListener("resize", resize);

    const padX = 26, padTop = 20, padBot = 26;
    const X = (s) => padX + (s / (STEPS - 1)) * (W - padX * 2);
    const Y = (v) => {
      const t = (v - vmin) / (vmax - vmin || 1);
      return padTop + (1 - t) * (H - padTop - padBot);
    };

    let start = performance.now();
    function frame(now) {
      const elapsed = (now - start) / 1000;
      // reveal sweeps 0→1 over 2.6s, then holds, loops every ~7s
      const cycle = elapsed % 7;
      const reveal = reduce ? 1 : Math.min(1, cycle / 2.6);
      const shimmer = reduce ? 0 : Math.sin(elapsed * 0.9) * 1.2;

      // ease pointer tilt toward target
      const t = tilt.current;
      t.cx += (t.tx - t.cx) * 0.06;
      t.cy += (t.ty - t.cy) * 0.06;
      wrap.style.transform =
        `perspective(1400px) rotateX(${(-t.cy * 6).toFixed(2)}deg) rotateY(${(t.cx * 10).toFixed(2)}deg)`;

      ctx.clearRect(0, 0, W, H);
      const cut = Math.floor(reveal * (STEPS - 1));

      // p5–p95 envelope
      ctx.beginPath();
      for (let s = 0; s <= cut; s++) ctx.lineTo(X(s), Y(hi[s]));
      for (let s = cut; s >= 0; s--) ctx.lineTo(X(s), Y(lo[s]));
      ctx.closePath();
      const band = ctx.createLinearGradient(0, padTop, 0, H - padBot);
      band.addColorStop(0, "rgba(33,208,173,0.10)");
      band.addColorStop(1, "rgba(27,79,114,0.04)");
      ctx.fillStyle = band;
      ctx.fill();

      // paths
      for (let i = 0; i < paths.length; i++) {
        const p = paths[i];
        ctx.beginPath();
        for (let s = 0; s <= cut; s++) {
          const y = Y(p.pts[s]) + Math.sin(elapsed * 0.8 + i + s * 0.05) * (reduce ? 0 : 0.4);
          if (s === 0) ctx.moveTo(X(s), y); else ctx.lineTo(X(s), y);
        }
        if (p.tail) {
          ctx.strokeStyle = "rgba(216,96,76,0.42)";
          ctx.lineWidth = 1.3;
          ctx.shadowColor = "rgba(216,96,76,0.35)";
          ctx.shadowBlur = 5;
        } else {
          ctx.strokeStyle = `rgba(33,208,173,${0.10 + (i % 5) * 0.03})`;
          ctx.lineWidth = 1;
          ctx.shadowBlur = 0;
        }
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      // median line (gold)
      ctx.beginPath();
      for (let s = 0; s <= cut; s++) {
        const mid = paths.map((p) => p.pts[s]).sort((a, b) => a - b)[Math.floor(paths.length / 2)];
        const y = Y(mid) + shimmer;
        if (s === 0) ctx.moveTo(X(s), y); else ctx.lineTo(X(s), y);
      }
      ctx.strokeStyle = "rgba(240,246,244,0.92)";
      ctx.lineWidth = 1.8;
      ctx.shadowColor = "rgba(33,208,173,0.55)";
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;

      // sweep head
      if (!reduce && reveal < 1) {
        const hx = X(cut);
        const g = ctx.createLinearGradient(hx - 40, 0, hx, 0);
        g.addColorStop(0, "rgba(33,208,173,0)");
        g.addColorStop(1, "rgba(33,208,173,0.5)");
        ctx.strokeStyle = g;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(hx, padTop); ctx.lineTo(hx, H - padBot); ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(frame);
    }
    rafRef.current = requestAnimationFrame(frame);

    function onMove(e) {
      const rect = wrap.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - 0.5;
      const y = (e.clientY - rect.top) / rect.height - 0.5;
      tilt.current.tx = Math.max(-0.6, Math.min(0.6, x));
      tilt.current.ty = Math.max(-0.6, Math.min(0.6, y));
    }
    function onLeave() { tilt.current.tx = 0; tilt.current.ty = 0; }
    const host = wrap.parentElement || wrap;
    host.addEventListener("pointermove", onMove);
    host.addEventListener("pointerleave", onLeave);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
      host.removeEventListener("pointermove", onMove);
      host.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  return (
    <div className="hero-stage">
      <div className="hero-plane" ref={wrapRef}>
        <canvas ref={canvasRef} className="hero-canvas" />
        {/* floating glass readouts at different depths for parallax layering */}
        <div className="hero-chip hero-chip--dd" style={{ "--z": "60px" }}>
          <span className="hero-chip-label">Max drawdown · p95</span>
          <span className="hero-chip-value">−38.4%</span>
        </div>
        <div className="hero-chip hero-chip--tail" style={{ "--z": "90px" }}>
          <span className="hero-chip-label">Tail loss · CVaR₉₅</span>
          <span className="hero-chip-value">−6.1%</span>
        </div>
        <div className="hero-chip hero-chip--frag" style={{ "--z": "40px" }}>
          <span className="hero-chip-label">Model fragility</span>
          <span className="hero-chip-value hero-chip-value--ok">Low · 0.21</span>
        </div>
      </div>
    </div>
  );
}

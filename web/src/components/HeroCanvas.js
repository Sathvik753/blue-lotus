import React, { useEffect, useRef } from "react";

/**
 * Full-bleed Monte-Carlo risk surface. A field of simulated equity paths fans
 * out from a common origin; tail paths dive into drawdown. A shaded p5–p95 band
 * sits behind them, a white median cuts through, and a sweep reveals it. It
 * bleeds edge-to-edge across the viewport and fades into black at the sides —
 * the "product shot", not a boxed chart. Pure canvas, no libraries.
 */

const N_PATHS = 110;
const STEPS = 220;

function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(r) {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function buildPaths() {
  const r = rng(20260914);
  const paths = [];
  for (let i = 0; i < N_PATHS; i++) {
    const tail = i % 12 === 0;
    const drift = tail ? -0.0018 : 0.0005 * (i % 2 ? 1 : -1);
    const vol = tail ? 0.019 : 0.013;
    // start already spread out so there's no thin single origin point
    let w = (r() - 0.5) * 0.5;
    const pts = new Array(STEPS);
    for (let s = 0; s < STEPS; s++) { w += drift + vol * gauss(r); pts[s] = w; }
    paths.push({ pts, tail });
  }
  return paths;
}

export default function HeroCanvas() {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const canvas = canvasRef.current;
    const wrap = canvas && canvas.parentElement;
    if (!canvas || !wrap) return;
    const ctx = canvas.getContext("2d");
    const paths = buildPaths();

    const lo = new Array(STEPS), hi = new Array(STEPS), mid = new Array(STEPS);
    for (let s = 0; s < STEPS; s++) {
      const col = paths.map((p) => p.pts[s]).sort((a, b) => a - b);
      lo[s] = col[Math.floor(col.length * 0.05)];
      hi[s] = col[Math.floor(col.length * 0.95)];
      mid[s] = col[Math.floor(col.length / 2)];
    }
    let vmin = Infinity, vmax = -Infinity;
    for (let s = 0; s < STEPS; s++) for (const p of paths) {
      if (p.pts[s] < vmin) vmin = p.pts[s];
      if (p.pts[s] > vmax) vmax = p.pts[s];
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

    // Bleed the field beyond all four edges so it fills the viewport — no
    // thin origin, no empty bands top or bottom.
    const padY = -H * 0.08;
    const X = (s) => -0.18 * W + (s / (STEPS - 1)) * 1.36 * W;
    const Y = (v) => {
      const t = (v - vmin) / (vmax - vmin || 1);
      return padY + (1 - t) * (H - padY * 2);
    };

    const start = performance.now();
    function frame(now) {
      const elapsed = (now - start) / 1000;
      const cycle = elapsed % 9;
      const reveal = reduce ? 1 : Math.min(1, cycle / 3.0);
      const shimmer = reduce ? 0 : Math.sin(elapsed * 0.9) * 1.2;

      ctx.clearRect(0, 0, W, H);
      const cut = Math.floor(reveal * (STEPS - 1));

      // envelope
      ctx.beginPath();
      for (let s = 0; s <= cut; s++) ctx.lineTo(X(s), Y(hi[s]));
      for (let s = cut; s >= 0; s--) ctx.lineTo(X(s), Y(lo[s]));
      ctx.closePath();
      const band = ctx.createLinearGradient(0, padY, 0, H - padY);
      band.addColorStop(0, "rgba(226,199,195,0.10)");
      band.addColorStop(1, "rgba(27,58,92,0.05)");
      ctx.fillStyle = band;
      ctx.fill();

      // paths
      for (let i = 0; i < paths.length; i++) {
        const p = paths[i];
        ctx.beginPath();
        for (let s = 0; s <= cut; s++) {
          const y = Y(p.pts[s]) + (reduce ? 0 : Math.sin(elapsed * 0.8 + i + s * 0.05) * 0.4);
          if (s === 0) ctx.moveTo(X(s), y); else ctx.lineTo(X(s), y);
        }
        if (p.tail) {
          ctx.strokeStyle = "rgba(216,96,76,0.40)";
          ctx.lineWidth = 1.3; ctx.shadowColor = "rgba(216,96,76,0.35)"; ctx.shadowBlur = 5;
        } else {
          ctx.strokeStyle = `rgba(226,199,195,${0.09 + (i % 5) * 0.028})`;
          ctx.lineWidth = 1; ctx.shadowBlur = 0;
        }
        ctx.stroke();
      }
      ctx.shadowBlur = 0;

      // median
      ctx.beginPath();
      for (let s = 0; s <= cut; s++) {
        const y = Y(mid[s]) + shimmer;
        if (s === 0) ctx.moveTo(X(s), y); else ctx.lineTo(X(s), y);
      }
      ctx.strokeStyle = "rgba(250,250,237,0.92)";
      ctx.lineWidth = 2; ctx.shadowColor = "rgba(242,193,78,0.5)"; ctx.shadowBlur = 9;
      ctx.stroke();
      ctx.shadowBlur = 0;

      if (!reduce && reveal < 1) {
        const hx = X(cut);
        const g = ctx.createLinearGradient(hx - 44, 0, hx, 0);
        g.addColorStop(0, "rgba(242,193,78,0)");
        g.addColorStop(1, "rgba(242,193,78,0.5)");
        ctx.strokeStyle = g; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(hx, padY); ctx.lineTo(hx, H - padY); ctx.stroke();
      }

      rafRef.current = requestAnimationFrame(frame);
    }
    rafRef.current = requestAnimationFrame(frame);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="hero-canvas" />;
}

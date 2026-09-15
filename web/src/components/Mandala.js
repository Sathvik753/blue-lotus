import React, { useEffect, useRef } from "react";

/**
 * A geometric lotus mandala rendered as line art, fixed behind the whole page.
 * It rotates as you scroll — simple, calm, and on-brand for "Blue Lotus".
 *
 * Two overlaid rhodonea (rose) curves plus a Maurer-rose web give the dense
 * radiating petals; concentric rings frame it. Pure SVG, no libraries.
 */

const R = 100;

// Smooth rose r = sin(n·θ) traced continuously → clean petal outlines.
function smoothRose(n, steps = 2000) {
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * Math.PI * 2;
    const r = Math.sin(n * t) * R;
    pts.push([r * Math.cos(t), r * Math.sin(t)]);
  }
  return pts;
}

// Maurer rose: sample the rose at whole-degree steps of `d` and connect them.
function maurerRose(n, d) {
  const pts = [];
  for (let i = 0; i <= 360; i++) {
    const k = (i * d * Math.PI) / 180;
    const r = Math.sin(n * k) * R;
    pts.push([r * Math.cos(k), r * Math.sin(k)]);
  }
  return pts;
}

const toPath = (pts) =>
  "M" + pts.map((p) => `${p[0].toFixed(2)} ${p[1].toFixed(2)}`).join(" L ");

export default function Mandala() {
  const ref = useRef(null);

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) return;
    let raf = 0;
    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        if (ref.current) ref.current.style.setProperty("--rot", `${window.scrollY * 0.05}deg`);
      });
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => { window.removeEventListener("scroll", onScroll); cancelAnimationFrame(raf); };
  }, []);

  const petals6 = toPath(smoothRose(6));
  const petals12 = toPath(smoothRose(12));
  const web = toPath(maurerRose(6, 71));

  return (
    <div className="mandala" aria-hidden="true">
      <svg ref={ref} className="mandala-svg" viewBox="-112 -112 224 224">
        <g>
          <circle className="mandala-ring" cx="0" cy="0" r="100" />
          <circle className="mandala-ring" cx="0" cy="0" r="82" />
          <path className="mandala-web" d={web} />
          <path className="mandala-petals" d={petals6} />
          <path className="mandala-petals mandala-petals--soft" d={petals12} />
        </g>
      </svg>
    </div>
  );
}

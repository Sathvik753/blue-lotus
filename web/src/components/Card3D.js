import React, { useRef } from "react";

/**
 * A surface with real depth: it tilts toward the pointer, a soft light tracks
 * the cursor across its face, and the content sits on a plane raised above the
 * card so it parallaxes as the card turns. This is the difference between an
 * Apple-style object and a flat rounded rectangle.
 *
 * Wrap it in a `.reveal` element for scroll-entrance — keep the entrance
 * transform off this node so it never fights the tilt transform.
 */
export default function Card3D({ className = "", children, tilt = 7, lift = 26, style, ...rest }) {
  const ref = useRef(null);
  const raf = useRef(0);

  function apply(mx, my, rx, ry) {
    const el = ref.current;
    if (!el) return;
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(() => {
      el.style.setProperty("--mx", mx);
      el.style.setProperty("--my", my);
      el.style.setProperty("--rx", rx);
      el.style.setProperty("--ry", ry);
    });
  }

  function onMove(e) {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width;
    const py = (e.clientY - r.top) / r.height;
    apply(`${px * 100}%`, `${py * 100}%`, `${(0.5 - py) * tilt}deg`, `${(px - 0.5) * tilt}deg`);
  }
  function onLeave() {
    apply("50%", "-10%", "0deg", "0deg");
  }

  return (
    <div
      ref={ref}
      className={`card3d ${className}`}
      style={{ "--lift": `${lift}px`, ...style }}
      onPointerMove={onMove}
      onPointerLeave={onLeave}
      {...rest}
    >
      <span className="card3d-spot" aria-hidden="true" />
      <span className="card3d-edge" aria-hidden="true" />
      <div className="card3d-body">{children}</div>
    </div>
  );
}

import React, { useId } from "react";

// A petal pointing straight up, anchored at the flower base (100, 150).
function petal(height, width) {
  const baseX = 100, baseY = 150;
  const tipY = baseY - height;
  const c1y = baseY - height * 0.32;
  const c2y = baseY - height * 0.82;
  const c2x = width * 0.5;
  return [
    `M${baseX} ${baseY}`,
    `C${baseX - width} ${c1y} ${baseX - c2x} ${c2y} ${baseX} ${tipY}`,
    `C${baseX + c2x} ${c2y} ${baseX + width} ${c1y} ${baseX} ${baseY}`,
    "Z",
  ].join(" ");
}

// Back, mid, and front petal rings — drawn furthest-back first.
const PETALS = [
  { d: petal(74, 30), rot: -78, layer: "back" },
  { d: petal(74, 30), rot: 78, layer: "back" },
  { d: petal(82, 30), rot: -52, layer: "back" },
  { d: petal(82, 30), rot: 52, layer: "back" },
  { d: petal(90, 27), rot: -27, layer: "mid" },
  { d: petal(90, 27), rot: 27, layer: "mid" },
  { d: petal(100, 27), rot: 0, layer: "front" },
];

export default function Logo({ size = 40, animated = true }) {
  const id = useId().replace(/:/g, "");
  const front = `${id}-front`;
  const mid = `${id}-mid`;
  const back = `${id}-back`;
  const halo = `${id}-halo`;
  const stroke = `${id}-stroke`;

  const fillFor = { back: `url(#${back})`, mid: `url(#${mid})`, front: `url(#${front})` };

  return (
    <svg width={size} height={size} viewBox="0 0 200 200"
      fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Blue Lotus">
      <defs>
        <linearGradient id={back} x1="100" y1="150" x2="100" y2="55" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#0F2B45" />
          <stop offset="1" stopColor="#1B4F72" />
        </linearGradient>
        <linearGradient id={mid} x1="100" y1="150" x2="100" y2="50" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#127A66" />
          <stop offset="1" stopColor="#21D0AD" />
        </linearGradient>
        <linearGradient id={front} x1="100" y1="150" x2="100" y2="50" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#18A88B" />
          <stop offset="0.55" stopColor="#D4AC0D" />
          <stop offset="1" stopColor="#F2CB3A" />
        </linearGradient>
        <linearGradient id={stroke} x1="100" y1="150" x2="100" y2="55" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="rgba(255,255,255,0)" />
          <stop offset="1" stopColor="rgba(255,255,255,0.55)" />
        </linearGradient>
        <radialGradient id={halo} cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor="#21D0AD" stopOpacity="0.55" />
          <stop offset="0.55" stopColor="#1B4F72" stopOpacity="0.25" />
          <stop offset="1" stopColor="#1B4F72" stopOpacity="0" />
        </radialGradient>
      </defs>

      <circle className={animated ? "lotus-glow" : undefined}
        cx="100" cy="118" r="74" fill={`url(#${halo})`} />

      <g className={animated ? "lotus-petals" : undefined}>
        {PETALS.map((p, i) => (
          <path key={i} d={p.d} transform={`rotate(${p.rot} 100 150)`}
            fill={fillFor[p.layer]}
            stroke={p.layer === "front" ? `url(#${stroke})` : "none"}
            strokeWidth={p.layer === "front" ? 1.2 : 0}
            opacity={p.layer === "back" ? 0.92 : 1} />
        ))}
        {/* seed of light at the heart of the flower */}
        <circle cx="100" cy="138" r="7" fill="#F7E08A" opacity="0.9" />
        <circle cx="100" cy="138" r="3" fill="#FFFDF2" />
      </g>
    </svg>
  );
}

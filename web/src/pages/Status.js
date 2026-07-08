import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../utils/api";
import Logo from "../components/Logo";

const COLORS = {
  operational: "var(--teal-2)",
  degraded: "var(--gold)",
  down: "var(--rose)",
};
const LABEL = {
  operational: "Operational",
  degraded: "Degraded",
  down: "Outage",
};

export default function Status() {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    api.get("/status")
      .then(r => r.ok ? setData(r.data) : setErr(true))
      .catch(() => setErr(true));
  }, []);

  const overall = err ? "down" : data?.status || "operational";

  return (
    <div className="fade-in" style={{ maxWidth: 640, margin: "0 auto", padding: "40px 24px" }}>
      <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 40 }}>
        <Link to="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <Logo size={34} />
          <span className="gradient-text" style={{ fontFamily: "Syne, sans-serif", fontWeight: 800 }}>Blue Lotus</span>
        </Link>
        <Link to="/" style={{ color: "var(--muted)", fontSize: 13 }}>← Home</Link>
      </header>

      <div className="card" style={{ marginBottom: 18, borderColor: `${COLORS[overall]}55` }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{
            width: 12, height: 12, borderRadius: "50%", background: COLORS[overall],
            boxShadow: `0 0 12px ${COLORS[overall]}`,
          }} />
          <span style={{ fontSize: 20, fontFamily: "Syne, sans-serif", fontWeight: 700 }}>
            {err ? "Cannot reach API" : overall === "operational" ? "All systems operational" : "Partial degradation"}
          </span>
        </div>
      </div>

      {data && (
        <div className="card">
          {data.components.map((c, i) => (
            <div key={c.name} style={{
              display: "flex", alignItems: "center", justifyContent: "space-between",
              padding: "14px 0", borderBottom: i < data.components.length - 1 ? "1px solid var(--border-soft)" : "none",
            }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: 14 }}>{c.name}</div>
                {c.detail && <div style={{ fontSize: 11.5, color: "var(--muted)", marginTop: 2 }}>{c.detail}</div>}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, color: COLORS[c.status], fontSize: 13, fontWeight: 600 }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: COLORS[c.status] }} />
                {LABEL[c.status]}
              </div>
            </div>
          ))}
          <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 16 }}>
            Version {data.version} · updated {new Date(data.time).toLocaleTimeString()}
          </div>
        </div>
      )}
    </div>
  );
}

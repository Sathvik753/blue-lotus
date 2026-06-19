import React from "react";
import { Outlet, NavLink } from "react-router-dom";
import { Clock, BarChart2, Zap } from "lucide-react";
import Logo from "./Logo";

const NAV = [
  { to: "/run", icon: Zap, label: "New Run" },
  { to: "/history", icon: Clock, label: "History" },
  { to: "/compare", icon: BarChart2, label: "Compare" },
];

const SIDEBAR_W = 244;

export default function Layout() {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <aside style={{
        width: SIDEBAR_W, flexShrink: 0, position: "fixed", top: 0, left: 0, height: "100vh",
        display: "flex", flexDirection: "column", padding: "28px 0",
        background: "linear-gradient(180deg, rgba(15,28,43,0.92), rgba(6,11,20,0.92))",
        backdropFilter: "blur(18px)", WebkitBackdropFilter: "blur(18px)",
        borderRight: "1px solid var(--border-soft)",
      }}>
        {/* Brand */}
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "0 22px 30px" }}>
          <Logo size={42} />
          <div>
            <div style={{
              fontFamily: "Syne, sans-serif", fontWeight: 800, fontSize: 17,
              letterSpacing: "-0.02em", lineHeight: 1,
            }} className="gradient-text">
              Blue Lotus
            </div>
            <div style={{
              fontSize: 9.5, color: "var(--muted)", letterSpacing: "0.32em",
              textTransform: "uppercase", marginTop: 4,
            }}>
              Labs · Risk
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: "0 14px" }}>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} style={({ isActive }) => ({
              position: "relative",
              display: "flex", alignItems: "center", gap: 12,
              padding: "12px 14px", borderRadius: 11, marginBottom: 4,
              color: isActive ? "var(--white)" : "var(--muted)",
              background: isActive
                ? "linear-gradient(100deg, rgba(212,172,13,0.16), rgba(33,208,173,0.06))"
                : "transparent",
              boxShadow: isActive ? "inset 0 0 0 1px rgba(212,172,13,0.25)" : "none",
              textDecoration: "none", fontSize: 13.5, fontWeight: 600,
              fontFamily: "Syne, sans-serif", letterSpacing: "0.01em",
              transition: "all 0.18s ease",
            })}>
              {({ isActive }) => (
                <>
                  {isActive && (
                    <span style={{
                      position: "absolute", left: -14, top: "50%", transform: "translateY(-50%)",
                      width: 3, height: 22, borderRadius: 3,
                      background: "linear-gradient(var(--gold), var(--teal-2))",
                      boxShadow: "0 0 10px var(--glow-gold)",
                    }} />
                  )}
                  <Icon size={17} color={isActive ? "var(--gold)" : "currentColor"} />
                  {label}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Footer status */}
        <div style={{ padding: "16px 22px 0", borderTop: "1px solid var(--border-soft)", margin: "0 8px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 11, color: "var(--muted)" }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%", background: "var(--teal-2)",
              boxShadow: "0 0 8px var(--teal-2)",
            }} className="status-pill" />
            Engine online
          </div>
          <div style={{ fontSize: 10, color: "var(--muted)", marginTop: 8, opacity: 0.6, letterSpacing: "0.04em" }}>
            v1.0 · Monte Carlo · EVT
          </div>
        </div>
      </aside>

      <main style={{
        marginLeft: SIDEBAR_W, flex: 1, padding: "44px 48px", minHeight: "100vh",
        maxWidth: 1280,
      }}>
        <Outlet />
      </main>
    </div>
  );
}

import React from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { Clock, BarChart2, Zap, CreditCard, Terminal, LogOut } from "lucide-react";
import Logo from "./Logo";
import { useAuth } from "../context/Auth";

const NAV = [
  { to: "/run", icon: Zap, label: "New Run" },
  { to: "/history", icon: Clock, label: "History" },
  { to: "/compare", icon: BarChart2, label: "Compare" },
  { to: "/billing", icon: CreditCard, label: "Billing" },
];

const SIDEBAR_W = 244;

function navStyle({ isActive }) {
  return {
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
  };
}

function NavItem({ to, icon: Icon, label }) {
  return (
    <NavLink to={to} style={navStyle}>
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
  );
}

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  function handleLogout() {
    logout();
    navigate("/login", { replace: true });
  }

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
          {NAV.map(item => <NavItem key={item.to} {...item} />)}
          {user?.is_developer && <NavItem to="/developer" icon={Terminal} label="Developer" />}
        </nav>

        {/* Account */}
        <div style={{ padding: "16px 18px 0", borderTop: "1px solid var(--border-soft)", margin: "0 8px" }}>
          {user && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12.5, fontWeight: 600, color: "var(--light)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {user.org?.name || user.email}
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 3 }}>
                <span style={{
                  fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.08em",
                  color: "var(--gold)", background: "rgba(212,172,13,0.12)",
                  padding: "2px 7px", borderRadius: 999, fontWeight: 700,
                }}>{user.plan}</span>
                <span style={{ fontSize: 10.5, color: "var(--muted)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {user.email}
                </span>
              </div>
            </div>
          )}
          <button onClick={handleLogout} style={{
            display: "flex", alignItems: "center", gap: 8, width: "100%",
            background: "none", border: "none", cursor: "pointer",
            color: "var(--muted)", fontSize: 12.5, padding: "6px 0", fontFamily: "Syne, sans-serif", fontWeight: 600,
          }}>
            <LogOut size={14} /> Sign out
          </button>
        </div>
      </aside>

      <main style={{
        marginLeft: SIDEBAR_W, flex: 1, padding: "44px 48px", minHeight: "100vh",
        maxWidth: 1280, display: "flex", flexDirection: "column",
      }}>
        <div style={{ flex: 1 }}><Outlet /></div>
        <div style={{ marginTop: 40, paddingTop: 16, borderTop: "1px solid var(--border-soft)",
          fontSize: 11, color: "var(--muted)", lineHeight: 1.6 }}>
          Outputs are probabilistic model estimates conditional on their assumptions — informational only, not
          investment advice, and not a guarantee of any result.{" "}
          <a href="/disclaimer" style={{ color: "var(--muted)", textDecoration: "underline" }}>Disclaimer</a>
          {" · "}
          <a href="/terms" style={{ color: "var(--muted)", textDecoration: "underline" }}>Terms</a>
          {" · "}
          <a href="/privacy" style={{ color: "var(--muted)", textDecoration: "underline" }}>Privacy</a>
        </div>
      </main>
    </div>
  );
}

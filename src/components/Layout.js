import React from "react";
import { Outlet, NavLink, useNavigate } from "react-router-dom";
import { api } from "../utils/api";
import {
  Activity, Clock, BarChart2, Key, LogOut, Zap
} from "lucide-react";

const NAV = [
  { to: "/run",     icon: Zap,       label: "New Run"   },
  { to: "/history", icon: Clock,     label: "History"   },
  { to: "/compare", icon: BarChart2, label: "Compare"   },
  { to: "/keys",    icon: Key,       label: "API Keys"  },
];

export default function Layout() {
  const navigate = useNavigate();
  const user = api.getUser();

  function logout() {
    api.logout();
    navigate("/login");
  }

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      {/* Sidebar */}
      <aside style={{
        width: 220, background: "var(--dark)", borderRight: "1px solid var(--border)",
        display: "flex", flexDirection: "column", padding: "24px 0", flexShrink: 0,
        position: "fixed", top: 0, left: 0, height: "100vh",
      }}>
        {/* Logo */}
        <div style={{ padding: "0 20px 32px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <Activity size={20} color="var(--gold)" />
            <span style={{
              fontFamily: "Syne, sans-serif", fontWeight: 800, fontSize: 16,
              color: "var(--gold)", letterSpacing: "-0.02em",
            }}>
              Blue Lotus
            </span>
          </div>
          <div style={{ fontSize: 10, color: "var(--muted)", letterSpacing: "0.1em", textTransform: "uppercase", paddingLeft: 30 }}>
            Labs
          </div>
        </div>

        {/* Nav links */}
        <nav style={{ flex: 1, padding: "0 12px" }}>
          {NAV.map(({ to, icon: Icon, label }) => (
            <NavLink key={to} to={to} style={({ isActive }) => ({
              display: "flex", alignItems: "center", gap: 10,
              padding: "10px 12px", borderRadius: 8, marginBottom: 2,
              color: isActive ? "var(--gold)" : "var(--muted)",
              background: isActive ? "rgba(212,172,13,0.08)" : "transparent",
              textDecoration: "none", fontSize: 13, fontWeight: 500,
              transition: "all 0.15s",
            })}>
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* User + logout */}
        <div style={{ padding: "16px 20px", borderTop: "1px solid var(--border)" }}>
          <div style={{ fontSize: 11, color: "var(--muted)", marginBottom: 4, letterSpacing: "0.06em" }}>
            {user?.email}
          </div>
          <div style={{
            display: "inline-block", background: "rgba(212,172,13,0.15)",
            color: "var(--gold)", fontSize: 10, fontWeight: 600,
            padding: "2px 8px", borderRadius: 20, letterSpacing: "0.08em",
            textTransform: "uppercase", marginBottom: 12,
          }}>
            {user?.plan || "free"}
          </div>
          <button onClick={logout} className="btn btn-secondary" style={{
            width: "100%", display: "flex", alignItems: "center",
            gap: 8, justifyContent: "center", fontSize: 12,
          }}>
            <LogOut size={14} /> Logout
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main style={{ marginLeft: 220, flex: 1, padding: "40px", minHeight: "100vh" }}>
        <Outlet />
      </main>
    </div>
  );
}

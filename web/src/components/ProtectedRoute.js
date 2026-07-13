import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/Auth";

function Splash({ label }) {
  return (
    <div style={{
      minHeight: "100vh", display: "flex", alignItems: "center",
      justifyContent: "center", color: "var(--muted)", gap: 12,
    }}>
      <span className="spinner" /> {label}
    </div>
  );
}

// Requires a signed-in user; otherwise bounces to /login (remembering where).
export function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  const location = useLocation();
  if (loading) return <Splash label="Loading workspace…" />;
  if (!user) return <Navigate to="/login" replace state={{ from: location.pathname }} />;
  return children;
}

// Requires sign-in; the Developer page itself handles the unlock flow for
// accounts that don't have the role yet.
export function DeveloperRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <Splash label="Loading…" />;
  if (!user) return <Navigate to="/login" replace />;
  return children;
}

import React from "react";

/* Catches render-time errors anywhere below it so a single crash shows a
   friendly message instead of blanking the whole app to the background. */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error("Render error:", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={{
        minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 24,
      }}>
        <div className="card" style={{ maxWidth: 440, textAlign: "center" }}>
          <h1 style={{ fontSize: 20, marginBottom: 8 }}>Something went wrong</h1>
          <p style={{ color: "var(--muted)", fontSize: 14, marginBottom: 20 }}>
            This page hit an unexpected error. Reloading usually fixes it.
          </p>
          <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
            <button className="btn btn-primary" onClick={() => window.location.reload()}>Reload</button>
            <button className="btn btn-secondary" onClick={() => { window.location.href = "/run"; }}>Go to app</button>
          </div>
        </div>
      </div>
    );
  }
}

const BASE = process.env.REACT_APP_API_BASE || "";

export const API_BASE = BASE;

const TOKEN_KEY = "bl_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}
export function setToken(t) {
  if (t) localStorage.setItem(TOKEN_KEY, t);
  else localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(extra = {}) {
  const t = getToken();
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

async function request(method, path, body = null) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: body ? JSON.stringify(body) : null,
  });

  // A 401 on an authenticated call means the session is gone — drop the token.
  if (res.status === 401 && getToken()) {
    setToken(null);
    if (!window.location.pathname.startsWith("/login")) {
      window.location.href = "/login";
    }
  }

  const text = await res.text();
  try {
    return { ok: res.ok, status: res.status, data: JSON.parse(text) };
  } catch {
    return { ok: res.ok, status: res.status, data: text };
  }
}

// OAuth2 password login uses form-encoding, not JSON.
async function login(email, password) {
  const form = new URLSearchParams();
  form.append("username", email);
  form.append("password", password);
  const res = await fetch(`${BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form.toString(),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Login failed.");
  setToken(data.access_token);
  return data;
}

async function register(payload) {
  const res = await fetch(`${BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(Array.isArray(d) ? d.map(e => e.msg).join("; ") : (d || "Registration failed."));
  }
  setToken(data.access_token);
  return data;
}

function logout() {
  setToken(null);
}

// Trigger a browser download from a binary/authorized endpoint (PDF, JSON).
async function download(path, fallbackName) {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Download failed (${res.status})`);
  const blob = await res.blob();
  const disp = res.headers.get("content-disposition") || "";
  const match = disp.match(/filename="?([^"]+)"?/);
  const name = match ? match[1] : fallbackName;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export const api = {
  get: (path) => request("GET", path),
  post: (path, body) => request("POST", path, body),
  delete: (path) => request("DELETE", path),
  login,
  register,
  logout,
  download,

  async pollRun(runId, onStatus, maxWait = 120) {
    for (let i = 0; i < maxWait; i++) {
      await new Promise(r => setTimeout(r, 1500));
      const res = await request("GET", `/run/${runId}`);
      if (!res) return null;
      onStatus(res.data.status, i);
      if (res.data.status === "completed") return res.data;
      if (res.data.status === "failed") throw new Error(res.data.error_msg || "Run failed");
    }
    throw new Error("Timed out waiting for results");
  },
};

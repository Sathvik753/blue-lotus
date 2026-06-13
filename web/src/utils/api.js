const BASE = process.env.REACT_APP_API_BASE || "";

export const API_BASE = BASE;

function getToken() {
  return localStorage.getItem("bl_token");
}

function getApiKey() {
  return localStorage.getItem("bl_apikey");
}

async function request(method, path, body = null) {
  const headers = { "Content-Type": "application/json" };
  const token = getToken();
  const apiKey = getApiKey();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (apiKey) headers["X-API-Key"] = apiKey;

  const res = await fetch(`${BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : null,
  });

  if (res.status === 401) {
    localStorage.removeItem("bl_token");
    window.location.href = "/login";
    return null;
  }

  const text = await res.text();
  try {
    return { ok: res.ok, status: res.status, data: JSON.parse(text) };
  } catch {
    return { ok: res.ok, status: res.status, data: text };
  }
}

export const api = {
  get: (path) => request("GET", path),
  post: (path, body) => request("POST", path, body),
  delete: (path) => request("DELETE", path),

  async login(email, password) {
    const form = new URLSearchParams({ username: email, password });
    const res = await fetch(`${BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form,
    });
    const data = await res.json();
    if (res.ok) {
      localStorage.setItem("bl_token", data.access_token);
      localStorage.setItem("bl_user", JSON.stringify(data));
    }
    return { ok: res.ok, data };
  },

  async register(email, password, name) {
    const res = await request("POST", "/auth/register", { email, password, name });
    if (res?.ok) {
      localStorage.setItem("bl_token", res.data.access_token);
      localStorage.setItem("bl_user", JSON.stringify(res.data));
    }
    return res;
  },

  logout() {
    localStorage.removeItem("bl_token");
    localStorage.removeItem("bl_user");
    localStorage.removeItem("bl_apikey");
  },

  getUser() {
    try { return JSON.parse(localStorage.getItem("bl_user")); } catch { return null; }
  },

  isLoggedIn() {
    return !!getToken();
  },

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
  }
};

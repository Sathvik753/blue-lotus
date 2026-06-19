const BASE = process.env.REACT_APP_API_BASE || "";

export const API_BASE = BASE;

async function request(method, path, body = null) {
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  const text = await res.text();
  try {
    return { ok: res.ok, status: res.status, data: JSON.parse(text) };
  } catch {
    return { ok: res.ok, status: res.status, data: text };
  }
}

// Trigger a browser download from a binary endpoint (PDF, etc.).
async function download(path, fallbackName) {
  const res = await fetch(`${BASE}${path}`);
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
  }
};

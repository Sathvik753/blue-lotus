# 🌸 Blue Lotus Labs — Stress-Testing Engine

Constraint-driven Monte Carlo stress-testing for financial strategies.

## Stack

| Layer    | Tech                        |
|----------|-----------------------------|
| Engine   | Python (NumPy, SciPy)       |
| API      | FastAPI + asyncpg           |
| Database | PostgreSQL (SQLAlchemy)     |
| Frontend | Streamlit                   |
| Auth     | JWT + API Keys (bcrypt)     |
| Deploy   | Docker + Railway/Render     |

---

## Local Development (Quickstart)

### 1. Prerequisites
- Docker + Docker Compose installed
- Python 3.11+

### 2. Clone and configure
```bash
git clone https://github.com/YOUR_USERNAME/blue-lotus.git
cd blue-lotus
cp .env.example .env
# Edit .env — generate SECRET_KEY with: openssl rand -hex 32
```

### 3. Run everything with Docker
```bash
docker-compose up --build
```

This starts:
- PostgreSQL on port `5432`
- FastAPI on `http://localhost:8000`
- Streamlit on `http://localhost:8501`

### 4. Open the app
- **Frontend**: http://localhost:8501
- **API docs**: http://localhost:8000/docs

---

## Running Without Docker (dev mode)

### API
```bash
pip install -r requirements.api.txt

# Set env vars
export DATABASE_URL=postgresql+asyncpg://postgres:password@localhost:5432/bluelotus
export SECRET_KEY=your-secret-key

uvicorn api.main:app --reload --port 8000
```

### Frontend
```bash
pip install -r requirements.frontend.txt
export API_URL=http://localhost:8000
streamlit run frontend/app.py
```

---

## API Reference

All endpoints require authentication via either:
- `Authorization: Bearer <jwt_token>` header
- `X-API-Key: bl_xxxxx` header

### Auth
```
POST /auth/register     Create account
POST /auth/login        Get JWT token
GET  /auth/me           Current user info
POST /auth/api-keys     Generate API key
GET  /auth/api-keys     List API keys
```

### Runs
```
POST /run/ticker        Start run on Yahoo Finance ticker (async)
POST /run/custom        Start run on custom return series (async)
GET  /run/{run_id}      Get status + results
GET  /runs              Paginated run history
```

### Analysis
```
POST /compare           Side-by-side multi-ticker comparison (sync)
```

### Example: Run SPY stress test via API
```python
import requests, time

BASE    = "http://localhost:8000"
API_KEY = "bl_your_key_here"
HEADERS = {"X-API-Key": API_KEY}

# Start run
r = requests.post(f"{BASE}/run/ticker",
    headers=HEADERS,
    json={"ticker": "SPY", "start_date": "2010-01-01",
          "n_paths": 10000, "horizon": 252}
)
run_id = r.json()["run_id"]
print(f"Run started: {run_id}")

# Poll until complete
while True:
    r = requests.get(f"{BASE}/run/{run_id}", headers=HEADERS)
    data = r.json()
    print(f"Status: {data['status']}")
    if data["status"] == "completed":
        result = data["result"]
        print(f"Mean max drawdown : {result['drawdown']['mean']:.4f}")
        print(f"Aggregate ES (5%) : {result['expected_shortfall']['aggregate']:.4f}")
        print(f"Fragility Index   : {result['fragility']['index']:.4f} ({result['fragility']['grade']})")
        break
    if data["status"] == "failed":
        print("Run failed:", data.get("error_msg"))
        break
    time.sleep(2)
```

---

## Deployment on Railway (recommended)

### 1. Push to GitHub
```bash
git init
git add .
git commit -m "Initial Blue Lotus deployment"
git remote add origin https://github.com/YOUR_USERNAME/blue-lotus.git
git push -u origin main
```

### 2. Deploy on Railway
1. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
2. Select your repo
3. Add a **PostgreSQL** plugin — Railway auto-sets `DATABASE_URL`
4. Set environment variables in Railway dashboard:
   - `SECRET_KEY` → output of `openssl rand -hex 32`
   - `ENVIRONMENT` → `production`
5. Railway auto-detects `Dockerfile.api` and deploys

### 3. Deploy Streamlit frontend
- Deploy as a second Railway service from the same repo
- Set `API_URL` to your Railway API service URL
- Start command: `streamlit run frontend/app.py --server.port $PORT`

### Alternative: Render
- Same process — connect GitHub repo, set env vars, deploy
- Use render.yaml for one-click multi-service deploy

---

## Project Structure

```
blue-lotus/
├── engine/
│   ├── core.py           ← Full stress-testing engine (all 7 modules)
│   └── serializer.py     ← numpy → JSON conversion
├── api/
│   ├── main.py           ← FastAPI app + all routes
│   ├── auth.py           ← JWT + API key auth
│   ├── schemas.py        ← Pydantic request/response models
│   └── jobs.py           ← Background job runner
├── db/
│   ├── models.py         ← SQLAlchemy ORM models
│   └── database.py       ← Async DB connection + session
├── frontend/
│   └── app.py            ← Streamlit UI
├── reports/
│   └── pdf.py            ← PDF report generator
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.frontend
├── requirements.api.txt
├── requirements.frontend.txt
├── .env.example
└── README.md
```

---

## Roadmap

- [ ] Portfolio-level stress testing (multi-asset)
- [ ] Historical scenario replay (2008, COVID, dot-com)
- [ ] Benchmark comparison (vs SPY, 60/40)
- [ ] Real-time data feeds (Bloomberg/Refinitiv)
- [ ] White-label PDF reports
- [ ] Supply chain optimization module
- [ ] Regulatory stress scenarios (Basel III)

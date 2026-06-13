FROM python:3.11-slim

WORKDIR /app

COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

COPY engine/ ./engine/
COPY api/    ./api/
COPY db/     ./db/
COPY reports/ ./reports/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

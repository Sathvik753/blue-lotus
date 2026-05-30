# ── Stage 1: Build React app ──────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm install

COPY public/ ./public/
COPY src/     ./src/

# REACT_APP_API_BASE is injected at build time as a build arg.
# Cloud Build / gcloud run deploy will pass this in.
ARG REACT_APP_API_BASE=""
ENV REACT_APP_API_BASE=$REACT_APP_API_BASE

RUN npm run build

# ── Stage 2: Serve with nginx ─────────────────────────────────────
FROM nginx:1.27-alpine

# Remove default nginx config and replace with SPA-friendly one
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/build /usr/share/nginx/html

# Cloud Run injects PORT; nginx listens on it
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]

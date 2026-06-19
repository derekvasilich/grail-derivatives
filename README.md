# 🏛️ Grail Derivatives: Black-Scholes HPC FDM Pricing Engine

[![AWS Certified Solutions Architect - Associate](https://shields.io/badge/AWS-Certified_Solutions_Architect_Associate-orange?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://www.credly.com/badges/0bba41aa-57e1-499d-8077-245626cc4d1d/public_url)
[![Security Vetted](https://shields.io/badge/Security_Vetted-GoC_Secret_Level_II-red?style=for-the-badge)](#)
[![License: Proprietary](https://shields.io/badge/License-Proprietary_Closed_Source-darkred?style=for-the-badge)](#)


A production-grade, high-availability Financial Engineering microservice designed to ingest high-volume derivative option contracts and orchestrate intensive mathematical valuations under extreme concurrency. This platform serves as a modern, secure cloud-native REST API wrapper executing stateless token rotation, validation, and data serialization over an elite, high-performance computing numerical matrix engine.

> 🔒 Intellectual Property Boundary: This public repository contains the complete full-stack Python/FastAPI application layer, schema validations, OpenAPI specifications, and integration test suites. The underlying high-performance execution loops, multi-core OpenMP parallelization models, and O(N) tridiagonal numerical matrix solvers are maintained inside a proprietary, closed-source compiled machine-code shared library (.so) belonging exclusively to Dealer Gears Inc.

## 🚀 Core Architectural Moats
### 1. High-Performance FinOps & Zero-Copy Execution
Traditional JSON parsing is an extreme performance bottleneck for real-time trading desks and risk modeling engines. This API introduces a specialized binary ingress pipe:

* POST `/v1/pricing/binary`: Bypasses JSON serialization entirely. It accepts a raw, contiguous byte stream representing native C/C++ memory arrays (OptionConfig structs) and maps them directly to the underlying engine via pointer passing, achieving absolute zero-copy execution speed.

* POST` /v1/pricing/grid`: Quant Grid Engine computes the entire 2D asset-time pricing surface and streams it back to the client instantly as a raw binary float64 array, completely preventing browser memory freezing and minimizing network transfer overhead.

### 2. Quantitative Matrix Math & Numerical Stability
The platform supports Vanilla (European) contracts alongside highly path-dependent Exotic instruments (American, Bermudan, Single Barrier, and Double Barrier options).

* Crank-Nicolson Discretization: Solves continuous Black-Scholes partial differential equations (PDEs) over structured 2D asset-time meshes.

* Rannacher Damping: Switches to fully implicit stencils for the initial temporal steps to completely damp out spurious numerical oscillations (Greeks mispricing) caused by non-smooth initial conditions at the strike boundary.

* O(N) Solver Optimization: Resolves the resulting tridiagonal matrix using a highly optimized, linear-time Thomas Algorithm variant, minimizing hardware cache-line misses and maximizing memory bus bandwidth.

### 3. Enterprise Identity Federation & Compliance
Engineered for highly regulated financial sectors and corporate clearing operations:

* JSON Web Key Sets (JWKS): Implements cryptographically secure, stateless identity verification by interacting directly with AWS Cognito JWKS endpoints.

* Middleware Signature Verification: Every transaction request must clear asynchronous signature token validation routines, verifying asymmetric cryptographic boundaries within secure network perimeters.

## 📡 API Routing Topology & Technical Overview
```
       HTTPS Ingress (Cognito JWT Verification)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │       Python FastAPI Engine      │
        └──────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   [JSON Payloads]  [Binary Array]  [Grid Steamer]
   /pricing/batch   /pricing/binary  /pricing/grid
        │                │                │
        └────────────────┼────────────────┘
                         ▼
  ctypes Memory Boundary (Pointer Passing / Contiguous Vectors)
                         │
                         ▼
 ┌────────────────────────────────────────────────────────┐
 │   Proprietary Compiled C++ OpenMP Shared Library (.so) │
 │   • Thomas Algorithm Solver    • Rannacher Damping     │
 └────────────────────────────────────────────────────────┘
```
## 🔧 Production Endpoint Matrix

### 🎯 POST `/v1/pricing/single`
Prices a single option layout instantly, executing a tight computational pass and returning only the compact, final Greeks (price, delta, gamma, theta, vega) without any dense surface-mesh overhead.

### ⚡ POST `/v1/pricing/batch`
High-Throughput Sweep Pipeline: Price thousands of options concurrently over multi-core processor threads using contiguous vector streams. Features on-demand high-precision volatility bump-and-scale passes to compute exact Vega arrays.

### 📊 POST `/v1/pricing/chart`
On-Demand Visualization Engine: Computes the entire multi-dimensional asset-time asset surface and streams back a pre-rendered, high-resolution 3D pricing mesh visualization as a raw PNG byte array to streamline front-end rendering overhead.

## 🛠️ Data Contract & Parameters
The option schema supports native integer flag handling to route execution vectors seamlessly down to the compiled computational core loops:
```
{
  "deriv": 4, 
  "s": 100.0,
  "k": 110.0,
  "time": 1.0,
  "sigma": 0.5,
  "r": 0.1,
  "q": 0.0,
  "Tn": 1000,
  "h": 1.0
}
```
* deriv: 0 = VanillaCall, 1 = VanillaPut, 2 = AmericanCall, 3 = AmericanPut, 4 = BermudanCall, 5 = BermudanPut.
* Tn: Number of temporal time grid increments (Matrix time-steps).
* h: Spatial mesh coordinate spacing thickness (dx).
* q: Continuous asset dividend payout yield percentage.

## 🔌 Technical Integration & API Specifications

All resource endpoints are mounted under the global `/v1` version prefix. With the
exception of the unauthenticated health probe and the raw binary ingress pipe, every
request must carry a valid AWS Cognito-issued OAuth2 JWT in the `Authorization: Bearer`
header. The examples below assume a `localhost:8000` deployment and a `$TOKEN`
environment variable holding your bearer token.

```bash
# Mint/export a bearer token once, then reuse it across every authenticated call below.
# In local dev (empty OAUTH2_JWKS_URL) any unsigned JWT with a "sub" claim is accepted.
export TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### ❤️ `GET /v1/health` — Liveness & Dependency Probe

```bash
# Unauthenticated readiness check. Reports app version, uptime, and whether the
# upstream Cognito JWKS endpoint is reachable for signature verification.
curl -s http://localhost:8000/v1/health
# → { "status": "ok", "version": "1.0.0-BETA", "uptime_seconds": 12.34, "authentication": "ok" }
```

### 🎯 `POST /v1/pricing/single` — Single Option Greeks

```bash
# Prices one contract and returns the compact Greeks block (price/delta/gamma/theta/vega)
# plus the resolved grid dimensions (Tn x Xm). Append ?calculate_vega=true to run the
# high-precision volatility bump-and-reprice pass (otherwise vega is skipped for speed).
curl -s -X POST "http://localhost:8000/v1/pricing/single?calculate_vega=true" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "deriv": 4,
        "s": 100.0,
        "k": 110.0,
        "time": 1.0,
        "sigma": 0.5,
        "r": 0.1,
        "q": 0.0,
        "Tn": 1000,
        "h": 1.0
      }'
# → { "price": ..., "delta": ..., "gamma": ..., "theta": ..., "vega": ..., "Tn": 1000, "Xm": ... }
```

### ⚡ `POST /v1/pricing/batch` — High-Throughput JSON Sweep

```bash
# Prices an entire ARRAY of option configs concurrently across the OpenMP core loops.
# The request body is a JSON list of OptionConfig objects; the response is a parallel
# list of compact Greeks. The optional ?calculate_vega flag applies to every item.
curl -s -X POST "http://localhost:8000/v1/pricing/batch?calculate_vega=false" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '[
        { "deriv": 0, "s": 100, "k": 110, "time": 1.0, "sigma": 0.5, "r": 0.1, "q": 0.0, "Tn": 1000, "h": 1.0 },
        { "deriv": 1, "s": 100, "k":  90, "time": 0.5, "sigma": 0.3, "r": 0.1, "q": 0.0, "Tn": 1000, "h": 1.0 }
      ]'
# → [ { "price": ..., "delta": ... }, { "price": ..., "delta": ... } ]
```

### 🧬 `POST /v1/pricing/binary` — Zero-Copy Binary Ingress

```bash
# Maximum-throughput pipe: POST a raw, contiguous byte stream of packed C OptionConfig
# structs (NOT JSON). The total payload length MUST be an exact multiple of the struct
# size or the request is rejected with HTTP 400. The response is the raw octet-stream
# of the computed Greeks structs, returned in the same order. Here we stream a prepared
# binary file and capture the raw bytes to disk.
curl -s -X POST "http://localhost:8000/v1/pricing/binary" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @configs.bin \
  --output greeks.bin
```

### 🌳 `POST /v1/pricing/grid` — Full 2D Surface (Binary)

```bash
# Computes the entire asset-time pricing surface and streams it back as a raw float64
# (row-major Tn x Xm) octet-stream. The scalar Greeks and grid geometry are delivered
# out-of-band in custom X-* response headers to keep the binary payload clean.
# Use -D to dump the headers (X-Price, X-Delta, X-Grid-Rows-Tn, X-Grid-Cols-Xm, ...).
curl -s -X POST "http://localhost:8000/v1/pricing/grid" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "deriv": 4, "s": 100, "k": 110, "time": 1.0, "sigma": 0.5, "r": 0.1, "q": 0.0, "Tn": 1000, "h": 1.0 }' \
  -D grid_headers.txt \
  --output option_surface.bin
```

### 📊 `POST /v1/pricing/chart` — Rendered 3D Surface (PNG)

```bash
# Computes the pricing grid and returns a pre-rendered, high-resolution 3D surface plot
# as a raw PNG image. Pipe the response straight to a file. Supports ?calculate_vega.
curl -s -X POST "http://localhost:8000/v1/pricing/chart" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "deriv": 4, "s": 100, "k": 110, "time": 1.0, "sigma": 0.5, "r": 0.1, "q": 0.0, "Tn": 1000, "h": 1.0 }' \
  --output pricing_surface.png
```

## 🧪 Deployment & Quality Assurance

* Containerization: Clean multi-stage Dockerfile pinning light production layers to support immediate horizontal scaling across AWS ECS/Fargate or Kubernetes clusters.

* Asynchronous Integration Testing: Full test suite powered by pytest, utilizing robust integration stencils to run 89 asynchronous validation tests covering memory alignment boundaries, token rotation failures, and parsing edge cases.

## Repository Architecture

```
chat-agent/
├── app/
│   ├── auth/jwt.py          <-- OAuth2 JWKS validation for AWS Cognito
│   ├── lib/                 <-- Proprietary .so and .dylib compiled HPC C++ OpenMP binaries
│   ├── middleware/          <-- SlowAPI rate limiter keyed by user sub
│   ├── routers/             <-- One router per resource group (incl. /pricing)
│   ├── schemas/             <-- Pydantic schemas for health and pricing CRUD requests and responses
│   ├── api.py               <-- Pydantic settings from .env
│   ├── config.py            <-- Pydantic settings from .env
│   └── main.py              <-- FastAPI app, CORS, and rate limiting
├── tests/                   <-- 11 pytests including Golden Master, quadratic convergence, image regressions, and batch testing
├── docs/                    <-- Design docs
└── pyproject.toml           <-- main project dependencies for `uv` package manager
```

### Install & Run Locally

```bash
cd grail-derivatives
cp .env.example .env          # fill in your API keys
uv venv && uv pip install -e ".[test]"
# or: python3 -m venv .venv && .venv/bin/pip install -e ".[test]"
uvicorn app.main:app --reload
```

> **Dev auth shortcut**: leave `OAUTH2_JWKS_URL` empty in `.env` — the server accepts any JWT without verifying the signature. Generate a test token at [jwt.io](https://jwt.io) with `{"sub": "user1"}` as the payload.

### Automated System Validation & Test Automation

Launch the regression suite to run all asynchronous integration tests within a mocked infrastructure layout:

```bash
pytest tests/ -v
```

*Note: The test layer provisions a low-latency, in-memory SQLite instances and fully simulates external provider endpoints via `moto`, eliminating external API overhead.*

### Interactive OpenAPI Documentation

*   **Swagger API Framework Interface**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   **ReDoc Schema Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Global Platform Configuration Settings

System behaviors are fully orchestrated using environment variables or localized security configurations via Pydantic Settings.

 Operational Variable | System Configuration Purpose | Default Factory Metric |
|---|---|---|
| `OAUTH2_JWKS_URL` | Cryptographic public key endpoint for token validation | `""` (dev bypass active) |
| `OAUTH2_AUDIENCE` | Token audience enforcement validation claim | `""` |
| `OAUTH2_ISSUER` | Token issuer identity validation claim | `""` |
| `OAUTH2_AUTH_URL` | Identity provider user authorization endpoint | `""` |
| `OAUTH2_TOKEN_URL` | Identity provider token exchange endpoint | `""` |
| `RATE_LIMIT_RPM` | Security threshold: Max requests per token per minute | `60` |
| `CORS_ORIGINS` | Permitted resource origins framework bounds | `*` (for local dev only!) |
| `AWS_REGION` | Cloud infrastructure deployment region geographic zone | `""` |
| `APP_VERSION` | Application semantic versioning tracking identifier | `0.0.1` |
| `LOG_LEVEL` | Application runtime message logging verbosity threshold | `"INFO"` |

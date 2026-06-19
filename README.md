# 🏛️ Grail Derivatives: Black-Scholes HPC FDM Pricing Engine

[![AWS Certified Solutions Architect - Associate](https://shields.io/badge/AWS-Certified_Solutions_Architect_Associate-orange?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://www.credly.com/badges/0bba41aa-57e1-499d-8077-245626cc4d1d/public_url)
[![Security Vetted](https://shields.io/badge/Security_Vetted-GoC_Secret_Level_II-red?style=for-the-badge)](#)
[![License: Proprietary](https://shields.io/badge/License-Proprietary_Closed_Source-darkred?style=for-the-badge)](#)


A production-grade, high-availability Financial Engineering microservice designed to ingest high-volume derivative option contracts and orchestrate intensive mathematical valuations under extreme concurrency. This platform serves as a modern, secure cloud-native REST API wrapper executing stateless token rotation, validation, and data serialization over an elite, high-performance computing numerical matrix engine.

> 🔒 Intellectual Property Boundary: This public repository contains the complete full-stack Python/FastAPI application layer, schema validations, OpenAPI specifications, and integration test suites. The underlying high-performance execution loops, multi-core OpenMP parallelization models, and O(N) tridiagonal numerical matrix solvers are maintained inside a proprietary, closed-source compiled machine-code shared library (.so) belonging exclusively to Dealer Gears Inc.

## 🚀 Core Architectural Moats
## 1. High-Performance FinOps & Zero-Copy Execution
Traditional JSON parsing is an extreme performance bottleneck for real-time trading desks and risk modeling engines. This API introduces a specialized binary ingress pipe:

* POST /v1/pricing/binary: Bypasses JSON serialization entirely. It accepts a raw, contiguous byte stream representing native C/C++ memory arrays (OptionConfig structs) and maps them directly to the underlying engine via pointer passing, achieving absolute zero-copy execution speed.
* POST /v1/pricing/grid: Quant Grid Engine computes the entire 2D asset-time pricing surface and streams it back to the client instantly as a raw binary float64 array, completely preventing browser memory freezing and minimizing network transfer overhead.

## 2. Quantitative Matrix Math & Numerical Stability
The platform supports Vanilla (European) contracts alongside highly path-dependent Exotic instruments (American, Bermudan, Single Barrier, and Double Barrier options).

* Crank-Nicolson Discretization: Solves continuous Black-Scholes partial differential equations (PDEs) over structured 2D asset-time meshes.
* Rannacher Damping: Switches to fully implicit stencils for the initial temporal steps to completely damp out spurious numerical oscillations (Greeks mispricing) caused by non-smooth initial conditions at the strike boundary.
* O(N) Solver Optimization: Resolves the resulting tridiagonal matrix using a highly optimized, linear-time Thomas Algorithm variant, minimizing hardware cache-line misses and maximizing memory bus bandwidth.

## 3. Enterprise Identity Federation & Compliance
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
## 🎯 POST /v1/pricing/single
Prices a single option layout instantly, executing a tight computational pass and returning only the compact, final Greeks (price, delta, gamma, theta, vega) without any dense surface-mesh overhead.
## ⚡ POST /v1/pricing/batch
High-Throughput Sweep Pipeline: Price thousands of options concurrently over multi-core processor threads using contiguous vector streams. Features on-demand high-precision volatility bump-and-scale passes to compute exact Vega arrays.
## 📊 POST /v1/pricing/chart
On-Demand Visualization Engine: Computes the entire multi-dimensional asset-time asset surface and streams back a pre-rendered, high-resolution 3D pricing mesh visualization as a raw PNG byte array to streamline front-end rendering overhead.

## 🛠️ Data Contract & Parameters
The option schema supports native integer flag handling to route execution vectors seamlessly down to the compiled computational core loops:

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


* deriv: 0 = VanillaCall, 1 = VanillaPut, 2 = AmericanCall, 3 = AmericanPut, 4 = BermudanCall, 5 = BermudanPut.
* Tn: Number of temporal time grid increments (Matrix time-steps).
* h: Spatial mesh coordinate spacing thickness (dx).
* q: Continuous asset dividend payout yield percentage.

## 🧪 Deployment & Quality Assurance

* Containerization: Clean multi-stage Dockerfile pinning light production layers to support immediate horizontal scaling across AWS ECS/Fargate or Kubernetes clusters.

* Asynchronous Integration Testing: Full test suite powered by pytest, utilizing robust integration stencils to run 89 asynchronous validation tests covering memory alignment boundaries, token rotation failures, and parsing edge cases.

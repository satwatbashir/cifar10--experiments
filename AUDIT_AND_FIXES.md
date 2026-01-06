# CIFAR-10 Federated Learning Experiments - Audit and Fixes

## Overview

This document describes the audit and fixes performed on the CIFAR-10 federated learning implementations (FedProx and HierFL) to ensure consistency with the HHAR reference implementations and correct various bugs.

**Audit Date:** January 2026
**Compared Against:** HHAR implementations (FedProx and HierFL)

---

## Executive Summary

| Method | Issues Found | Status |
|--------|--------------|--------|
| **FedProx** | 2 critical, 4 minor | Fixed |
| **HierFL** | 3 critical, 4 minor | Fixed |

---

## FedProx Fixes

### Critical Issues Fixed

#### 1. Duplicate `n_total` Calculation
**File:** `fedprox/fedge/server_app.py`

**Problem:** `n_total` was calculated twice in `aggregate_and_log_seeded()`, wasting computation.

**Fix:** Removed duplicate line, kept single calculation.

#### 2. Module-Level Global State Pollution
**File:** `fedprox/fedge/server_app.py`

**Problem:** Global mutable counters (`dist_round_counter`, `fit_round_counter`) at module level could cause state pollution between runs.

**Fix:** Moved counters inside `server_fn()` scope so they're captured by closures and reset per run.

### Minor Issues Fixed

| File | Issue | Fix |
|------|-------|-----|
| `server_app.py` | Unused `DATA_FLAGS` import | Removed |
| `server_app.py` | Redundant `import os, csv` inside function | Removed (already imported at top) |
| `server_app.py` | PEP8 violation (multiple imports on one line) | Split to separate lines |
| `client_app.py` | Unused `import pickle` | Removed |
| `client_app.py` | Unused `ndarrays_to_parameters` import | Removed |
| `client_app.py` & `server_app.py` | Dirichlet alpha fallback mismatch (0.3 vs config 0.5) | Aligned to 0.5 |

---

## HierFL Fixes

### Critical Issues Fixed

#### 1. Wrong `alpha_client` Value (Non-IID Configuration)
**File:** `HierFL/fedge/pyproject.toml`

**Problem:** `alpha_client = 1.0` produces nearly **uniform** (IID) data distribution, defeating the purpose of heterogeneous federated learning.

**Fix:** Changed to `alpha_client = 0.3` for proper non-IID distribution.

**Explanation of Dirichlet Alpha:**
- `alpha < 0.5`: Highly heterogeneous (very non-IID)
- `alpha = 0.3`: Moderate-high heterogeneity (recommended for FL research)
- `alpha = 1.0`: Uniform distribution (defeats non-IID purpose)
- `alpha > 1.0`: Approaching IID

#### 2. Private Flower API Usage
**File:** `HierFL/fedge/fedge/partitioning.py`

**Problem:** Code used private Flower API methods (`_partition_id_to_indices`, `_determine_partition_id_to_indices_if_needed()`) which will break on API updates.

**Fix:** Complete rewrite using direct Dirichlet distribution sampling with numpy, eliminating dependency on private APIs.

**New Implementation:**
- `_dirichlet_partition_indices()`: Direct numpy implementation
- `hier_dirichlet_indices()`: Updated to use numpy arrays instead of HuggingFace datasets

#### 3. Empty Failure Handlers
**Files:** `HierFL/fedge/fedge/leaf_server.py`, `HierFL/fedge/fedge/cloud_flower.py`

**Problem:** Empty `for failure in failures: pass` loops silently ignored errors.

**Fix:** Added proper logging warnings when failures occur.

### Minor Issues Fixed

| File | Issue | Fix |
|------|-------|-----|
| `leaf_server.py` | Duplicate `import sys` | Removed duplicate |
| `cloud_flower.py` | Empty failure handler | Added logging |
| `cloud_flower.py` | Unused loop extracting `acc` | Removed |
| `orchestrator.py` | Updated for new partitioning API | Uses numpy arrays now |
| `flower_topo.py` | Dead code (Mininet topology, never used) | **DELETED** |

---

## Architecture Notes

### Simulation Mode
Both FedProx and HierFL use local subprocess simulation (not distributed gRPC):
- **FedProx:** Uses Flower's built-in Ray-based simulation
- **HierFL:** Uses orchestrator.py with signal-file coordination

### Hierarchical Structure (HierFL)
```
Cloud Server (port 6000)
    |
    +-- Leaf Server 0 (port 5000) -- Clients 0-4
    +-- Leaf Server 1 (port 5001) -- Clients 0-4
    +-- Leaf Server 2 (port 5002) -- Clients 0-4
```

### Dirichlet Partitioning (HierFL)
Two-level Dirichlet split for hierarchical non-IID:
1. **Outer split** (`alpha_server = 0.5`): Distributes data across servers
2. **Inner split** (`alpha_client = 0.3`): Distributes each server's data across its clients

---

## Running the Experiments

### FedProx
```bash
cd fedprox
flwr run . --config seed=42
```

### HierFL
```bash
cd HierFL/fedge
SEED=42 python orchestrator.py
```

### Configuration Files
- **FedProx:** `fedprox/pyproject.toml`
- **HierFL:** `HierFL/fedge/pyproject.toml`

---

## File Structure

```
CIFAR-10/
├── fedprox/                    # FedProx implementation
│   ├── fedge/
│   │   ├── client_app.py      # Client logic
│   │   ├── server_app.py      # Server/aggregation logic
│   │   └── task.py            # Model & data loading
│   └── pyproject.toml         # Configuration
│
├── HierFL/                     # Hierarchical FL implementation
│   └── fedge/
│       ├── fedge/
│       │   ├── client_app.py  # Client app for simulation
│       │   ├── server_app.py  # Server app for simulation
│       │   ├── leaf_server.py # Leaf server process
│       │   ├── leaf_client.py # Leaf client process
│       │   ├── cloud_flower.py# Cloud aggregator
│       │   ├── proxy_client.py# Proxy for cloud upload
│       │   ├── partitioning.py# Dirichlet partitioning (REWRITTEN)
│       │   └── task.py        # Model & data loading
│       ├── orchestrator.py    # Main orchestrator
│       └── pyproject.toml     # Configuration
│
└── AUDIT_AND_FIXES.md         # This documentation
```

---

## Comparison with HHAR

| Aspect | HHAR | CIFAR-10 | Notes |
|--------|------|----------|-------|
| Dataset | HAR sensor data | CIFAR-10 images | Different domains |
| Model | 1D-CNN (6 channels) | LeNet-5 (3 channels) | Appropriate for data |
| Partitioning | User-based (natural non-IID) | Dirichlet (synthetic non-IID) | Different approaches |
| alpha_client | 0.5 (HHAR) | 0.3 (CIFAR-10) | Tuned for dataset |
| Simulation Mode | Yes | Yes | Consistent |
| Clients | 9 | 15 (3 servers × 5) | Different scale |

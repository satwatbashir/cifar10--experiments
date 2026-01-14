# Fedge Version History - CIFAR-10

## Results Summary

| Method | Accuracy | Rounds | Rank | Status |
|--------|----------|--------|------|--------|
| Fedge v8 | 32.1% | 186 | - | ❌ Failed (immediate collapse) |
| Fedge v3 | 60.23% | 100 | 1st | ✅ Done |
| Fedge v7 | 58.5% | 200 | - | ❌ Failed (collapse) |
| Fedge v2 | 59.16% | 100 | 2nd | ✅ Done |
| FedProx | 56.29% | 200 | 3rd | ✅ Baseline |
| Fedge v6 | ~56.5% | 200 | 4th | ❌ Failed |
| Fedge v4 | ~56% | 200 | 5th | ❌ Failed |
| HierFL | 50.58% | 200 | 6th | ✅ Baseline |
| Fedge v1 | 45.07% | 200 | 7th | ✅ Done |

---

## v8: FAILED - SCAFFOLD from Round 1

### Approach

Fix v7's collapse by starting SCAFFOLD from round 1 instead of after 30-round warmup.

### v8 Results (186 rounds)

| Round | Accuracy | Event |
|-------|----------|-------|
| 1 | 24.2% | Start |
| 3 | **10.4%** | IMMEDIATE COLLAPSE |
| 10 | 12.2% | Stuck |
| 50 | 22.8% | Slow recovery |
| 100 | 27.4% | Still recovering |
| 150 | 30.3% | Climbing |
| 186 | **32.1%** | Final |

**Final: 32.1%** - MUCH WORSE than v7 (58.5%) and v3 (60.23%)

### Why v8 Failed

Starting SCAFFOLD from round 1 caused **immediate collapse** at round 2-3:
1. Control variates initialized to zero made large corrections immediately
2. No stable model state to build corrections from
3. Training destabilized from the very beginning

### Conclusion: SCAFFOLD Doesn't Work

| Version | SCAFFOLD Config | Result |
|---------|-----------------|--------|
| v7 | Warmup 30 rounds | 58.5% (collapsed at r32, recovered) |
| v8 | From round 1 | 32.1% (collapsed at r2, never recovered) |
| v3 | **Disabled** | **60.23%** (best) |

**SCAFFOLD is fundamentally incompatible with this hierarchical FL setup.**

---

## v7: FAILED - SCAFFOLD with 30-Round Warmup

### v7 Results (200 rounds)

| Round | Accuracy | Event |
|-------|----------|-------|
| 1-31 | 25% → 52.2% | Good progress |
| 32 | **32.7%** | SCAFFOLD activated → COLLAPSE |
| 35 | 19.8% | Bottom |
| 100 | 54.1% | Recovery |
| 150 | 56.9% | Slow climb |
| 200 | **58.5%** | Final |

**Final: 58.5%** (worse than v3's 60.23% at 100 rounds)

### Why v7 Failed

1. **Sudden SCAFFOLD activation**: Control variates initialized at round 31 with model that had already converged partially
2. **20x gradient amplification**: `model_diff / (local_epochs * lr)` = `diff / 0.05`
3. **Model instability**: Large corrections pushed model off its learned distribution

### Lesson Learned

SCAFFOLD warmup causes collapse. Must start SCAFFOLD from round 1 so control variates evolve with the model.

---

## v6: FAILED - Server Isolation + Server-Level SCAFFOLD

### v6 Results (200 rounds)

| Round | Accuracy |
|-------|----------|
| 30 | 48.96% |
| 50 | 51.49% |
| 100 | 54.70% |
| 161 | 56.52% |
| 200 | ~57% |

**Final: ~57%** (worse than v3's 60.23%)

### Why v6 Failed

1. **Immediate isolation**: All 3 servers in separate clusters from round 1
2. **No knowledge sharing during warmup**: Each server got its own model back
3. **Server-level SCAFFOLD couldn't help**: `theta_cluster = theta_server` → control variates stayed ~0
4. **Same problem as v4**: Server isolation kills accuracy

### v6 Configuration (for reference)

| Parameter | Value |
|-----------|-------|
| server_isolation | true |
| scaffold_enabled | true |
| scaffold_server_enabled | true |
| prox_mu | 0.0 |

### Lesson Learned

Server isolation doesn't work for this setup. Global averaging (v3 approach) is necessary for good accuracy

---

## v3 Final Results (seed 42, 100 rounds)

| Metric | Value |
|--------|-------|
| Final accuracy | **60.23%** |
| Peak accuracy | **60.93%** (round 98) |
| Plateau start | ~round 70-80 |
| num_clusters | 1 (always) |

### v3 vs Baselines

| Baseline | v3 Improvement |
|----------|----------------|
| FedProx (56.29%) | **+3.94%** |
| HierFL (50.58%) | **+9.65%** |
| Fedge v2 (59.16%) | **+1.07%** |
| Fedge v1 (45.07%) | **+15.16%** |

### v3 Accuracy Progression

| Rounds | Accuracy | Gain per 10 rounds |
|--------|----------|-------------------|
| 1→10 | 25% → 43% | +18% |
| 10→20 | 43% → 49% | +6% |
| 20→30 | 49% → 51.4% | +2.4% |
| 30→40 | 51.4% → 54.9% | +3.5% |
| 40→50 | 54.9% → 55.8% | +0.9% |
| 50→60 | 55.8% → 57.1% | +1.3% |
| 60→70 | 57.1% → 59.4% | +2.3% |
| 70→80 | 59.4% → 58.8% | -0.6% ← dip |
| 80→90 | 58.8% → 59.0% | +0.2% |
| 90→100 | 59.0% → 60.2% | +1.2% |

### v3 Clustering Analysis

Server similarities at round 100: **0.9985** (all pairs)

**Problem:** Even tau=0.4 can't split servers when similarity > 0.99

---

## Current: v3 200-Round Experiment

Running v3 for 200 rounds to find plateau maximum.

### Expected Results

| Rounds | Expected Accuracy |
|--------|------------------|
| 100 | 60.23% (actual) |
| 150 | ~62% |
| 200 | ~63% (plateau) |

---

## v5: v3 + LR Decay (Current)

### Changes from v3

| Parameter | v3 | v5 | Reason |
|-----------|-----|-----|--------|
| lr_gamma | 1.0 | **0.995** | LR decay for finer convergence |
| Everything else | - | Same as v3 | v4 server isolation hurt accuracy |

### Expected Outcome

LR decay should help with late-round convergence where v3 plateaued.

---

## v4: FAILED - Server Isolation Hurt Accuracy

### What Was Tried

- Each server keeps its own model (no global averaging before clustering)
- Gradient-based clustering
- LR decay (lr_gamma=0.995)

### v4 Results

| Metric | v3 | v4 |
|--------|-----|-----|
| Accuracy | 60.23% | **~56%** |
| num_clusters | 1 | 3 (all separate) |
| Gradient similarities | N/A | 0.003-0.022 |

### Why v4 Failed

1. **Server isolation killed knowledge sharing**: Each server became a local model
2. **Gradient similarities too low**: 0.003-0.022 → all 3 servers in separate clusters
3. **No aggregation benefit**: Without knowledge sharing, non-IID hurt more than helped

### Lesson Learned

Clustering for clustering's sake doesn't help. Knowledge sharing through global averaging is critical for accuracy, even if it means servers stay in 1 cluster.

---

## v4 Original Plan (Archived)

### Root Cause Discovery

**Critical Bug Found**: In v1-v3, all servers received the SAME global model before clustering started (rounds 1-29). This prevented servers from diverging based on their non-IID data.

```python
# BUG in orchestrator.py lines 500-502 (v1-v3):
else:
    self.cluster_map = {sid: 0 for sid in server_ids}
    self.cluster_parameters = {0: global_weights}  # ALL servers get SAME model!
```

**Result**: By round 30, server similarities were 0.997+ → always 1 cluster → clustering never worked.

### v4 Fixes

#### Fix 1: Each Server Keeps Own Model (Critical)

```python
# FIX in orchestrator.py:
else:
    self.cluster_map = {sid: sid for sid in server_ids}
    self.cluster_parameters = {
        sid: weights_list[i] for i, sid in enumerate(server_ids)
    }  # Each server keeps its OWN model
```

**Impact**: Servers now naturally diverge based on non-IID data.

#### Fix 2: Gradient-Based Clustering

Instead of clustering by weight similarity (where models ARE), cluster by gradient direction (where models WANT TO GO).

```python
# New function in cluster_utils.py:
def gradient_based_clustering(server_weights_list, previous_weights_list, tau, round_num):
    # gradient = current_weights - previous_weights
    # similarity = cosine(gradient_i, gradient_j)
    # cluster by similarity threshold
```

**Why it works**: Even if weights converge, gradients reflect local data distribution.

#### Fix 3: LR Decay

```toml
lr_gamma = 0.995  # Decays to ~0.37 by round 200
```

### v4 Configuration Changes

| Parameter | v3 | v4 | Reason |
|-----------|-----|-----|--------|
| Server model sharing | All same | **Each keeps own** | Allow divergence |
| Clustering method | weight | **gradient** | Better for non-IID |
| tau | 0.4 | **0.5** | Adjusted for gradients |
| lr_gamma | 1.0 | **0.995** | Fine convergence |

### v4 Expected Results

| Metric | v3 | v4 Target |
|--------|-----|-----------|
| avg_accuracy (200 rounds) | ~63% | **65-70%** |
| num_clusters | 1 (always) | **2-3** (meaningful) |
| Server similarities | 0.997+ | **0.3-0.8** |

### v4 Key Files Modified

| File | Change |
|------|--------|
| `orchestrator.py:500-506` | Each server keeps own model |
| `orchestrator.py:476-488` | Gradient-based clustering support |
| `orchestrator.py:438` | Track previous_server_weights |
| `fedge/cluster_utils.py` | New `gradient_based_clustering()` function |
| `pyproject.toml` | method="gradient", lr_gamma=0.995 |

---

## v2 Final Results (seed 42, 100 rounds)

| Metric | Value |
|--------|-------|
| Final accuracy | 59.16% |
| Peak accuracy | 59.68% (round 98) |
| Plateau start | ~round 70-80 |
| num_clusters | 1 (always) |

### v2 vs Baselines

| Baseline | v2 Improvement |
|----------|----------------|
| FedProx (56.29%) | **+2.87%** |
| HierFL (50.58%) | **+8.58%** |
| Fedge v1 (45.07%) | **+14.09%** |

### v2 Accuracy Progression

| Rounds | Accuracy | Gain per 10 rounds |
|--------|----------|-------------------|
| 1→10 | 25% → 42% | +17% |
| 10→20 | 42% → 49% | +7% |
| 20→30 | 49% → 51.5% | +2.5% |
| 30→40 | 51.5% → 54.4% | +2.9% |
| 40→50 | 54.4% → 55.1% | +0.7% |
| 50→60 | 55.1% → 56.2% | +1.1% |
| 60→70 | 56.2% → 58.1% | +1.9% |
| 70→80 | 58.1% → 58.3% | +0.2% ← plateau |
| 80→90 | 58.3% → 58.5% | +0.2% |
| 90→100 | 58.5% → 59.2% | +0.7% |

---

## v1 → v2 Changes

### Configuration Changes

| Parameter | v1 | v2 | Reason |
|-----------|-----|-----|--------|
| `prox_mu` | 0.001 | **0.01** | Match FedProx regularization strength |
| `momentum` | 0.0 | **0.9** | Standard SGD momentum for faster convergence |
| `start_round` | 1 | **30** | Let models differentiate before clustering |

### What Worked in v2

| Change | Impact |
|--------|--------|
| momentum=0.9 | Faster, smoother convergence |
| prox_mu=0.01 | Reduced client drift |

### What Didn't Work in v2

| Issue | Observation |
|-------|-------------|
| Clustering | Still 1 cluster always (tau=0.7 too high) |
| start_round=30 | No effect (similarities still > 0.7) |

---

## v3 Attempt: SCAFFOLD Failed

### What We Tried

| Parameter | v2 | v3 (attempted) |
|-----------|-----|----------------|
| `scaffold_enabled` | false | **true** |
| `tau` | 0.7 | **0.4** |
| `label_smoothing` | 0.1 | **0.0** |

### SCAFFOLD Failure (seed 42)

```
Round 6:  38.97%  ← SCAFFOLD activates after warmup (5 rounds)
Round 7:  21.08%  ← CATASTROPHIC COLLAPSE
Round 8:  18.20%
...
Round 44: 19.48%  ← stuck near random
```

### Root Cause Analysis

1. **SCAFFOLD warmup too short** - Only 5 rounds, not enough for stable control variates
2. **Control variate explosion** - Division by `(local_epochs * lr)` = 0.05 amplifies by 20x
3. **Conflict with FedProx** - Both methods try to correct drift differently
4. **No gradient clipping on SCAFFOLD corrections** - Unbounded corrections

### Lesson Learned

SCAFFOLD needs careful tuning:
- Longer warmup (30+ rounds)
- Scaled-down corrections
- Possibly incompatible with FedProx

---

## v3 Revised Plan (No SCAFFOLD)

### Configuration Changes

| Parameter | v2 | v3 | Reason |
|-----------|-----|-----|--------|
| `scaffold_enabled` | false | **false** | SCAFFOLD caused collapse |
| `tau` | 0.7 | **0.4** | Force multiple clusters |
| `label_smoothing` | 0.1 | **0.0** | Match baselines |

### Expected Outcome

| Metric | v2 | v3 Target |
|--------|-----|-----------|
| avg_accuracy | 59.16% | **60-63%** |
| num_clusters | 1 | >1 after round 30 |

### Expected Gains

| Change | Expected Gain |
|--------|---------------|
| Lower tau (0.4) | +1-2% (if clusters form) |
| Remove label smoothing | +1-2% |
| **Total** | **+2-4%** → **61-63%** |

---

## v2 Issues & Plateau Analysis

### Issue A: Clustering Never Activated
- `num_clusters = 1` for all 100 rounds
- tau=0.7 too high (all server similarities > 0.7)
- No server specialization happening

### Issue B: Label Smoothing Mismatch
- v2 uses `label_smoothing=0.1`
- FedProx baseline uses `label_smoothing=0.0`
- Cost: ~1-2% accuracy

### Issue C: Plateau at ~59%
- Gains dropped to +0.2% per 10 rounds after round 70
- Possible causes:
  1. LeNet capacity ceiling (~62-65% theoretical max)
  2. Fixed learning rate (no decay)
  3. Clustering not helping (single cluster)
  4. Non-IID drift accumulation

---

## Brainstorming: Breaking the Plateau

### Ideas Within Sacred Constraints

| Idea | Expected Gain | Risk | Priority |
|------|---------------|------|----------|
| **Lower tau (0.4→0.3)** | +1-3% | May over-fragment | High |
| **LR decay (cosine)** | +1-2% | Slower early | Medium |
| **Gradient clipping adjustment** | +0.5-1% | May destabilize | Low |
| **Warmup LR schedule** | +0.5-1% | Complexity | Low |

### Ideas Requiring Sacred Parameter Changes

| Idea | Expected Gain | Constraint Violation |
|------|---------------|---------------------|
| ResNet-18 | +10-15% | Model (LeNet) |
| More local epochs | +2-3% | local_epochs=5 |
| Lower alpha_server | +2-4% | alpha_server=0.5 |

### Clustering Improvements

| Idea | Description |
|------|-------------|
| **Gradient-based clustering** | Cluster by gradient direction, not weight similarity |
| **Adaptive tau** | Start high (0.7), decay to 0.3 over rounds |
| **Per-layer clustering** | Cluster based on last FC layer only (already doing this) |
| **Cluster-specific LR** | Different learning rates per cluster |

### Novel Approaches

| Idea | Description | Complexity |
|------|-------------|------------|
| **FedDyn** | Dynamic regularizer instead of FedProx | High |
| **MOON** | Contrastive learning between local/global | High |
| **Personalization layers** | Freeze backbone, personalize last layer per cluster | Medium |

---

## Sacred Parameters (Cannot Change)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `lr_init` | 0.01 | NIID-Bench standard |
| `local_epochs` | 5 | Match baselines |
| `batch_size` | 64 | Match baselines |
| `num_servers` | 3 | Match HierFL |
| `clients_per_server` | [5, 5, 5] | Match HierFL |
| `alpha_server` | 0.5 | Non-IID standard |
| `alpha_client` | 1000.0 | IID within server |
| Model | LeNet | Match all baselines |

---

## File Locations

- Config: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/pyproject.toml`
- Task: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/fedge/task.py`
- Orchestrator: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/orchestrator.py`

## Git History

- **v4: (current)** - Architecture fix + gradient clustering + LR decay
  - Fix: Each server keeps own model (no global averaging before clustering)
  - New: Gradient-based clustering (cluster by update direction)
  - New: LR decay (lr_gamma=0.995)
- v3: 60.23% (100 rounds) - tau=0.4, no label smoothing, SCAFFOLD disabled
- v2: commit `b524d42` - 59.16%
- v1: commit `32a5d29` and earlier - 45.07%

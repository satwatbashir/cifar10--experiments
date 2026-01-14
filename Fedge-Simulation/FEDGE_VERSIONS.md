# Fedge Version History - CIFAR-10

## Results Summary

| Method | Accuracy | Rounds | Rank | Status |
|--------|----------|--------|------|--------|
| NIID-Bench SCAFFOLD | 69.8% | - | - | 📊 Target Baseline |
| Fedge v14 | 68-70% | 200 | - | 📋 Planned (scaling=20.0, paper formula) |
| Fedge v13 | 67-70% | 200 | - | 📋 Planned (scaling=5.0, aggressive) |
| Fedge v12 | 66-68% | 200 | - | 📋 Planned (scaling=2.0, moderate) |
| **Fedge v11** | **64-67%** | 200 | - | 🔄 Testing (scaling=1.0, 10x increase) |
| Fedge v10 | ~57-59% | 200 | - | ❌ Failed (augmentation + weak SCAFFOLD) |
| Fedge v9 | 62.9% | 200 | - | ✅ Done (SCAFFOLD fixed, but 200x undercorrection) |
| Fedge v3 | 60.23% | 100 | 1st | ✅ Done |
| Fedge v2 | 59.16% | 100 | 2nd | ✅ Done |
| Fedge v7 | 58.5% | 200 | - | ❌ Failed (collapse) |
| FedProx | 56.29% | 200 | 3rd | ✅ Baseline |
| Fedge v6 | ~56.5% | 200 | 4th | ❌ Failed |
| Fedge v4 | ~56% | 200 | 5th | ❌ Failed |
| HierFL | 50.58% | 200 | 6th | ✅ Baseline |
| Fedge v1 | 45.07% | 200 | 7th | ✅ Done |
| Fedge v8 | 32.1% | 186 | - | ❌ Failed (immediate collapse) |

---

## v10: Data Augmentation - FAILED (~57-59%)

### Goal

Add data augmentation to v9 base to break the 62% plateau.

### Approach

v9 SCAFFOLD is stable but plateaued at 62.9%. Per the fallback plan (Scenario B: accuracy < 65%), adding data augmentation is the next step.

### v10 Results: FAILED - Worse Than v9

| Round | v10 Actual | v9 Actual | Gap |
|-------|-----------|-----------|-----|
| 20 | 42.4% | 48.5% | **-6.1%** |
| 50 | 49.4% | 55.8% | **-6.4%** |
| 100 | 53.4% | 59.6% | **-6.2%** |
| 144 | 55.7% | ~61.5% | **~-6%** |
| **200 (proj)** | **~57-59%** | **62.9%** | **-4 to -6%** |

**v10 is consistently ~6% behind v9 and NOT catching up.**

### Root Cause Analysis

1. **Augmentation makes training harder** - AutoAugment + RandomErasing increase sample diversity, requiring more gradient updates to learn

2. **Weak SCAFFOLD can't compensate** - With scaling_factor=0.1, SCAFFOLD corrections are ~200x smaller than the paper formula (20.0). Control variates barely influence training.

3. **Double penalty** - Harder training + essentially disabled SCAFFOLD = worse than v9 (which had basic transforms but same weak SCAFFOLD)

### v10 Configuration (for reference)

| Component | v9 | v10 | Actual Gain |
|-----------|-----|-----|-------------|
| AutoAugment | None | CIFAR-10 policy | **-6%** (hurt!) |
| Cutout | None | 16x16 random erasing | (included above) |
| SCAFFOLD | Enabled (0.1 scaling) | Enabled (same) | No benefit |

### Lesson Learned

Data augmentation requires stronger SCAFFOLD to compensate for harder training. With weak SCAFFOLD (scaling=0.1), augmentation hurts more than it helps.

---

## v11: SCAFFOLD Scaling Fix - Target 64-67%

### Goal

Fix the root cause of v9/v10 underperformance: SCAFFOLD is ~200x undercorrected.

### Why v11 (Not v10b)?

| Option | Change | Expected Effect |
|--------|--------|-----------------|
| v10b (weight_decay) | Add L2 regularization | Won't fix core problem - SCAFFOLD still ~200x undercorrection |
| **v11 (scaling=1.0)** | **10x SCAFFOLD increase** | **Addresses root cause - control variates actually work** |

**v11 targets the fundamental issue: SCAFFOLD is essentially disabled.**

### v11 Configuration

```toml
# pyproject.toml - v11 SCAFFOLD config
scaffold_enabled = true
scaffold_scaling_factor = 1.0    # v11: 10x increase from v9's 0.1
scaffold_correction_clip = 0.1   # Keep same (prevents gradient explosion)
scaffold_warmup_rounds = 10      # Keep same (gradual activation)
scaffold_clip_value = 1.0        # Keep same (bounds control variates)
```

### v11 Code Changes

**File:** `fedge/task.py` - Remove v10 augmentation, revert to v9 basic transforms

```python
# v11: Revert to basic transforms (isolate scaling fix effect)
def _train_transform():
    return Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])
```

### Expected Results

| Phase | Rounds | v9 Actual | v11 Expected |
|-------|--------|-----------|--------------|
| Early | 1-20 | 24.9% → 48.5% | 25% → 50% |
| Mid | 20-50 | 48.5% → 55.8% | 50% → 58% |
| Late | 50-100 | 55.8% → 59.6% | 58% → 63% |
| Final | 100-200 | 59.6% → 62.9% | 63% → **65-67%** |

### Why This Should Work

v9's safeguards (correction_clip, warmup, clip_value) prevent v7/v8-style collapse. Increasing scaling_factor with these safeguards should be safe AND effective.

### If v11 Fails: Next Steps

| Result | Next Version | Change |
|--------|--------------|--------|
| Collapses | v11a | Reduce to scaling=0.5, increase warmup to 20 |
| Works but < 65% | v12 | Increase to scaling=2.0 |
| Works at 65%+ | v11+aug | Add augmentation back (now SCAFFOLD can handle it) |

---

## v11-v14: SCAFFOLD Scaling Fix Plans (Pending v10 Results)

### Root Cause Analysis: Why v9 Plateaued at 62.9%

**Key Discovery:** The v9 "fix" overcorrected by 200x, essentially disabling SCAFFOLD.

#### The Original SCAFFOLD Formula (Paper)
```
c_i^{new} = c_i - c_server + (1/Kη) × (w_global - w_local)
```
Where K=5 (local_epochs), η=0.01 (learning_rate)
**Scaling factor = 1/(Kη) = 1/0.05 = 20**

#### What v9 Changed
| Component | v7/v8 (Collapsed) | v9 (Stable but Weak) | Impact |
|-----------|-------------------|----------------------|--------|
| Scaling factor | 20 (paper formula) | **0.1** | **200x reduction** |
| Control variate clip | 10.0 | 1.0 | Good fix |
| Correction clip | None | 0.1 | Good fix |
| Warmup | None | 10 rounds | Good fix |

**Result:** Control variates are ~200x smaller than intended → SCAFFOLD corrections are nearly zero → Algorithm essentially disabled.

#### Evidence
| Version | Scaling | Result | Improvement over v3 |
|---------|---------|--------|---------------------|
| NIID-Bench | ~20 | 69.8% | +9.6% |
| v9 | 0.1 | 62.9% | **+2.7%** (should be +6-10%) |
| v3 | N/A | 60.23% | Baseline |

The +2.7% improvement is too small for proper SCAFFOLD, indicating it's barely functioning.

### Proposed Fix: Increase Scaling Factor with Safeguards

#### v11: Conservative (scaling_factor = 1.0)
```toml
scaffold_scaling_factor = 1.0    # 10x increase from v9's 0.1
scaffold_correction_clip = 0.1  # Keep same
scaffold_clip_value = 1.0       # Keep same
scaffold_warmup_rounds = 10     # Keep same
```
**Expected:** 64-67% | **Risk:** Low

#### v12: Moderate (scaling_factor = 2.0)
```toml
scaffold_scaling_factor = 2.0    # 20x increase from v9's 0.1
scaffold_correction_clip = 0.2  # Slightly larger
scaffold_clip_value = 1.0       # Keep same
scaffold_warmup_rounds = 10     # Keep same
```
**Expected:** 66-68% | **Risk:** Low-Medium

#### v13: Aggressive (scaling_factor = 5.0)
```toml
scaffold_scaling_factor = 5.0    # 50x increase from v9's 0.1
scaffold_correction_clip = 0.3  # Allow larger corrections
scaffold_clip_value = 1.0       # Keep same (critical)
scaffold_warmup_rounds = 15     # Longer warmup
```
**Expected:** 67-70% | **Risk:** Medium

#### v14: Paper Formula with Safeguards (scaling_factor = 20.0)
```toml
scaffold_scaling_factor = 20.0   # Match paper exactly
scaffold_correction_clip = 0.5   # Allow larger corrections
scaffold_clip_value = 1.0        # Critical safety bound
scaffold_warmup_rounds = 20      # Longer warmup
```
**Expected:** 68-70% | **Risk:** Higher (but we have safeguards v7/v8 lacked)

### Testing Strategy

| Order | Version | Change from v9 | Target | Risk |
|-------|---------|----------------|--------|------|
| 1 | v11 | scaling 0.1→1.0 | 64-67% | Low |
| 2 | v12 | scaling 0.1→2.0, clip 0.1→0.2 | 66-68% | Low-Medium |
| 3 | v13 | scaling 0.1→5.0, clip 0.1→0.3 | 67-70% | Medium |
| 4 | v14 | scaling 0.1→20.0 (paper) | 68-70% | Higher |

**Strategy:** Start with v11. If stable but accuracy insufficient, proceed to v12, etc.

### Stability Monitoring

**Good signs:**
- Faster accuracy improvement in rounds 20-100 than v9
- Control variate magnitudes: 0.01-0.5 (not 0.001)
- Smaller oscillation in late rounds

**Warning signs (reduce scaling):**
- Sudden accuracy drop >5% in one round
- Wild oscillation (±3% per round)
- Control variates hitting clip bounds constantly

### Why This Should Work

The v7/v8 collapse was NOT caused by the 20x scaling factor. It was caused by:
1. No clipping on control variates (exploded to 10+)
2. No clipping on corrections (destroyed gradients)
3. Sudden activation (no warmup)

**v9 fixed these safety issues but overcorrected the scaling.** By keeping the safeguards but increasing the scaling factor, we should get both stability AND proper drift correction.

---

## v9: SCAFFOLD Fixed - Result: 62.9%

### Goal

Fix SCAFFOLD implementation bugs that caused v7/v8 collapse. Target: Match NIID-Bench SCAFFOLD results (69.8%).

### Final Result: 62.9% (Below 68% Target)

| Metric | Value |
|--------|-------|
| Final accuracy | **62.9%** |
| Peak accuracy | **63.0%** (round 200) |
| Plateau start | ~round 100 |
| Improvement over v3 | **+2.7%** |

### v9 Training Progression

| Round | Accuracy | Phase |
|-------|----------|-------|
| 1 | 24.9% | Start |
| 20 | 48.5% | Early (on track) |
| 50 | 55.8% | Mid (on track) |
| 100 | 59.6% | Late (slightly behind) |
| 150 | 62.3% | **Plateau forming** |
| 200 | **62.9%** | **Stuck at plateau** |

### Oscillation Pattern Analysis

v9 showed increasing oscillation in later rounds:

| Range | Pattern | Max Dip |
|-------|---------|---------|
| r1-50 | Steady climb | -1.1% |
| r50-100 | Slower climb | -0.6% |
| r100-150 | Oscillating | -0.8% |
| r150-200 | **Stuck** (bouncing 61-63%) | -0.8% |

**Key Insight:** SCAFFOLD corrections are too conservative (0.1 scaling, 0.1 clip) to push past the 62% barrier. The model needs additional regularization (data augmentation) to escape the local minimum.

### Expected vs Actual: Detailed Comparison

| Phase | Rounds | Expected | Actual | Gap | Status |
|-------|--------|----------|--------|-----|--------|
| Early | 1-20 | 25% → 45% | 24.9% → 48.5% | +3.5% | ✅ **Ahead** |
| Mid | 20-50 | 45% → 55% | 48.5% → 55.8% | +0.8% | ✅ **On track** |
| Late | 50-100 | 55% → 62% | 55.8% → 59.6% | **-2.4%** | ⚠️ **Behind** |
| Final | 100-200 | 62% → 68% | 59.6% → 62.9% | **-5.1%** | ❌ **Plateaued** |

**Key Observation:** v9 started strong but fell behind after round 50 and completely plateaued after round 100.

### Gap Analysis: Where Did We Lose 5%?

| Factor | Impact | Evidence | Fix |
|--------|--------|----------|-----|
| **Correction clipping too tight** | -2-3% | SCAFFOLD gains capped at ±0.1 per gradient | Try `correction_clip=0.2` |
| **Missing data augmentation** | -3-5% | NIID-Bench likely uses augmentation | ✅ Added in v10 |
| **Scaling factor too conservative** | -1-2% | 0.1 × model_diff may underestimate correction | Try `scaling_factor=0.2` |
| **No weight decay** | -0.5-1% | L2 regularization helps generalization | Add `weight_decay=0.0005` |

### Why v9 Underperformed

1. **SCAFFOLD stability achieved** ✓ (no collapse like v7/v8)
2. **But corrections too weak** - 0.1 scaling factor may be underdoing variance reduction
3. **Plateau at 62%** - Model hit a local minimum that SCAFFOLD alone can't escape
4. **Missing data augmentation** - Standard FL baseline, but augmentation adds +3-5%

### Possible Implementation Issues to Investigate

| Issue | Description | How to Verify |
|-------|-------------|---------------|
| **NIID-Bench setup mismatch** | Their SCAFFOLD config may differ | Check NIID-Bench source code |
| **Control variate initialization** | Starting at zero may not be optimal | Try small random init |
| **Server control update** | Are we updating c_server correctly? | Add debug logging |
| **Gradient magnitude** | Are gradients in expected range (0.001-0.1)? | Log gradient stats |

### Recommendation for v10+

Based on gap analysis, priority order:

1. **v10 (current):** Add data augmentation (+3-5% expected)
2. **v10b (if needed):** Add weight_decay=0.0005 (+0.5-1% expected)
3. **v10c (if still low):** Increase scaling_factor to 0.2 (+1-2% expected)
4. **v10d (experimental):** Loosen correction_clip to 0.2 (risk: oscillation)

### Root Cause Analysis of v7/v8 Collapse

| Bug | Location | Impact |
|-----|----------|--------|
| **20x amplification** | `scaffold_utils.py:82` | `model_diff / (5 * 0.01) = model_diff / 0.05` |
| **No correction clipping** | `scaffold_utils.py:127-128` | Corrections applied unbounded to gradients |
| **Loose control variate clip** | `scaffold_utils.py:86` | clip_value=10.0 way too large |
| **Sudden activation** | `orchestrator.py` | No gradual warmup when SCAFFOLD starts |

### v9 Fixes

| Fix | Old Value | New Value | Effect |
|-----|-----------|-----------|--------|
| **Scaling factor** | `÷ (K*lr) = ÷0.05` | `× 0.1` | Prevents 20x amplification |
| **Correction clipping** | None | `[-0.1, 0.1]` | Bounds gradient corrections |
| **Control variate clip** | 10.0 | **1.0** | Tighter bound on c_i |
| **Warmup scaling** | None | Gradual over 10 rounds | `warmup_factor = round / 10` |

### v9 Configuration

```toml
# pyproject.toml - v9 SCAFFOLD config
scaffold_enabled = true
scaffold_scaling_factor = 0.1    # Replaces 1/(K*lr) division
scaffold_correction_clip = 0.1   # Clip corrections before applying
scaffold_warmup_rounds = 10      # Gradual activation
scaffold_clip_value = 1.0        # Tighter control variate bound
```

### v9 Code Changes

| File | Lines | Change |
|------|-------|--------|
| `fedge/scaffold_utils.py` | 56-57 | Added `scaling_factor=0.1`, `clip_value=1.0` defaults |
| `fedge/scaffold_utils.py` | 83-90 | `scaling_factor * model_diff` instead of division |
| `fedge/scaffold_utils.py` | 117-154 | Added warmup scaling + correction clipping |
| `fedge/task.py` | 530-531, 618-626 | Pass new SCAFFOLD params to train() |
| `orchestrator.py` | 96-105, 248-268 | Read config + pass params |

### Key Formula Changes

**Control Variate Update (update_client_control):**
```python
# OLD (v7/v8) - caused 20x amplification:
new_control = c_i - c_server + model_diff / (local_epochs * lr)

# NEW (v9) - controlled magnitude:
new_control = c_i - c_server + scaling_factor * model_diff  # scaling_factor=0.1
```

**Gradient Correction (apply_scaffold_correction):**
```python
# OLD (v7/v8) - unbounded corrections:
correction = -c_i + c_server
param.grad += correction

# NEW (v9) - clipped + warmup scaled:
warmup_factor = min(1.0, current_round / warmup_rounds)
raw_correction = -c_i + c_server
clipped = torch.clamp(raw_correction, -0.1, 0.1)
param.grad += warmup_factor * clipped
```

### Expected Results

| Phase | Rounds | Expected Accuracy |
|-------|--------|-------------------|
| Early | 1-20 | 25% → 45% |
| Mid | 20-50 | 45% → 55% |
| Late | 50-100 | 55% → 62% |
| Final | 100-200 | 62% → **68%+** |

### v9 vs Previous SCAFFOLD Attempts

| Version | SCAFFOLD Config | Result |
|---------|-----------------|--------|
| v7 | Warmup 30 rounds, no fixes | 58.5% (collapsed at r32) |
| v8 | From round 1, no fixes | 32.1% (collapsed at r2) |
| **v9** | Fixed formula + clipping + warmup | **Target: 68-70%** |

### Verification Plan

1. **Smoke test (10 rounds):** No collapse, accuracy > 30%
2. **Short run (50 rounds):** Monotonically increasing, > 50%
3. **Full run (200 rounds):** Target 68%+

### Fallback & Improvement Scenarios

#### If v9 SCAFFOLD Still Collapses (Scenario A)

| Parameter | Try Values | Expected Effect |
|-----------|------------|-----------------|
| `scaling_factor` | 0.05, 0.2, 0.5 | Lower = more conservative |
| `correction_clip` | 0.05, 0.2 | Tighter = more stable |
| `warmup_rounds` | 20, 30 | Longer = smoother activation |

#### If v9 Works But Accuracy < 65% (Scenario B)

**Add Data Augmentation (v10a)** - Expected: +3-5%

| Technique | Expected Gain | File |
|-----------|---------------|------|
| AutoAugment (CIFAR-10) | +2-3% | task.py |
| Cutout (random erasing) | +1-2% | task.py |
| RandAugment | +2-3% | task.py |

#### If v9 Works But Accuracy < 67% (Scenario C)

**Add Regularization (v10b)** - Expected: +1-2%

```toml
weight_decay = 0.0005    # L2 regularization
prox_mu = 0.01           # FedProx (may conflict with SCAFFOLD)
```

#### If SCAFFOLD Fundamentally Broken (Scenario E)

**Fallback to v3 + All Enhancements (v10d)**

```toml
scaffold_enabled = false
weight_decay = 0.0005
prox_mu = 0.01
lr_gamma = 0.99
```

Plus data augmentation → Expected: 64-67%

### Decision Tree

```
v9 Smoke Test
├── COLLAPSE → Tune params (A) or disable SCAFFOLD (E)
└── NO COLLAPSE → Full run
    ├── 68%+ → SUCCESS
    ├── 65-68% → Add augmentation (B)
    └── <65% → Add regularization (C) + augmentation
```

### Maximum Theoretical Accuracy

| Configuration | Expected |
|---------------|----------|
| v3 baseline | 60.23% |
| v9 (SCAFFOLD fixed) | 68-70% |
| v9 + Augmentation | 70-72% |
| v9 + Aug + Regularization | 71-73% |
| **LeNet max** | **~75%** |

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

### Conclusion: SCAFFOLD Implementation Was Buggy

| Version | SCAFFOLD Config | Result |
|---------|-----------------|--------|
| v7 | Warmup 30 rounds | 58.5% (collapsed at r32, recovered) |
| v8 | From round 1 | 32.1% (collapsed at r2, never recovered) |
| v3 | **Disabled** | **60.23%** (best without SCAFFOLD) |
| **v9** | **Fixed bugs** | **Target: 68-70%** |

**v9 Update:** SCAFFOLD collapse was due to implementation bugs (20x amplification, no correction clipping), not fundamental incompatibility. See v9 section above for fixes.

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

- **v11: (current)** - SCAFFOLD scaling fix (10x increase)
  - Change: `scaffold_scaling_factor = 0.1` → `1.0` (10x increase)
  - Revert: Remove v10 data augmentation (isolate scaling effect)
  - Target: 64-67%
  - Files: `pyproject.toml`, `fedge/task.py` (_train_transform)
- v10: ~57-59% - Data augmentation (FAILED - worse than v9)
  - Added: AutoAugment + RandomErasing
  - Problem: Weak SCAFFOLD + harder training = double penalty
  - Files: `fedge/task.py` (_train_transform)
- v9: 62.9% - SCAFFOLD bug fixes (stable but plateaued)
  - Fix: Replace `÷(K*lr)` with `× scaling_factor` (prevents 20x amplification)
  - Fix: Add correction clipping `[-0.1, 0.1]` before applying to gradients
  - Fix: Tighten control variate clip from 10.0 to 1.0
  - New: Gradual warmup scaling over 10 rounds
  - Result: +2.7% over v3, but below 68% target
  - Files: `fedge/scaffold_utils.py`, `fedge/task.py`, `orchestrator.py`, `pyproject.toml`
- v8: 32.1% - SCAFFOLD from round 1 (failed - immediate collapse)
- v7: 58.5% - SCAFFOLD with 30-round warmup (failed - collapse at r32)
- v3: 60.23% (100 rounds) - tau=0.4, no label smoothing, SCAFFOLD disabled
- v2: commit `b524d42` - 59.16%
- v1: commit `32a5d29` and earlier - 45.07%

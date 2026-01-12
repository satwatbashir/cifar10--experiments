# Fedge Version History - CIFAR-10

## Results Summary

| Method | Accuracy (200 rounds) | Rank |
|--------|----------------------|------|
| FedProx | 56.29% | 1st |
| HierFL | 50.58% | 2nd |
| Fedge v1 | 45.07% | 3rd |

---

## v1 → v2 Changes

### Configuration Changes

| Parameter | v1 | v2 | Reason |
|-----------|-----|-----|--------|
| `prox_mu` | 0.001 | **0.01** | Match FedProx regularization strength |
| `momentum` | 0.0 | **0.9** | Standard SGD momentum for faster convergence |
| `start_round` | 1 | **30** | Let models differentiate before clustering |

### Sacred Parameters (Unchanged)

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
| `scaffold_enabled` | false | Save for v3 |

---

## v1 Issues Identified

### Issue A: Clustering Ineffective
- Server similarity: 0.999+ throughout training
- `num_clusters = 1` for all 200 rounds
- tau=0.7 merges everything (all similarities > 0.7)

### Issue B: Weak Proximal Term
- v1 used `prox_mu=0.001` (10x weaker than FedProx)
- Caused excessive client drift

### Issue C: No Momentum
- v1 used `momentum=0.0`
- Slower convergence, oscillating gradients

### Issue D: Early Clustering
- v1 started clustering at round 1
- Models couldn't differentiate first

---

## v2 Expected Outcome

| Metric | v1 (100 rounds) | v2 Target |
|--------|-----------------|-----------|
| avg_accuracy | ~42% | >50% |
| num_clusters | 1 | Maybe >1 after round 30 |

If v2 reaches >50% at 100 rounds, full 200-round run should beat FedProx (56.29%).

---

## Future: v3 (If v2 succeeds)

| Parameter | v2 | v3 |
|-----------|-----|-----|
| `scaffold_enabled` | false | **true** |

SCAFFOLD may help with client drift correction in non-IID settings.

---

## File Location

Config: `/mnt/d/learn/CIFAR-10/Fedge-Simulation/pyproject.toml`

## Git History

- v2: commit `b524d42`
- v1: commit `32a5d29` and earlier

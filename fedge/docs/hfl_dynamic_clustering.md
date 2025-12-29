# Hierarchical Federated Learning with Dynamic Clustering

This document describes the design and practical usage of the HFL pipeline implemented in `fedge`.

---

## 1. System Architecture

```
┌──────────┐      ┌─────────────┐      ┌────────────┐
│  Client  │ ... │  LeafServer │ ... │  CloudAGG  │
└──────────┘      └─────────────┘      └────────────┘
```

1. **Clients** train on private data and send model updates to their *Leaf Server*.
2. **Leaf Servers** aggregate client updates (FedAvg) and forward their model to the *Cloud Aggregator* through **Proxy Clients**.
3. **Cloud Aggregator** performs *dynamic hierarchical clustering* over Leaf-Server models, does **cluster-wise aggregation**, and redistributes the resulting cluster models back to the leaf layer for the next global round.

The whole process repeats for `N_GLOBAL_ROUNDS`.

---

## 2. Dynamic Clustering Algorithm

### 2.1 Reference Set
A small public dataset (default: `cifar10_test`) is used to compare model outputs. Images are cached in memory once per run.

### 2.2 Similarity Metric
Pair-wise **symmetric KL divergence** on the mean probability vector for the reference set.

```
D(i,j) = ½·[ KL(p_i || p_j) + KL(p_j || p_i) ]
```

### 2.3 Clustering Procedure
1. Compute distance matrix `D ∈ ℝ^{S×S}` for `S` leaf servers.
2. Run Agglomerative Clustering (`sklearn`) with
   * linkage = "average"
   * metric   = "precomputed"
   * distance_threshold = **τ**
3. Produce integer labels `L = cluster_labels(D, τ)`.
4. For each cluster `c` compute a weighted FedAvg using each leaf’s sample count.

### 2.4 τ (Tau) — How to Choose
* τ controls granularity. Smaller → more clusters.
* Typical ranges for CIFAR-10 10-class tasks:
  * 0.05 – 0.15  → very strict (only near-identical models merge)
  * 0.2 – 0.4  → moderate (default `0.08`)
  * 0.5 – 1.0  → loose (merges most)
* Practical recipe:
  1. Run one global round with `SAVE_DISTANCE_MATRIX=1`.
  2. Inspect distances:  
     `python -c "import numpy as np, sys; print(np.load(sys.argv[1]))" rounds/global/round_0/distance_matrix.npy`
  3. Pick τ slightly **above** the 75-th percentile of observed distances if you want ~25 % splits.

> Notes: Distances grow as models diverge; you may need to adjust τ dynamically or rerun with a schedule.

---

## 3. Runtime Flags

* `CLUSTER_TAU` – float threshold (default `0.08`).
* `CLUSTER_REF` – reference set name (default `cifar10_test`).
* `SAVE_DISTANCE_MATRIX` – `1` saves `distance_matrix.npy` each global round.
* `SERVER_ROUNDS_PER_GLOBAL` – local FL rounds every leaf server performs before each upload.

Set environment variables in **PowerShell**:
```powershell
$Env:CLUSTER_TAU = "0.5"
$Env:SAVE_DISTANCE_MATRIX = "1"
python .\orchestrator.py
```

---

## 4. Output Artifacts

```
rounds/
 ├─ global/round_K/
 │   ├─ model.pkl               # overall FedAvg
 │   ├─ cluster_map.json        # {server_id: cluster_label}
 │   ├─ distance_matrix.npy     # if enabled
 │   └─ cluster_C/
 │       ├─ model.pkl           # cluster-specific model
 │       └─ server_ids.txt
 ├─ leaf/server_S/…             # per-leaf metrics & models
 └─ partitions.json             # dataset split description
```

---

## 5. Suggested Experimental Protocol

1. **Warm-up**: 1–2 global rounds with `τ = 0.75` to observe baseline merging.
2. **Sensitivity Sweep**: try τ in {0.25, 0.5, 1.0, 1.5} and record accuracy & cluster counts.
3. **Scalability**: increase number of clients/servers; τ often needs to be slightly higher with more heterogeneity.
4. **Ablation**: run with clustering disabled (`τ = 0`) to compare against vanilla FedAvg.

---

## 6. Known Limitations & Future Work
* Symmetric-KL requires identical output dimension; different architectures need alignment.
* Agglomerative clustering is `O(S²)`; consider approximate methods for >100 servers.
* τ is fixed per run; an adaptive schedule (e.g. decaying τ) could capture early exploration and late convergence.

---

## 7. References
* Flower AI – *A Friendly Federated Learning Framework*.
* MedMNIST – “A large-scale lightweight benchmark for classification and segmentation” (Yang et al., 2021).
* FedAvg – “Communication-Efficient Learning of Deep Networks from Decentralized Data” (McMahan et al., 2017).
* FedCluster – dynamic client grouping for heterogeneous FL (various).

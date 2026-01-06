# CIFAR-10 Federated Learning Experiments

This repository contains a collection of **federated learning (FL)** experiments on the CIFAR-10 dataset. It brings together both flat and hierarchical FL methods under a common structure so you can compare them side by side.

## Implemented Methods

- **Scaffold**  
  Control variate method to reduce client drift in heterogeneous FL.

- **FedProx**  
  Proximal regularization to stabilize training under system and data heterogeneity.

- **CFL (Clustered Federated Learning)**  
  Learns multiple cluster-specific models instead of a single global model.

- **pFedMe**  
  Personalized FL using Moreau envelopes (personalized model per client).

- **HierFL**  
  Baseline **hierarchical** FL (3 servers × 5 clients each) with two-level Dirichlet data splits.

- **Fedge / Fedge-150**  
  Novel hierarchical FL variants with dynamic clustering and SCAFFOLD + FedProx style regularization.

All methods are implemented as separate subprojects with their own `pyproject.toml` and Flower application configuration.

## Repository Structure

```text
.
├─ CFL/          # Clustered Federated Learning implementation
├─ fedprox/      # FedProx implementation
├─ pFedMe/       # pFedMe (personalized FL) implementation
├─ Scaffold/     # Scaffold implementation
├─ HierFL/       # Baseline hierarchical FL (3×5 clients)
├─ fedge/        # Fedge-v1 (novel hierarchical FL)
├─ fedge150/     # Fedge variant with 150 global rounds & different local epochs
├─ cifar10-seed/ # Seed-level analysis scripts and reports (metrics CSVs are git-ignored)
├─ cifar10-metrics/  # Auto-generated metrics (git-ignored)
├─ plots/        # Generated comparison plots and tables
├─ cifar10_accuracy_plot.py
├─ cifar10_loss_plot.py
└─ cifar10_results_table.py
```

> **Note:** Directories that contain large or auto-generated artefacts (dataset copies, training runs, metrics, logs) are excluded from version control via `.gitignore`. Only the code and configuration needed to run the experiments are tracked.

## Environment & Dependencies

Each method is packaged as a small Python project using `pyproject.toml` and [Flower](https://flower.dev/) for FL simulation. The recommended workflow is to create a virtual environment and install the method you want to run in editable mode.

Example (for Scaffold):

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows

cd Scaffold
pip install -e .
```

Repeat the `cd ...` / `pip install -e .` step for other methods (`fedprox`, `CFL`, `pFedMe`, `HierFL/fedge`, `fedge`, `fedge150`) depending on what you want to run.

## Running Experiments

The exact entry points differ slightly per method, but follow these guidelines:

- **Flat methods (Scaffold, FedProx, CFL, pFedMe)**  
  Each project defines a Flower app in its `pyproject.toml` under `[tool.flwr.app]` with `serverapp` and `clientapp` entries. You can run them using the Flower CLI according to the Flower documentation, using the configuration in the corresponding `pyproject.toml`.

- **Hierarchical methods (HierFL, Fedge, Fedge-150)**  
  These methods are orchestrated by Python scripts such as `HierFL/fedge/orchestrator.py`, `fedge/orchestrator.py`, and `fedge150/orchestrator.py`. They read their configuration from the local `pyproject.toml` and launch the hierarchical simulation (cloud + servers + clients).

For detailed hyperparameters (number of rounds, clients, Dirichlet alphas, learning rates, etc.), see the comments and `[tool.flwr.*]` sections inside each method's `pyproject.toml`.

## Reproducing CIFAR-10 Results

1. **Prepare environment**
   - Create and activate a virtual environment.
   - Install the method you care about via `pip install -e .` inside its folder.

2. **Download dataset**
   - The first run will usually download CIFAR-10 automatically using `torchvision` / `flwr-datasets`.

3. **Run the experiment**
   - For flat methods, launch the Flower app as per the Flower CLI instructions.
   - For hierarchical methods, run the corresponding `orchestrator.py` script.

4. **Generate metrics & plots**
   - After runs complete, helper scripts at the repository root can aggregate metrics and create figures:
     - `cifar10_results_table.py` → final comparison table.
     - `cifar10_accuracy_plot.py` → accuracy vs. rounds plot.
     - `cifar10_loss_plot.py` → loss vs. rounds plot.

These scripts expect metrics CSVs in `cifar10-metrics/`, which are produced by the training runs. The CSVs themselves are not committed to git, but are regenerated when you run experiments.

## License

Specify your chosen license here (for example, Apache-2.0 to match the project `pyproject.toml` files), or add a separate `LICENSE` file.

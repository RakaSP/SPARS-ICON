# SNF-ICON Experiments in SPARS

This repository contains the simulator extension, experiment configurations, workloads, platform models, and visualization notebook used for the paper:

> **A Smallest-Need-First Job Scheduling Framework with Adaptive Optimization of Idle Node Counts for Energy-Efficient HPC Systems**

SNF-ICON is an event-driven scheduling and power-management method for rigid parallel jobs in high-performance computing (HPC) systems. It combines:

- **Smallest-Need-First (SNF)** job ordering;
- predictive wake-up planning for queued jobs;
- adaptive selection of warm spare nodes for future arrivals;
- a recent-window workload screen; and
- an **SNF+IPM fallback** when the recent data are insufficient or do not fit the model closely enough.

This repository is self-contained for reproducing the paper experiments. The general-purpose SPARS simulator is included directly in this repository, so a separate simulator repository is not required.

## Repository contents

```text
SPARS/                     SPARS discrete-event simulator and SNF-ICON implementation
RL_Agent/SPARS/            reinforcement-learning baseline components
workloads/                 paper workload files
platforms/                 paper platform descriptions
RunSPARSOuter.py           complete paper experiment definitions and parameter sweeps
RunAll.py                  experiment launcher and worker control
RunSPARSConfig.py          runner for one generated YAML configuration
PaperVis.ipynb             notebook used to generate the paper figures
PlotMetrics.py             general metric-comparison plotting utility
PlotGantt.py               general Gantt-chart plotting utility
requirements-linux.txt     Linux Python dependencies
requirements-windows.txt   Windows Python dependencies
```

## Requirements

- Python **3.11**
- Linux or Windows
- Python virtual environment support
- Jupyter Notebook or JupyterLab for `PaperVis.ipynb`
- PyTorch for the reinforcement-learning baseline

Run all commands from the repository root.

`RunAll.py` expects the virtual environment to be named `SPARS-venv` and to be located inside the repository directory.

## Installation

### 1. Clone the repository

```bash
git clone --depth 1 https://github.com/RakaSP/SPARS-ICON.git
cd SPARS-ICON
```

### 2. Create the virtual environment and install dependencies

#### Linux

```bash
python3.11 -m venv SPARS-venv
source SPARS-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux.txt
```

#### Windows PowerShell

```powershell
py -3.11 -m venv SPARS-venv
.\SPARS-venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
```

### 3. Install PyTorch

PyTorch is installed separately because CPU and GPU systems require different builds.

For a CPU-only installation:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For a GPU installation, use the command generated for the machine by the official PyTorch installer and run it inside `SPARS-venv`.

Verify the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('Accelerator available:', torch.cuda.is_available())"
```

## Simulator overview

SPARS is an event-driven simulator for HPC scheduling, resource allocation, and node power management. A simulation combines three types of input:

1. **Workload** — job arrival time, requested node count, requested wall time, actual runtime, and job identifier.
2. **Platform** — simulated nodes, power values, state-transition times, and compute-speed profiles.
3. **Experiment configuration** — scheduler, power-management parameters, workload and platform paths, logging, output, sweeps, and optional RL settings.

The simulator models nodes in the following states:

- computing;
- active and idle;
- switching on;
- switching off; and
- sleeping.

At each simulation timestamp, SPARS processes the pending events, records the resulting state, invokes the scheduler, inserts the scheduler's actions, and then advances to the next event time.

## SNF-ICON overview

SNF-ICON uses two operating modes that are reconsidered at every scheduler invocation.

### ICON mode

When recent interarrival and completed-service samples pass the configured screen, the controller:

1. starts every queued job that currently fits, in SNF order;
2. predicts release times for nodes used by running jobs;
3. creates a sequential future plan for the remaining queued jobs;
4. schedules node wake-ups near the predicted job start times; and
5. when the queue becomes empty, evaluates how many warm spare nodes should remain ready for future arrivals.

### SNF+IPM fallback mode

When the recent window has too few samples or fails the configured checks, the current scheduler invocation uses the SNF+IPM baseline. Queued jobs are still scheduled and required nodes may still be woken, but ICON-specific prediction and warm-spare optimization are not used.

The default paper configuration also includes an arrival-recency gate and a two-hour idle timeout.

## Paper experiment suite

The complete paper experiment definitions are stored in:

```text
RunSPARSOuter.py
```

The repository includes the paper workloads and platform models under `workloads/` and `platforms/`.

The main evaluation covers six workload-platform cases:

| Platform | Workload |
| --- | --- |
| AOBA-derived 64-node model | DAS2 FS1, first 3000 jobs |
| AOBA-derived 64-node model | DAS2 FS2, first 3000 jobs |
| AOBA-derived 64-node model | DAS2 FS3, first 3000 jobs |
| AOBA-derived 64-node model | DAS2 FS4, first 3000 jobs |
| AOBA-derived 64-node model | generated 3000-job Markovian workload |
| AOBA-derived 1152-node model | SDSC Blue workload |

The configured comparisons include:

- **FCFS/B+IPM**;
- **SNF+IPM**;
- **SNF-ICON**;
- **SNF-ICON-NG**, with the arrival-recency gate disabled;
- **SNF-ICON-NF**, with fallback disabled;
- **SNF-ICON-NGNF**, with both safeguards disabled; and
- the available **RL Budiarjo** baseline experiments.

The suite also contains the paper's parameter and platform studies, including:

- Markovianity lookback-window sweeps;
- waiting-weight sweeps;
- gate and fallback ablations;
- waiting-time analysis by requested job size; and
- AOBA-versus-Taurus platform comparisons.

Unless an experiment overrides them, the main SNF-ICON defaults used in the paper are:

| Parameter | Default |
| --- | ---: |
| Waiting-time weight | `5000` |
| Energy weight | `1` |
| Initial arrival rate | `1/3600 s^-1` |
| Arrival EMA factor | `0.2` |
| Service/runtime-ratio EMA factor | `0.1` |
| Requested-node EMA factor | `0.1` |
| Minimum spare nodes | `1` |
| Maximum spare nodes | platform size |
| Maximum prediction horizon | `24 h` |
| Arrival-recency window | `12 h` |
| Markovianity lookback window | `2 h` |
| Minimum screen samples | `3` |
| Maximum screen samples | `1000` |
| CV tolerance | `0.45` |
| Maximum absolute lag-one correlation | `0.65` |
| KS multiplier | `1.25` |
| KS scale | `3.0` |
| Idle timeout | `2 h` |

## End-to-end reproduction

### Step 1: Review the experiment selection

Open `RunSPARSOuter.py` before launching a full run.

This file defines:

- the shared base configuration;
- enabled and disabled experiment groups;
- workload and platform combinations;
- parameter sweeps;
- heuristic and RL configurations; and
- automatic post-processing controls.

For a complete paper reproduction, keep the paper experiment groups enabled as provided. For a smaller test, temporarily disable large groups and enable only one workload-platform combination.

### Step 2: Run the experiments

Run with one worker:

```bash
python RunAll.py 1
```

Run with a larger worker limit, for example 24:

```bash
python RunAll.py 24
```

The numeric argument controls the CPU-core limit and the maximum number of concurrent non-RL experiment workers. RL experiments are run one at a time.

`RunAll.py` launches `RunSPARSOuter.py`, generates concrete YAML configurations, and executes them through `RunSPARSConfig.py`.

A full reproduction contains many workload, method, ablation, sweep, and platform combinations. It may require substantial runtime, memory, and disk space. A one-worker smoke test is recommended before starting the complete suite.

### Step 3: Monitor the run

Normal progress is printed to the terminal. Generated configurations and results are written under `results/`.

A typical output path has the form:

```text
results/<UID>/<experiment>/<platform>/<workload>/<algorithm-and-parameters>/
```

A non-RL run normally contains files such as:

```text
simulator_config_used.yaml
raw_job_log.csv
unfinished_jobs_log.csv
node_log.csv
waiting_time_log.csv
energy_log.csv
metrics.csv
state_switch.csv
runtime_seconds.txt
profile.csv
```

SNF-ICON experiments also record diagnostics for the selected operating mode, workload-screen results, prediction values, spare-target candidates, and node power-state actions.

RL runs may additionally contain epoch directories, step logs, checkpoints, Gym metadata, and a `best_epoch_<n>` directory.

### Step 4: Generate the paper figures

Start Jupyter from the repository root:

```bash
jupyter notebook PaperVis.ipynb
```

or:

```bash
jupyter lab PaperVis.ipynb
```

Before running all cells, verify that any result-directory variables in the first configuration cells point to the local output produced in Step 2. Then select **Run All**.

`PaperVis.ipynb` reads the generated experiment results and creates the plots used in the paper, including the base comparison, multi-metric profiles, mode and Markovianity diagnostics, variant trade-offs, job-size waiting-time analysis, parameter sweeps, and cross-platform comparison.

The draft paper presents these analyses as Figures 3–10.

## General-purpose plots

The repository also includes two plotting utilities independent of the paper notebook.

Generate metric-comparison plots for one result UID:

```bash
python PlotMetrics.py results/<UID>
```

Generate Gantt plots for one result UID:

```bash
python PlotGantt.py results/<UID>
```

`PlotMetrics.py` reads the `metrics.csv` files below the selected UID. `PlotGantt.py` reads `node_log.csv` and writes per-run Gantt images.

## Reproducibility notes

For every reported experiment, retain:

- the workload JSON file;
- the platform JSON file;
- the generated YAML configuration;
- `simulator_config_used.yaml` from the result directory;
- the exact repository commit; and
- the Python and PyTorch versions used for the run.

The simulator uses seconds as its time unit. Values written as hours in the paper configuration are converted to seconds in the experiment definitions.

Do not compare result folders produced by different code revisions without also checking their saved `simulator_config_used.yaml` files.

## Troubleshooting

### `RunAll.py` cannot find the virtual environment

Confirm that the environment is named exactly `SPARS-venv` and is located in the repository root:

```text
SPARS-ICON/SPARS-venv/
```

### PyTorch import error

Activate `SPARS-venv` and install the CPU or GPU PyTorch build separately. PyTorch is intentionally not pinned in the operating-system requirement files.

### A workload requests more nodes than the platform provides

Use a matching workload-platform pair from `RunSPARSOuter.py`. The requested node count of every job must not exceed the selected platform size.

### The full run is too large for the machine

First run a single experiment group with:

```bash
python RunAll.py 1
```

After confirming that the configuration completes correctly, increase the worker limit gradually.

### `PaperVis.ipynb` cannot find results

Check the result root or UID variables in the notebook and confirm that the selected directory contains the expected experiment subdirectories and `metrics.csv` files.

### A figure is missing an RL point

The paper reports RL results only for workload datasets with an available corresponding RL experiment. Confirm that the required RL result or checkpoint is present and that the relevant experiment is enabled in `RunSPARSOuter.py`.

## Citation

The paper is currently identified as:

> Reza Pulungan, Raka Satya Prasasta, Santana Yuda Pradata, Mursalim, Hiroyuki Takizawa, and Muhammad Alfian Amrizal, “A Smallest-Need-First Job Scheduling Framework with Adaptive Optimization of Idle Node Counts for Energy-Efficient HPC Systems.”

The simulator used by this repository is SPARS:

> M. A. Amrizal, R. S. Prasasta, S. Y. Pradata, K. G. Santiyuda, R. Pulungan, and H. Takizawa, “SPARS: A reinforcement learning-enabled simulator for power management in HPC job scheduling,” *SoftwareX*, vol. 34, 102693, 2026.

## License

This repository is released under the MIT License. See `LICENSE` for details.

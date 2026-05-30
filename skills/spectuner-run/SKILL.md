---
name: spectuner-run
description: Work with the Spectuner codebase — a Python framework for automated spectral line analysis of interstellar molecules (LTE model, broadband line identification, pixel-by-pixel cube fitting). Use this skill whenever the user mentions spectuner, line identification, molecular spectra, CDMS database queries, LTE fitting, T_ex / N_tot fitting, spectral cube fitting, or runs any of the `exec_fit` / `exec_identify` / `exec_modify` / `exec_config` commands — even if they don't explicitly name the package. Long-running fits are routine in this codebase, so the skill also covers how to keep tqdm progress visible.
---

# Spectuner

Spectuner is a Python framework for automated spectral line analysis of interstellar molecules. It implements the **one-dimensional LTE (Local Thermodynamic Equilibrium) spectral line model** and is used by radio astronomers to identify molecular species in observed spectra and fit physical parameters (Qiu et al. 2025, ApJS 277, 21).

## What it does

Two main applications:

1. **Broadband line identification** — given an observed spectrum (frequency vs. intensity), determine which molecular species are present and recover their physical parameters.
2. **Pixel-by-pixel spectral line fitting** — for spectral cubes (e.g. ALMA data), fit the LTE model at every pixel, optionally using a neural network for fast initial guesses.

## Fitting is slow — keep the progress bar visible

This is the single most important operational fact about spectuner: **fits typically take minutes to hours**, and in extreme cases (large surveys, dense cubes, many species) longer. Spectuner already wires tqdm bars into every long loop so the user can tell it is making progress. Your job, when running fits on the user's behalf, is to make sure those bars actually reach them.

### Where the bars appear

| Stage                                  | Source                            | Bar label                              |
|----------------------------------------|-----------------------------------|----------------------------------------|
| Individual fitting phase (per species) | `optimize/optimize.py`            | `Fitting`                              |
| PSO / ABC optimization cycles          | `optimize/abc.py`                 | `loss = <value>` (updates each cycle)  |
| Combining phase (joint refinement)     | `identify/run_combine.py`         | `Combining <species name>`             |
| Pixel-by-pixel cube fitting            | `cube.py`                         | `Optimizing` (per batch of 32 pixels)  |
| Cube preprocessing                     | `cube.py`                         | `Checking invalid values...`           |
| Neural-network inference               | `ai/inference.py`                 | `Predicting`                           |

If the user reports they "don't see any output" or asks whether a fit has hung, **the absence of a progress bar is the diagnostic**, not the absence of stdout. Bars go to stderr.

### How to run long fits without hiding the bars

When running a fit yourself, follow these rules — they exist because tqdm writes carriage-return-based bars to stderr, and stderr-suppressing or output-buffering tricks turn the bar into either silence or a wall of repeated lines.

- **Do not redirect stderr** (`2>/dev/null`, `&>file`) — that kills the bar.
- **Do not pipe through `cat`/`head`** without `--line-buffered` equivalents — that buffers tqdm output and the user sees nothing for minutes.
- **For long fits, prefer Bash with `run_in_background: true`.** The user is notified when the job exits, and you can `Read` the output file at any time to check progress. This is far better than blocking the conversation on a 30-minute fit.
- **For Monitor-based watching**, point the monitor at the log file with `tail -F` and grep for the bar's description tokens (`Fitting`, `Combining`, `loss =`, `Optimizing`). Emit a line per *new* species or cycle, not every refresh — otherwise you flood the conversation.
- **Never set `disable_pbar=True`** in `ai.predict_single_pixel` unless the user explicitly asks for silent runs. It exists for batch/CI use, not as a default.

### Setting expectations before kicking off a fit

If the user is about to start a fit and you can estimate the cost, say so. Drivers of runtime:

- **Frequency coverage × number of species** — broadband surveys with no species filter (`config.set_ident_species(species=None)`) hit every molecule CDMS suggests for that frequency range, which is expensive.
- **`n_swarm` × `n_trial` × number of optimization cycles** — PSO is the default for line ID; doubling `n_swarm` roughly doubles wall time.
- **`n_process`** — the multiprocessing pool size; set to the physical core count via `config.set_n_process(N)`.
- **Cube size × `batch_size`** — pixel-by-pixel fits scale with pixel count; with the neural network on GPU, throughput is dominated by the optimizer (`slsqp`) not inference.

When in doubt, suggest the user start with a narrow species list (e.g. `["CH3OH", "H2CO"]`) to verify the pipeline, then widen the scope once the config is dialled in.

## Core dependencies

- Python ≥ 3.10
- `astropy`, `h5py`, `numpy`, `numba`, `scipy`, `pandas`, `matplotlib`, `swing-opt`, `tqdm`
- Optional: `pytorch` (only required for the AI / inference module)
- External data:
  - **CDMS** (Cologne Database for Molecular Spectroscopy) SQLite file — the spectroscopic database, required input
  - Neural network weights from Hugging Face (`yqiuu/Spectuner-D1`) — optional, for the AI module

## Code layout

```
spectuner/
├── __init__.py          # Re-exports public API from submodules
├── scripts.py           # CLI entry points (exec_config, exec_fit, exec_identify, exec_modify)
├── config/              # YAML config loading; the Config dict subclass with setter methods
│   ├── config.py        # load_config, save_config, Config class
│   └── templates/       # config.yml, species.yml, modify.yml templates
├── sl_model/            # The LTE spectral line model
│   ├── sl_model.py      # SLModel — the forward model
│   ├── sl_database.py   # CDMS database interface
│   ├── parameters.py    # ParameterManager — handles theta, T_ex, N_tot, delta_v, v_offset
│   └── atoms.py
├── identify/            # Line identification pipeline (two phases)
│   ├── identify.py      # Identification logic, IdentResult, LineTable, ResultManager
│   ├── run_single.py    # run_individual_line_id — phase 1
│   ├── run_combine.py   # run_combining_line_id — phase 2 (tqdm "Combining ...")
│   └── run_all.py
├── optimize/            # Optimizers (PSO / ABC via swing-opt, SLSQP via scipy)
│   ├── optimize.py      # tqdm "Fitting"
│   └── abc.py           # tqdm "loss = ..."
├── ai/                  # Neural network for initial-guess inference (PyTorch)
├── cube.py              # Pixel-by-pixel cube fitting; CubeItem, CubePipeline, HDFCubeManager
├── peaks.py             # Peak detection / PeakManager
├── preprocess.py        # Spectrum preprocessing
├── spectral_plot.py     # Plotting utilities
├── slm_factory.py       # Factory functions for SLModel
├── modify.py            # Post-hoc modification of results (include/exclude species)
└── tests/               # pytest tests
```

`spectuner/__init__.py` re-exports everything from submodules, so users typically `import spectuner` and access via `spectuner.X`.

## The spectral line model — the five fitting parameters

| Param      | Meaning                | Typical unit | Conventional scale |
|------------|------------------------|--------------|--------------------|
| `theta`    | Source size            | arcsec       | linear             |
| `T_ex`     | Excitation temperature | K            | linear             |
| `N_tot`    | Column density         | cm⁻²         | **log** (`is_log=True`) |
| `delta_v`  | Velocity width         | km/s         | linear             |
| `v_offset` | Velocity offset        | km/s         | linear (±12 limit if using AI) |

These are the canonical names — they appear as keys in `config["param_info"]` and in `ParameterManager.param_names`. When editing fitting code, do not invent aliases.

## CLI commands

Installed as console scripts (defined in `spectuner/scripts.py`):

- `exec_config <dir>` — copy YAML config templates into `<dir>`
- `exec_fit <config_dir> <save_dir> [--mode single|combine|entire] [--fbase <pickle>]` — run spectral fitting
- `exec_identify <config_dir> <save_dir> [--mode single|combine]` — perform identification on existing fitting results
- `exec_modify <config_dir> <save_dir>` — apply post-hoc modifications (include/exclude species, frequencies)

Typical line-identification workflow:

1. `exec_config ./myconfig` → creates `config.yml`, `species.yml`, `modify.yml`
2. Edit the YAMLs (set `obs_info`, `sl_model.fname_db`, species list, etc.)
3. `exec_fit ./myconfig ./results --mode entire` → runs both phases; **`tqdm` bars stream to the terminal — keep them visible**
4. Optionally `exec_modify` then `exec_identify` to refine results

## The configuration system

`spectuner.load_default_config()` returns a `Config` (a dict subclass) with sensible defaults. Use its setter methods rather than poking into the dict directly — they validate and keep the schema honest:

- `config.append_spectral_window(spec, beam_info, noise, T_bg, need_cmb)` — add an observed spectrum (2D array: freq [MHz], intensity [K])
- `config.set_fname_db(path)` — set the CDMS sqlite database path
- `config.set_n_process(n)` — multiprocessing pool size (drives fit wall time)
- `config.set_param_info(name, is_log, bound, is_shared, special)` — bounds/scale per parameter
- `config.set_optimizer(method, **kwargs)` — `"pso"` for line ID, `"slsqp"` for pixel-by-pixel
- `config.set_ident_species(species, collect_iso, combine_iso, combine_state, ...)`
- `config.set_pixel_by_pixel_fitting(species, loss_fn, need_spectra)`
- `config.set_inference_model(ckpt, device, batch_size, num_workers)`

Configs round-trip as **pickle files**, YAML directories, or HDF attributes. `load_config()` auto-detects which. The modern workflow (via spectuner-extract-spectrum) produces a pickle config that can be loaded directly:

```python
config = spectuner.load_config("spectuner_config.pkl")  # from extract-spectrum --write-pickle
```

If starting from raw `.dat` files and a `extract_summary.txt` instead, build the config programmatically:

```python
import numpy as np
import spectuner

config = spectuner.load_default_config()
config.set_fname_db("/path/to/cdms.db")

# Read each spectrum and its metadata from the summary
spec = np.loadtxt("data/54_spw25.cube.I.pbcor.dat")  # freq_MHz, Tb_K
config.append_spectral_window(
    spec,
    beam_info=(0.000172898, 0.000122676),  # (BMAJ, BMIN) in degrees
    noise=0.376,
    T_bg=19.384,
    need_cmb=True
)
# Repeat for each spectral window...
```

## End-to-end workflow: extract → fit → plot

The three spectuner skills form a pipeline. Data flows from left to right:

```
[spectuner-extract-spectrum] → [spectuner-run] → [spectuner-plot-results]
     FITS/txt → .dat + .pkl          fit → .h5          .h5 → .png/.txt
```

### Starting from spectuner-extract-spectrum output

If the user has already run the extraction skill, they will have:
- `.dat` spectrum files (freq [MHz], intensity [K])
- `extract_summary.txt` (per-file metadata: noise, T_bg, beam size)
- `spectuner_config.pkl` (optional, generated with `--write-pickle`)

**Option A: Load the pre-built pickle config (fastest)**

```python
import spectuner

config = spectuner.load_config("extracted/spectuner_config.pkl")
# The pickle already contains all spectral windows, beam_info, noise, T_bg.
# Just set species and run.
config.set_ident_species(species=["CH3OH", "H2CO"])
config.set_n_process(8)

spectuner.run_individual_line_id(config, "results/")
spectuner.run_combining_line_id(config, "results/")
```

**Option B: Build config from summary + .dat files**

Use this when no pickle was generated, or when you need to select a subset of windows.

```python
import numpy as np
import spectuner

config = spectuner.load_default_config()
config.set_fname_db("/path/to/cdms.db")

# Read metadata from extract_summary.txt
# Format: name freq_start freq_end T_bg noise bmaj bmin n_pixels aperture status
summary = np.genfromtxt("extracted/extract_summary.txt", skip_header=3,
                        dtype=None, encoding='utf-8',
                        names=['name','f1','f2','T_bg','noise','bmaj','bmin',
                               'n_pix','aperture','status'])

for row in summary:
    if row['status'] != 'ok':
        continue
    fname = f"extracted/{row['name']}.dat"
    spec = np.loadtxt(fname)  # freq_MHz, Tb_K
    beam = (float(row['bmaj']), float(row['bmin']))
    config.append_spectral_window(
        spec, beam_info=beam,
        noise=float(row['noise']),
        T_bg=float(row['T_bg']),
        need_cmb=True
    )
```

## Two main user-facing workflows

### 1. Line identification

```python
import spectuner

config = spectuner.load_default_config()
config.append_spectral_window(spec, beam_info=30., noise=1.)
config.set_fname_db("cdms.db")
config.set_ident_species(species=["CH3OH", "H2CO"])
config.set_n_process(8)

# Two-phase pipeline — both phases print tqdm bars
spectuner.run_individual_line_id(config, "results/")   # "Fitting" bar, one tick per species
spectuner.run_combining_line_id(config, "results/")    # "Combining <species>" bar
```

For broadband surveys, the individual phase dominates wall time; the bar shows `len(species)` total ticks.

### 2. Pixel-by-pixel cube fitting

```python
# Preprocess FITS cubes → compact HDF
file_list = [spectuner.CubeItem(line="spw1_line.fits", continuum="spw1_continuum.fits")]
pipeline = spectuner.CubePipeline(noise_factor_local=6., number_cut=3, v_LSR=0.)
pipeline.run(file_list, "cube.h5")     # "Checking invalid values..." bars during preprocessing

# Fit
config = spectuner.load_default_config()
config.set_fname_db("cdms.db")
config.set_inference_model(ckpt="weights.pt", device="cuda:0")
config.set_pixel_by_pixel_fitting(species=["CH3OH;v=0;"], loss_fn="chi2")
spectuner.fit_pixel_by_pixel(config, "cube.h5", "results.h5")   # "Predicting" + "Optimizing" bars

# Access results
cube_mgr = spectuner.HDFCubeManager("cube.h5")
T_ex = cube_mgr.load_pred_data("results.h5", "CH3OH;v=0;/T_ex")
cube_mgr.pred_data_to_fits("results.h5", "fits/", add_v_LSR=True)
```

The neural network was trained only for `v_offset ∈ [-12, +12] km/s` — if the source has a larger LSR velocity, set `v_LSR` correctly during preprocessing so the residual velocity falls inside that range.

## Running fits — directory structure, timing, and reporting

Every spectuner run must be organized, timed, and summarized. Follow these rules whenever executing a fit on the user's behalf.

### 1. Create a timestamped output directory

Never overwrite previous results. Create a new directory with a timestamp. **All outputs go here**: the Python script, fit results, summary table, and any plots.

```python
import os
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
base_name = "results_CH3OH"  # descriptive, based on species or project
save_dir = f"{base_name}_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
```

### 2. Save the Python script inside the output directory

Write the exact Python script that performs the fit into the output directory so the run is fully reproducible:

```python
script_path = os.path.join(save_dir, "run_fit.py")
with open(script_path, "w") as f:
    f.write(script_content)
```

The script should be self-contained — it imports spectuner, loads the config, runs the fit, and saves results. Do this before executing the fit, not after.

### 3. Time the fit and collect metadata

Spectuner does not record wall time internally. Wrap the fit call with `time.time()`:

```python
import time

start_time = time.time()
spectuner.run_individual_line_id(config, save_dir)
end_time = time.time()
wall_time = end_time - start_time  # seconds
```

Also collect the spectral frequency coverage from the config:

```python
freq_ranges = []
for obs in config["obs_info"]:
    spec = obs["spec"]
    freq_ranges.append((float(spec[0, 0]), float(spec[-1, 0])))
total_freq = sum(f2 - f1 for f1, f2 in freq_ranges)
```

### 4. Generate a results summary after fitting

After the fit completes, load the result, extract the physical parameters, and produce a summary table:

```python
import spectuner
from spectuner.sl_model import ParameterManager

# Load the identification result
ident_result = spectuner.load_previous_ident_result(save_dir)

# Statistics: number of master groups and individual molecules
stats = ident_result.derive_stats_dict()

# DataFrame of identified molecules: name, score, num_tp_i, etc.
df_mol = ident_result.derive_df_mol(max_order=3)

# Extract physical parameters: theta, T_ex, N_tot, delta_v, v_offset
param_mgr = ParameterManager.from_config(ident_result.specie_list, config)
params_mol = param_mgr.derive_params(ident_result.x)  # shape (n_species, 5)

# Build name -> params mapping (order follows specie_list)
name_to_params = {}
idx = 0
for mol_item in ident_result.specie_list:
    for name in mol_item["species"]:
        name_to_params[name] = params_mol[idx]
        idx += 1

# Attach parameters to the DataFrame
for i, row in df_mol.iterrows():
    name = row["name"]
    if name in name_to_params:
        p = name_to_params[name]
        df_mol.at[i, "theta"]    = p[0]    # arcsec
        df_mol.at[i, "T_ex"]     = p[1]    # K
        df_mol.at[i, "N_tot"]    = p[2]    # cm^-2
        df_mol.at[i, "delta_v"]  = p[3]    # km/s
        df_mol.at[i, "v_offset"] = p[4]    # km/s
```

Present this information concisely to the user:

```
=== Fit Summary ===
Output directory: results_CH3OH_20250529_143022/
Total frequency coverage: 234.5 - 356.2 GHz (121.7 GHz total)
Fit duration: 18 min 32 s

Species identified: 12

Molecule summary:
  name                lines  score    theta    T_ex    N_tot       delta_v  v_offset
  ----------------------------------------------------------------------------------
  CH3OH;v=0;           23   0.923    1.25    145.2   1.23e+16    2.31      3.45
  H2CO;v=0;            18   0.891    0.89    112.5   8.76e+15    1.98      3.42
  CH3OCHO;v=0;         15   0.754    1.56    178.3   2.45e+16    2.67      3.38
  ...
```

The `lines` column comes from `num_tp_i` (true positives in the individual phase). If `num_tp_i` is not available, use `num_tp_c` (combine phase) or any count column present in the DataFrame.

### 5. Write summary.txt to the output directory

Save the summary to a text file inside the output directory:

```python
summary_path = os.path.join(save_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write(f"# Spectuner Fit Summary\n")
    f.write(f"# Generated: {timestamp}\n")
    f.write(f"Output directory: {save_dir}\n\n")

    f.write(f"## Spectral coverage\n")
    for i, (f1, f2) in enumerate(freq_ranges):
        f.write(f"  Window {i+1}: {f1:.3f} - {f2:.3f} MHz ({(f2-f1)/1e3:.3f} GHz)\n")
    f.write(f"  Total: {total_freq/1e3:.3f} GHz\n\n")

    f.write(f"## Fit performance\n")
    f.write(f"  Duration: {int(wall_time // 60)} min {wall_time % 60:.1f} s\n\n")

    f.write(f"## Identification results\n")
    f.write(f"  Species identified: {stats['n_mol']}\n")
    f.write(f"  Master groups: {stats['n_master']}\n\n")

    f.write(f"## Molecule table\n")
    f.write(f"# {'name':<22} {'lines':>5} {'score':>7} "
            f"{'theta':>7} {'T_ex':>7} {'N_tot':>12} {'delta_v':>7} {'v_off':>7}\n")
    for _, row in df_mol.iterrows():
        n_lines = row.get('num_tp_i', row.get('num_tp_c', 0))
        f.write(f"  {row['name']:<22} {n_lines:>5} {row['score']:>7.4f} "
                f"{row.get('theta', float('nan')):>7.2f} "
                f"{row.get('T_ex', float('nan')):>7.1f} "
                f"{row.get('N_tot', float('nan')):>12.3e} "
                f"{row.get('delta_v', float('nan')):>7.2f} "
                f"{row.get('v_offset', float('nan')):>7.2f}\n")
```

The `summary.txt` uses fixed-width columns for readability. Each row shows the molecule name, number of identified lines, fitting score, and the five physical parameters: `theta` (arcsec), `T_ex` (K), `N_tot` (cm^-2), `delta_v` (km/s), `v_offset` (km/s).

### 6. Full example — line identification with full reporting

Here is a complete, self-contained script that follows all the rules above. Save it as `run_fit.py` in the output directory before executing:

```python
#!/usr/bin/env python3
"""Spectuner line identification with full reporting."""

import os
import time
import numpy as np
import spectuner
from spectuner.sl_model import ParameterManager
from datetime import datetime

# --- 1. Create output directory ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"results_{timestamp}"
os.makedirs(save_dir, exist_ok=True)
print(f"Output directory: {save_dir}")

# --- 2. Load config ---
# Option A: from pickle (generated by spectuner-extract-spectrum)
# config = spectuner.load_config("extracted/spectuner_config.pkl")

# Option B: build from .dat files
config = spectuner.load_default_config()
config.set_fname_db("/path/to/cdms.db")

spec = np.loadtxt("data/54_spw25.cube.I.pbcor.dat")
config.append_spectral_window(
    spec, beam_info=(0.000173, 0.000123),
    noise=0.376, T_bg=19.384, need_cmb=True
)

# --- 3. Configure fitting ---
config.set_ident_species(species=["CH3OH", "H2CO", "CH3OCHO"])
config.set_n_process(8)

# --- 4. Run fit with timing ---
start_time = time.time()
spectuner.run_individual_line_id(config, save_dir)
spectuner.run_combining_line_id(config, save_dir)
end_time = time.time()
wall_time = end_time - start_time

# --- 5. Load results and generate summary ---
ident_result = spectuner.load_previous_ident_result(save_dir)
stats = ident_result.derive_stats_dict()
df_mol = ident_result.derive_df_mol(max_order=3)

# Extract physical parameters
param_mgr = ParameterManager.from_config(ident_result.specie_list, config)
params_mol = param_mgr.derive_params(ident_result.x)

name_to_params = {}
idx = 0
for mol_item in ident_result.specie_list:
    for name in mol_item["species"]:
        name_to_params[name] = params_mol[idx]
        idx += 1

for i, row in df_mol.iterrows():
    name = row["name"]
    if name in name_to_params:
        p = name_to_params[name]
        df_mol.at[i, "theta"]    = p[0]
        df_mol.at[i, "T_ex"]     = p[1]
        df_mol.at[i, "N_tot"]    = p[2]
        df_mol.at[i, "delta_v"]  = p[3]
        df_mol.at[i, "v_offset"] = p[4]

# Frequency coverage
freq_ranges = []
for obs in config["obs_info"]:
    s = obs["spec"]
    freq_ranges.append((float(s[0, 0]), float(s[-1, 0])))
total_freq = sum(f2 - f1 for f1, f2 in freq_ranges)

# Write summary.txt
summary_path = os.path.join(save_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write(f"# Spectuner Fit Summary\n")
    f.write(f"# Generated: {timestamp}\n")
    f.write(f"Output directory: {save_dir}\n\n")
    f.write(f"## Spectral coverage\n")
    for i, (f1, f2) in enumerate(freq_ranges):
        f.write(f"  Window {i+1}: {f1:.3f} - {f2:.3f} MHz\n")
    f.write(f"  Total: {total_freq/1e3:.3f} GHz\n\n")
    f.write(f"## Fit performance\n")
    f.write(f"  Duration: {int(wall_time // 60)} min {wall_time % 60:.1f} s\n\n")
    f.write(f"## Identification results\n")
    f.write(f"  Species identified: {stats['n_mol']}\n")
    f.write(f"  Master groups: {stats['n_master']}\n\n")
    f.write(f"## Molecule table\n")
    f.write(f"# {'name':<22} {'lines':>5} {'score':>7} "
            f"{'theta':>7} {'T_ex':>7} {'N_tot':>12} {'delta_v':>7} {'v_off':>7}\n")
    for _, row in df_mol.iterrows():
        n_lines = row.get('num_tp_i', row.get('num_tp_c', 0))
        f.write(f"  {row['name']:<22} {n_lines:>5} {row['score']:>7.4f} "
                f"{row.get('theta', float('nan')):>7.2f} "
                f"{row.get('T_ex', float('nan')):>7.1f} "
                f"{row.get('N_tot', float('nan')):>12.3e} "
                f"{row.get('delta_v', float('nan')):>7.2f} "
                f"{row.get('v_offset', float('nan')):>7.2f}\n")

print(f"Summary written to: {summary_path}")
```

## Working with the codebase

- **Testing**: `pytest` from the project root. Tests live in `spectuner/tests/`. `setup.cfg` configures coverage. Full coverage of `sl_model` requires the CDMS database.
- **Public API surface**: anything re-exported in `spectuner/__init__.py`. When adding new public symbols, update the relevant submodule's `__init__.py` or `__all__`.
- **`Config` is a dict subclass** — direct key access (`config["sl_model"]["fname_db"]`) works, but the setters validate and document; prefer them in user-facing examples.
- **Numba-jitted code**: `sl_model` and parts of `peaks` use numba for hot loops. Changes to function signatures invalidate the jit cache; expect the first run after a change to spend extra seconds compiling.
- **The two-phase identification design** is load-bearing — "single" / "individual" fits each species independently to find candidates; "combine" fits surviving candidates jointly to handle line blending. They have separate entry points (`run_individual_line_id` vs `run_combining_line_id`) and separate result formats. Do not conflate them.
- **Species naming follows CDMS** conventions, often with vibration/state suffixes like `"CH3OH;v=0;"`. The strings key into the database; be precise.
- **Multiprocessing**: the optimizer uses `multiprocessing.Pool`. On macOS or in notebooks the default `spawn` start method can cause issues — if a fit hangs immediately with no tqdm output, suspect this first.

## Citing / attribution

Qiu et al. 2025, ApJS, 277, 21 — `doi:10.3847/1538-4365/adaeba`, `arXiv:2408.06004`. Point users here when they ask about citing.

## Documentation

Full docs: https://spectuner.readthedocs.io/. Source lives under `docs/source/` (Sphinx). Tutorials are Jupyter notebooks under `docs/source/notebooks/`.

## Skill workflow and handoffs

This skill sits in the middle of the spectuner pipeline. Know when to hand off to the neighboring skills:

### Before this skill: spectuner-extract-spectrum

If the user has raw FITS cubes or plain-text spectra and has not yet extracted them, they need [spectuner-extract-spectrum](spectuner-extract-spectrum) first. Signs:
- They mention FITS cubes, ALMA data, or raw telescope data
- They have `.fits` files but no `.dat` spectra yet
- They need to compute noise, beam size, or continuum level
- They say things like "prepare data", "extract spectrum", "FITS to txt"

**Handoff trigger**: when the user asks about data preparation, extraction, or converting FITS to spectuner input, invoke spectuner-extract-spectrum.

### After this skill: spectuner-plot-results

Once a fit completes, the user typically wants to visualize results. Automatically invoke [spectuner-plot-results](spectuner-plot-results) when:
- The user says "plot", "show", "visualize", "画图", "绘制" in the context of fitting results
- A fit just finished and the user asks what to do next
- They mention peaks, spectral overview, parameter maps, or molecule identification plots
- They have `.h5` or `.pkl` result files and want to inspect them

**Default behavior**: after running a spectuner fit (individual, combining, or pixel-by-pixel), if the user does not specify a next step, proactively offer to plot the results by invoking spectuner-plot-results. Do this before moving on to any other task.

When invoking spectuner-plot-results, pass the **same output directory** so plots are saved alongside the fit results and summary. The directory should contain:

```
results_CH3OH_20250529_143022/
├── run_fit.py              # the script that performed the fit
├── identify_results.h5     # spectuner fitting results
├── summary.txt             # text summary with molecules and parameters
├── peak_plot.png           # (from plot-results) peak detail plots
├── spectral_overview.png   # (from plot-results) full spectrum overview
└── *.fits                  # (from plot-results) parameter maps (cube fitting)
```
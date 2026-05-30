---
name: spectuner-plot-results
description: >
  Use this skill whenever the user wants to visualize spectral line fitting
  results from the Spectuner framework. This includes: plotting individual
  molecule peaks with excitation temperature labels, creating full spectral
  overview plots after single-position fitting, and generating 2D parameter
  distribution maps after pixel-by-pixel cube fitting. Trigger this skill when
  the user mentions "plot", "show", "visualize", "画图", "展示", or "绘制"
  in the context of Spectuner fitting results, spectral line identification,
  peak plots, spectral plots, or cube fitting output files.
  This skill handles three main visualization scenarios:
  1. PeakPlot for individual molecule peaks with energy level annotations
  2. SpectralPlot for full spectrum overview after single-position fitting
  3. Converting pixel-level HDF results to FITS and plotting parameter maps.
---

# Spectuner Fitting Result Visualization

This skill provides guidance for visualizing Spectuner spectral line fitting
results in three common scenarios.

## Prerequisites

- A working Spectuner installation with access to `spectuner.spectral_plot`,
  `spectuner.identify`, and `spectuner.cube` modules.
- Matplotlib and Astropy installed.

## Scenario 1: PeakPlot for Individual Molecule Peaks

Use this after fitting a single-position spectrum to show each fitted peak of a
given molecule in individual subplots.

### Workflow

1. **Load the fitting result** into an `IdentResult` object.

   ```python
   from spectuner import load_previous_ident_result
   ident_result = load_previous_ident_result("path/to/identify_results.h5")
   ```

2. **Select a molecule** by key or name.

   ```python
   # Option A: extract by key
   key = 0  # molecule ID
   ident_sub = ident_result.extract(key)

   # Option B: extract by name
   # Find the key first
   for k, sub_dict in ident_result.specie_data.items():
       for name in sub_dict:
           if target_name in name:
               key = k
               break
   ident_sub = ident_result.extract(key)
   ```

3. **Get the fitted parameters** to annotate each peak with the excitation
   temperature. The parameters are stored as a 2D array of shape `(n_species,
   5)` in the order `[theta, T_ex, N_tot, delta_v, v_offset]`.

   ```python
   from spectuner.sl_model import ParameterManager
   # Create a ParameterManager from the config and species list
   param_mgr = ParameterManager.from_config(ident_result.specie_list, config)
   params_mol = param_mgr.derive_params(ident_result.x)
   # params_mol[i] gives [theta, T_ex, N_tot, delta_v, v_offset] for species i
   ```

4. **Create the PeakPlot** centered on each identified peak frequency.

   ```python
   from spectuner import PeakPlot

   # Get the frequencies of peaks identified for this molecule
   freqs = ident_sub.line_table.freq
   # Filter out None entries
   freqs = freqs[ident_sub.line_table.name != None]

   plot = PeakPlot(freqs, delta_v=20., n_col=4, plot_width=4, plot_height=3)
   ```

5. **Plot the observed spectrum** (black) and the **identified model spectrum**
   (red) on each subplot.

   ```python
   from spectuner import load_preprocess, get_freq_data, get_T_data

   # Observed spectrum
   obs_data = load_preprocess(config["obs_info"], clip=False)
   freq_data = get_freq_data(obs_data)
   T_data = get_T_data(obs_data)
   plot.plot_spec(freq_data, T_data, step_plot=True, ylim_factor=1.5,
                  color="k")

   # Model spectrum for this molecule (red)
   T_pred = ident_sub.get_T_pred()
   if T_pred is not None:
       plot.plot_spec(ident_sub.freq_data, T_pred, color="r")
   ```

6. **Draw vertical lines** for each identified peak.

   ```python
   plot.vlines(freqs, linestyle="--", color="r")
   ```

7. **Annotate E_u next to each vertical line** using `PeakPlot.vtexts()`.

   ```python
   from spectuner.sl_model import create_spectral_line_db

   sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])
   props = ident_sub.query_sl_dict(sl_db, key, name)

   if "E_up" in props:
       texts = np.array([f"{props['E_up'][j]:.1f}K" for j in range(len(freqs))])
       plot.vtexts(freqs, texts, h_txt_offset=0.02, v_txt_offset=0.92,
                   fontsize=9, color="darkgreen")
   ```

8. **Fix x-axis tick label overlap.** Limit ticks to 4 per subplot, no
   rotation, hide offset text.

   ```python
   from matplotlib.ticker import ScalarFormatter

   for i_a, ax in enumerate(plot.axes.flat):
       if i_a >= plot.n_plot:
           continue
       ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
       ax.xaxis.get_major_formatter().set_scientific(False)
       ax.tick_params(axis="x", labelrotation=0, labelsize=7)
       ax.locator_params(axis="x", nbins=4)
   ```

9. **Set axis labels and tighten spacing.** Add "Intensity [K]" to the leftmost
   column and "Frequency [MHz]" to the bottom row. Minimize subplot gaps.

   ```python
   import matplotlib.pyplot as plt

   axes = plot.axes
   n_plot = plot.n_plot
   n_col = 4  # same as passed to PeakPlot
   n_row = (n_plot + n_col - 1) // n_col

   for i_a, ax in enumerate(axes.flat):
       if i_a >= n_plot:
           continue
       # Leftmost column gets y-label
       if i_a % n_col == 0:
           ax.set_ylabel("Intensity [K]")
       # Bottom row gets x-label
       row = i_a // n_col
       is_last_row = (row == n_row - 1) or \
                     (row == n_row - 2 and i_a + n_col >= n_plot)
       if is_last_row:
           ax.set_xlabel("Frequency [MHz]")

   plt.subplots_adjust(wspace=0.15, hspace=0.1, top=0.98)
   ```

10. **Set the title** near the top of the figure, including all fitted
    parameters.

    ```python
    T_ex = params_mol[specie_idx, 1]
    N_tot = params_mol[specie_idx, 2]
    theta = params_mol[specie_idx, 0]
    delta_v = params_mol[specie_idx, 3]
    v_offset = params_mol[specie_idx, 4]

    fig = plot.fig
    fig.suptitle(
        f"Peaks for {name}, T_ex={T_ex:.1f}K, N_tot={N_tot:.2e}cm^-2, "
        f"theta={theta:.1f}, delta_v={delta_v:.1f}km/s, v_off={v_offset:.1f}km/s",
        fontsize=11, y=0.995
    )
    ```

11. **Export a peak details text file** listing every transition. The
    `query_sl_dict` method does **not** return quantum numbers, so they must be
    queried directly from the SQLite database by matching `freq_rest` (rest
    frequency) against `T_Frequency`.

    ```python
    import sqlite3

    # Query quantum numbers directly from the database
    db_path = config["sl_model"]["fname_db"]
    base_name = name + "%"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT T_Frequency, T_UpperStateQuantumNumbers, "
        "T_LowerStateQuantumNumbers FROM Transitions WHERE T_Name LIKE ?",
        (base_name,),
    )
    rows = cursor.fetchall()
    conn.close()

    # Build lookup: exact frequency string -> (qn_u, qn_l)
    qn_map = {}
    for freq_db, qn_u, qn_l in rows:
        key = f"{float(freq_db):.4f}"
        if key not in qn_map:
            qn_map[key] = (qn_u.strip() if qn_u else "-",
                           qn_l.strip() if qn_l else "-")

    # Match against props["freq_rest"] (rest/catalog frequency)
    qn_upper = []
    qn_lower = []
    for f in props["freq_rest"]:
        key = f"{float(f):.4f}"
        if key in qn_map:
            qn_upper.append(qn_map[key][0])
            qn_lower.append(qn_map[key][1])
        else:
            qn_upper.append("-")
            qn_lower.append("-")

    # Write file: quantum numbers in the LAST two columns.
    # Note: qn_upper/qn_lower are the raw quantum number strings from the CDMS
    # database (e.g. "ElecStateLabel = X; J = 22; Ka = 8; Kc = 15;"), not
    # just the J quantum number. Use wide columns or truncate if needed.
    v_off = params_mol[specie_idx, 4]
    with open("peak_details.txt", "w") as fp:
        fp.write("# idx  freq_catalog_MHz  freq_shifted_MHz  E_up_K  "
                 "E_low_K  A_ul  g_u  qn_upper  qn_lower\n")
        for j in range(len(props["freq"])):
            freq_cat = props["freq"][j]
            freq_raw = freq_cat / (1. - v_off / 3e5)
            E_up = props["E_up"][j]
            E_low = props.get("E_low", [np.nan] * n_lines)[j]
            A_ul = props.get("A_ul", [np.nan] * n_lines)[j]
            g_up = props.get("g_u", [np.nan] * n_lines)[j]
            qn_u = qn_upper[j]
            qn_l = qn_lower[j]
            fp.write(f"{j:3d}  {freq_raw:14.4f}  {freq_cat:14.4f}  "
                     f"{E_up:10.2f}  {E_low:10.2f}  {A_ul:14.4e}  {g_up:6.1f}  "
                     f"{qn_u}  {qn_l}\n")
    ```

    Alternatively, use the helper script:

    ```python
    from scripts.plot_peak_details import (
        plot_peak_details_with_annotations,
        save_peak_details_txt,
    )

    plot_peak_details_with_annotations(
        ident_result, config, key, name,
        include_tau=True,  # or False, ask user first
        delta_v=20.,
        save_path="peak_plot.png"
    )
    save_peak_details_txt(ident_result, config, key, name,
                          sl_db=sl_db, save_path="peak_details.txt")
    ```

### Important Notes

- The `delta_v` parameter controls the velocity width around each peak in km/s.
  A smaller value zooms in closer. Use `delta_v=20.` for a tight view around
  each peak.
- Overlapping windows are automatically merged by `PeakPlot`.
- When extracting a molecule with `extract()`, the returned `IdentResult` only
  contains peaks associated with that molecule.
- Use `plt.subplots_adjust(wspace=0.15, hspace=0.1, top=0.98)` and
  `fig.suptitle(..., y=0.995)` to bring the main title close to the subplots.
- Always add y-labels ("Intensity [K]") to the leftmost column and x-labels
  ("Frequency [MHz]") to the bottom row of subplots.
- Overlay the model spectrum in red using `plot.plot_spec(ident_sub.freq_data,
  ident_sub.get_T_pred(), color="r")` so each subplot shows both the observed
  (black) and fitted (red) spectrum.
- Use `vtexts()` to place E_u annotations next to vertical lines, not generic
  `ax.text()` calls which can overlap when multiple peaks share a panel.

## Scenario 2: SpectralPlot for Full Spectrum Overview

Use this after single-position fitting to display all spectral windows with
only the fitted spectrum highlighted.

### Workflow

1. **Load the config and identification result**.

   ```python
   from spectuner import load_config, load_previous_ident_result
   config = load_config("path/to/config")
   ident_result = load_previous_ident_result("path/to/identify_results.h5")
   ```

2. **Create the SpectralPlot from the config**.

   ```python
   from spectuner import SpectralPlot

   plot = SpectralPlot.from_config(
       config,
       freq_per_row=1000.,  # MHz per row
       width=20.,
       height=3.,
       color="k"
   )
   ```

3. **Overlay the fitted spectrum** for all identified molecules combined.

   ```python
   plot.plot_ident_result(
       ident_result,
       show_lines=True,
       color="r",           # matched peaks
       color_blen="orange", # blended peaks
       color_fp="b",        # false positives
       fontsize=10
   )
   ```

4. **Set y-axis limits** based on noise.

   ```python
   import numpy as np
   noise = np.mean([item["noise"] for item in config["obs_info"]])
   plot.set_ylim(-10. * noise, 100. * noise)
   ```

5. **Save or display** with the title pulled close to the subplots.

   ```python
   import matplotlib.pyplot as plt

   fig = plot.axes[0].figure
   fig.suptitle("Full spectrum overview", fontsize=12, y=0.995)
   fig.subplots_adjust(top=0.98)
   fig.savefig("spectral_overview.png", dpi=150, bbox_inches="tight")
   # or plt.show()
   ```

### To Show Only Specific Molecules

If the user wants to highlight only specific molecules, pass `key` or `name` to
`plot_ident_result`:

```python
# Show only molecule with key=0
plot.plot_ident_result(ident_result, key=0, show_lines=True, color="r")

# Show only molecule with specific name
plot.plot_ident_result(ident_result, name="CH3OH;v=0", show_lines=True, color="r")
```

## Scenario 3: Pixel-Level Fitting Results to Parameter Maps

Use this after pixel-by-pixel cube fitting to convert the HDF5 result file to
FITS files and plot 2D distributions of the 5 fitted parameters.

### Workflow

1. **Convert HDF5 results to FITS files**.

   ```python
   from spectuner import HDFCubeManager

   cube_mgr = HDFCubeManager("path/to/cube_data.h5")
   cube_mgr.pred_data_to_fits(
       "path/to/fitting_results.h5",
       save_dir="output_fits",
       add_v_LSR=True,
       overwrite=True
   )
   ```

   This creates a directory structure:

   ```
   output_fits/
   ├── <species_name_1>/
   │   ├── theta.fits
   │   ├── T_ex.fits
   │   ├── N_tot.fits
   │   ├── delta_v.fits
   │   └── v_offset.fits
   ├── <species_name_2>/
   │   └── ...
   └── total/
       └── ... (if multiple species)
   ```

   Alternatively, for more control over the conversion and plotting, use the
   helper script:

   ```python
   from scripts.plot_param_maps import convert_and_plot_params
   convert_and_plot_params(
       cube_file="path/to/cube_data.h5",
       result_file="path/to/fitting_results.h5",
       save_dir="output_plots",
       species=["CH3OH;v=0"],  # or None for all
       log_params=["N_tot", "theta"],  # plot these in log scale
       show_cb_label=True,
       cmap="viridis"
   )
   ```

2. **Plot parameter maps manually** (if not using the helper script).

   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from astropy.io import fits
   from matplotlib.colors import LogNorm

   PARAM_NAMES = ["theta", "T_ex", "N_tot", "delta_v", "v_offset"]
   PARAM_UNITS = ["arcsec", "K", "cm-2", "km/s", "km/s"]
   LOG_PARAMS = {"N_tot", "theta"}  # typically log-scaled parameters

   species = "CH3OH;v=0"
   fits_dir = f"output_fits/{species}"

   fig, axes = plt.subplots(2, 3, figsize=(18, 12))
   axes = axes.flatten()

   for idx, (param, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
       ax = axes[idx]
       data = fits.open(f"{fits_dir}/{param}.fits")[0].data
       header = fits.open(f"{fits_dir}/{param}.fits")[0].header

       # Determine normalization
       if param in LOG_PARAMS:
           vmin, vmax = np.nanpercentile(data[data > 0], [1, 99])
           norm = LogNorm(vmin=vmin, vmax=vmax)
           label = f"{param} [{unit}]"
       else:
           vmin, vmax = np.nanpercentile(data, [1, 99])
           norm = None
           label = f"{param} [{unit}]"

       im = ax.imshow(data, origin="lower", cmap="viridis", norm=norm,
                      vmin=vmin, vmax=vmax)
       ax.set_title(f"{param}")

       # Colorbar with scientific notation for log-scale parameters
       cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
       if param in LOG_PARAMS:
           cb.formatter.set_powerlimits((0, 0))
           cb.update_ticks()
       cb.set_label(label)

   # Remove empty subplot
   if len(PARAM_NAMES) < len(axes):
       for ax in axes[len(PARAM_NAMES):]:
           ax.remove()

   plt.tight_layout()
   plt.savefig(f"{species}_params_map.png", dpi=150, bbox_inches="tight")
   plt.show()
   ```

### Important Notes for Scenario 3

- The HDF5 result file from pixel-level fitting stores parameters in the order
  defined by `spectuner.cube.PARAM_NAMES` = `("theta", "T_ex", "N_tot",
  "delta_v", "v_offset")`.
- The `N_tot` (column density) and `theta` (source size) parameters are
  typically stored in log scale if `param_info["N_tot"]["is_log"]` and
  `param_info["theta"]["is_log"]` are `True` in the config. When plotting,
  use `LogNorm` for these parameters and format colorbar labels with scientific
  notation.
- Use `np.nanpercentile(data, [1, 99])` to set vmin/vmax robustly, avoiding
  extreme outliers.
- The FITS header from the original cube is preserved, so you can optionally
  add WCS coordinates to the plots using `astropy.wcs.WCS`.

## General Guidelines

- Always use `fig.savefig()` with `bbox_inches="tight"` and a reasonable DPI
  (150-300) when saving figures.
- For `PeakPlot` and `SpectralPlot`, the y-axis limits should be set based on
  the noise level (`config["obs_info"][i]["noise"]`) to ensure good
  visualization.
- When plotting multiple molecules in one figure, use different colors for each
  molecule and include a legend.
- The `IdentResult.specie_data` dict contains per-molecule information
  including the fitted parameters, scores, and counts. Use
  `ident_result.derive_df_mol()` or `ident_result.derive_df_mol_master()` to
  get a pandas DataFrame summary.

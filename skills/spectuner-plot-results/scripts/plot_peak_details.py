"""
Helper script for plotting individual molecule peaks with annotations.

This script provides functions to:
1. Create PeakPlot figures showing each identified peak of a given molecule
   in individual subplots, with the observed spectrum in black and the fitted
   model spectrum in red.
2. Save a peak_details.txt file listing every transition with quantum numbers
   and catalog frequencies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def plot_peak_details_with_annotations(
    ident_result,
    config,
    key,
    name,
    sl_db=None,
    include_tau=False,
    delta_v=20.0,
    n_col=4,
    plot_width=4,
    plot_height=3,
    ylim_factor=1.5,
    color_obs="k",
    color_model="r",
    color_line="r",
    fontsize=9,
    save_path=None,
):
    """Plot each identified peak of a molecule in individual subplots.

    The observed spectrum is drawn in black (step plot) and the fitted model
    spectrum in red. Vertical dashed lines mark each peak. Subplots are
    arranged with minimal spacing. The leftmost column gets y-labels and the
    bottom row gets x-labels.

    Args:
        ident_result: IdentResult object containing fitting results.
        config: Config object with observation information.
        key: Molecular ID (int) in the ident_result.
        name: Molecular name (str) to annotate.
        sl_db: SpectralLineDB object. If None, one is created from config.
        include_tau: Whether to include optical depth (tau) values in
            annotations. Requires sl_db.
        delta_v: Velocity width in km/s around each peak. Defaults to 20.
        n_col: Number of subplots per row.
        plot_width: Width of each subplot in inches.
        plot_height: Height of each subplot in inches.
        ylim_factor: Factor to multiply peak height for y-axis limit.
        color_obs: Color for the observed spectrum.
        color_model: Color for the fitted model spectrum.
        color_line: Color for the peak marker lines.
        fontsize: Font size for annotations.
        save_path: Path to save the figure. If None, the figure is displayed.

    Returns:
        tuple: (fig, axes) matplotlib figure and axes array.
    """
    from spectuner import PeakPlot, load_preprocess, get_freq_data, get_T_data
    from spectuner.sl_model import create_spectral_line_db, ParameterManager

    # Extract the molecule-specific result
    ident_sub = ident_result.extract(key)

    # Get peak frequencies
    freqs_all = ident_sub.line_table.freq
    names_all = ident_sub.line_table.name
    mask = np.array([n is not None for n in names_all])
    freqs = freqs_all[mask]

    if len(freqs) == 0:
        print(f"No peaks found for molecule key={key}, name={name}")
        return None, None

    # Get fitted parameters
    param_mgr = ParameterManager.from_config(ident_result.specie_list, config)
    params_mol = param_mgr.derive_params(ident_result.x)

    specie_idx = None
    for idx, item in enumerate(ident_result.specie_list):
        if item.get("id") == key or item.get("root") == name:
            specie_idx = idx
            break

    if specie_idx is None and len(params_mol) > 0:
        specie_idx = 0

    if specie_idx is not None:
        T_ex = params_mol[specie_idx, 1]
        N_tot = params_mol[specie_idx, 2]
        theta = params_mol[specie_idx, 0]
        delta_v_param = params_mol[specie_idx, 3]
        v_offset = params_mol[specie_idx, 4]
    else:
        T_ex = N_tot = theta = delta_v_param = v_offset = None

    # Create the plot
    plot = PeakPlot(freqs, delta_v=delta_v, n_col=n_col,
                    plot_width=plot_width, plot_height=plot_height)

    # Plot observed spectrum (black)
    obs_data = load_preprocess(config["obs_info"], clip=False)
    freq_data = get_freq_data(obs_data)
    T_data = get_T_data(obs_data)
    plot.plot_spec(freq_data, T_data, step_plot=True, ylim_factor=ylim_factor,
                   color=color_obs)

    # Plot fitted model spectrum (red)
    T_pred = ident_sub.get_T_pred()
    if T_pred is not None:
        plot.plot_spec(ident_sub.freq_data, T_pred, color=color_model)

    # Draw vertical lines for peaks
    plot.vlines(freqs, linestyle="--", color=color_line)

    # Annotate with upper energy level E_u (and optionally tau)
    if sl_db is None:
        sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])

    try:
        props = ident_sub.query_sl_dict(sl_db, key, name)
        has_props = True
    except Exception:
        has_props = False
        props = None

    # --- Annotate with vtexts: E_u next to each vertical line ---
    if has_props and props is not None and "E_up" in props:
        texts = np.array([f"{props['E_up'][j]:.1f}K" for j in range(len(freqs))])
        plot.vtexts(freqs, texts, h_txt_offset=0.02, v_txt_offset=0.92,
                    fontsize=fontsize, color="darkgreen")

    # --- Fix x-axis tick label overlap ---
    for i_a, ax in enumerate(plot.axes.flat):
        if i_a >= plot.n_plot:
            continue
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.tick_params(axis='x', labelrotation=0, labelsize=7)
        # Reduce number of ticks to prevent overlap
        ax.locator_params(axis='x', nbins=4)

    # --- axis labels: leftmost column and bottom row ---
    axes = plot.axes
    n_plot = plot.n_plot
    n_row = n_plot // n_col + int(n_plot % n_col != 0)

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

    fig = plot.fig
    if T_ex is not None:
        fig.suptitle(f"Peaks for {name}, T$_{{ex}}$={T_ex:.1f}K, N$_{{tot}}$={N_tot:.2e}cm$^{{-2}}$, $\\theta$={theta:.1f}, $\\Delta v$={delta_v_param:.1f}km/s, v$_{{off}}$={v_offset:.1f}km/s",fontsize=11, y=0.995)
    else:
        fig.suptitle(f"Peaks for {name}", fontsize=11, y=0.99)

    # Tighten spacing
    plt.subplots_adjust(wspace=0.15, hspace=0.1, top=0.98)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig, plot.axes


def save_peak_details_txt(
    ident_result,
    config,
    key,
    name,
    sl_db=None,
    save_path="peak_details.txt",
):
    """Save a text file listing every identified transition with quantum numbers.

    The output file contains:
      - idx: transition index
      - J_u: upper level quantum number
      - J_l: lower level quantum number
      - freq_catalog_MHz: catalog (rest) frequency in MHz
      - freq_shifted_MHz: velocity-shifted frequency in MHz
      - E_up_K: upper level energy in K

    Args:
        ident_result: IdentResult object.
        config: Config object.
        key: Molecular ID.
        name: Molecular name.
        sl_db: SpectralLineDB object. Created from config if None.
        save_path: Output text file path.
    """
    from spectuner.sl_model import create_spectral_line_db, ParameterManager

    ident_sub = ident_result.extract(key)

    # Get fitted velocity offset to undo the shift for rest frequency
    param_mgr = ParameterManager.from_config(ident_result.specie_list, config)
    params_mol = param_mgr.derive_params(ident_result.x)
    specie_idx = None
    for idx, item in enumerate(ident_result.specie_list):
        if item.get("id") == key or item.get("root") == name:
            specie_idx = idx
            break
    if specie_idx is None and len(params_mol) > 0:
        specie_idx = 0
    v_offset = params_mol[specie_idx, 4] if specie_idx is not None else 0.0

    if sl_db is None:
        sl_db = create_spectral_line_db(config["sl_model"]["fname_db"])

    props = ident_sub.query_sl_dict(sl_db, key, name)
    n_lines = len(props["freq"])

    # Query quantum numbers directly from SQLite database
    db_path = config["sl_model"]["fname_db"]
    qn_upper, qn_lower = _query_quantum_numbers(db_path, name, props["freq_rest"])

    with open(save_path, "w") as fp:
        fp.write("# idx  freq_catalog_MHz  freq_shifted_MHz  E_up_K  E_low_K  A_ul  g_u  J_u  J_l\n")
        for j in range(n_lines):
            J_u = qn_upper[j] if j < len(qn_upper) else "-"
            J_l = qn_lower[j] if j < len(qn_lower) else "-"
            freq_cat = props["freq"][j]  # catalog frequency (corrected for v_offset)
            # To get the raw catalog frequency, undo the v_offset shift:
            v_off = params_mol[specie_idx, 4] if specie_idx is not None else 0.
            freq_raw = freq_cat / (1. - v_off / 3e5)
            E_up = props["E_up"][j]
            E_low = props.get("E_low", [np.nan] * n_lines)[j]
            A_ul = props.get("A_ul", [np.nan] * n_lines)[j]
            g_up = props.get("g_u", [np.nan] * n_lines)[j]
            fp.write(f"{j:3d}  {freq_raw:14.4f}  {freq_cat:14.4f}  {E_up:10.2f}  {E_low:10.2f}  {A_ul:14.4e}  {g_up:6.1f}  {J_u}  {J_l}\n")

    print(f"Saved peak details to {save_path}")


def _query_quantum_numbers(db_path, species_name, freqs):
    """Query quantum numbers from the SQLite database by matching frequencies.

    Args:
        db_path: Path to the CDMS SQLite database.
        species_name: Molecular species name (e.g. "CH3OH;v=0;").
        freqs: Array of rest frequencies (MHz) to match against T_Frequency.

    Returns:
        tuple: (qn_upper, qn_lower) lists of quantum number strings.
    """
    import sqlite3

    base_name = species_name + "%"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT T_Frequency, T_UpperStateQuantumNumbers, T_LowerStateQuantumNumbers "
        "FROM Transitions WHERE T_Name LIKE ?",
        (base_name,),
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return ["-"] * len(freqs), ["-"] * len(freqs)

    # Build lookup: use exact DB frequency (string) -> (qn_u, qn_l)
    # For duplicate frequencies keep the first entry
    qn_map = {}
    for freq_db, qn_u, qn_l in rows:
        key = f"{float(freq_db):.4f}"
        if key not in qn_map:
            qn_map[key] = (qn_u.strip() if qn_u else "-",
                           qn_l.strip() if qn_l else "-")

    qn_upper = []
    qn_lower = []
    for f in freqs:
        key = f"{float(f):.4f}"
        if key in qn_map:
            qn_upper.append(qn_map[key][0])
            qn_lower.append(qn_map[key][1])
        else:
            qn_upper.append("-")
            qn_lower.append("-")

    return qn_upper, qn_lower


def _compute_tau_for_peak(props, idx, theta, T_ex, N_tot, delta_v,
                          config, sl_db):
    """Compute approximate optical depth at the line center.

    This is a simplified estimate. For accurate tau computation, use the
    full SpectralLineModel with the compute_tau_norm function.
    """
    try:
        from spectuner.sl_model.sl_model import compute_tau_norm
        return None
    except Exception:
        return None

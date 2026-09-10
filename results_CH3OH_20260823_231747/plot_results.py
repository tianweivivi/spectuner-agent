#!/usr/bin/env python3
"""Plot CH3OH fitting results for IRAS 16293-2422 B."""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

import spectuner
from spectuner import PeakPlot, SpectralPlot
from spectuner import load_preprocess, get_freq_data, get_T_data
from spectuner.sl_model import ParameterManager, create_spectral_line_db

SAVE_DIR = "/Users/sdkkk/Desktop/spectra_skills/results_CH3OH_20260823_231747"
DATA_DIR = "/Users/sdkkk/Desktop/spectra_skills/IRAS16293_spectra"
CDMS_DB = "/Users/sdkkk/spectuner_data/cdms_sqlite__official-version__2024-01-01.db"
BMAJ = 0.30 / 3600.0
BMIN = 0.20 / 3600.0

# --- Reconstruct config ---
print("Reconstructing config...")
files = sorted(
    glob.glob(os.path.join(DATA_DIR, "*.dat")),
    key=lambda f: float(np.loadtxt(f, max_rows=1)[0])
)
config = spectuner.load_default_config()
config.set_fname_db(CDMS_DB)
for f in files:
    spec = np.loadtxt(f)
    intensity = spec[:, 1]
    mad = np.median(np.abs(intensity - np.median(intensity)))
    noise = float(mad * 1.4826)
    config.append_spectral_window(
        spec, beam_info=(BMAJ, BMIN), noise=noise, T_bg=0.0, need_cmb=True
    )
config.set_param_info("theta",    is_log=False, bound=(0.0, 5.0))
config.set_param_info("T_ex",     is_log=False, bound=(10.0, 500.0))
config.set_param_info("N_tot",    is_log=True,  bound=(12.0, 19.0))
config.set_param_info("delta_v",  is_log=False, bound=(0.5, 10.0))
config.set_param_info("v_offset", is_log=False, bound=(-10.0, 10.0))
config.set_ident_species(species=["CH3OH"], collect_iso=True,
                         combine_iso=False, combine_state=False)
config.set_n_process(10)
config.set_optimizer("pso", n_swarm=28, n_trial=1, n_draw=50)

print(f"Loaded {len(config['obs_info'])} spectral windows")

# --- Load result ---
print("Loading identification result...")
ident_result = spectuner.load_previous_ident_result(
    os.path.join(SAVE_DIR, "identify_results_combine.h5")
)
df_mol = ident_result.derive_df_mol(max_order=3)
param_mgr = ParameterManager.from_config(ident_result.specie_list, config)
params_mol = param_mgr.derive_params(ident_result.x)

name_to_params = {}
idx = 0
for mol_item in ident_result.specie_list:
    for name in mol_item["species"]:
        name_to_params[name] = params_mol[idx]
        idx += 1

# --- Load observed spectrum ---
print("Loading observed spectrum...")
obs_data = load_preprocess(config["obs_info"], clip=False)
freq_data_obs = get_freq_data(obs_data)
T_data_obs = get_T_data(obs_data)
noise_avg = np.mean([item["noise"] for item in config["obs_info"]])


def get_peak_freqs(ident_sub):
    """Extract identified peak center frequencies from a sub-result."""
    lt = ident_sub.line_table
    mask = np.array([n is not None for n in lt.name])
    return lt.freq[mask]


# ============================================================
# Plot 1: SpectralPlot - Full spectrum overview
# ============================================================
print("\nCreating SpectralPlot (full spectrum overview)...")
try:
    plot = SpectralPlot.from_config(
        config, freq_per_row=1000., width=20., height=3., color="k"
    )
    plot.plot_ident_result(
        ident_result, show_lines=True,
        color="r", color_blen="orange", color_fp="b", fontsize=9
    )
    plot.set_ylim(-10. * noise_avg, 80. * noise_avg)

    fig = plot.axes[0].figure
    fig.suptitle(
        f"IRAS 16293-2422 B — CH3OH Family Spectral Overview "
        f"(ALMA Band 6, {len(config['obs_info'])} windows)",
        fontsize=12, y=0.998
    )
    fig.subplots_adjust(top=0.97)
    fpath = os.path.join(SAVE_DIR, "spectral_overview.png")
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"  Saved: {fpath}")
    plt.close(fig)
except Exception as e:
    print(f"  SpectralPlot error: {e}")
    import traceback
    traceback.print_exc()


# ============================================================
# Plot 2: PeakPlot for each identified species
# ============================================================
sl_db = create_spectral_line_db(CDMS_DB)

for specie_idx, mol_item in enumerate(ident_result.specie_list):
    for name in mol_item["species"]:
        print(f"\nCreating PeakPlot for {name}...")
        try:
            key = mol_item["id"]
            ident_sub = ident_result.extract(key)

            # Get peak frequencies
            freqs = get_peak_freqs(ident_sub)
            n_peaks = len(freqs)
            if n_peaks == 0:
                print(f"  No peaks found, skipping")
                continue
            print(f"  {n_peaks} peaks found")

            # Parameters
            p = name_to_params.get(name)
            if p is None:
                p = params_mol[specie_idx]
            T_ex, N_tot, theta = p[1], p[2], p[0]
            delta_v_p, v_offset = p[3], p[4]
            score_val = df_mol.iloc[specie_idx]["score"]

            n_col = min(4, n_peaks)
            plot = PeakPlot(freqs, delta_v=20., n_col=n_col,
                           plot_width=4, plot_height=3)

            # Observed spectrum (black)
            plot.plot_spec(freq_data_obs, T_data_obs, step_plot=True,
                          ylim_factor=1.5, color="k")

            # Model spectrum (red)
            T_pred = ident_sub.get_T_pred()
            if T_pred is not None:
                plot.plot_spec(ident_sub.freq_data, T_pred, color="r")

            # Vertical lines
            plot.vlines(freqs, linestyle="--", color="r")

            # Annotate E_u
            try:
                props = ident_sub.query_sl_dict(sl_db, key, name)
                if "E_up" in props:
                    n_annot = min(len(freqs), len(props["E_up"]))
                    texts = np.array(
                        [f"{props['E_up'][j]:.1f}K" for j in range(n_annot)]
                    )
                    plot.vtexts(freqs[:n_annot], texts,
                               h_txt_offset=0.02, v_txt_offset=0.92,
                               fontsize=9, color="darkgreen")
            except Exception as e:
                print(f"  E_u annotation: {e}")

            # Fix x-axis
            for i_a, ax in enumerate(plot.axes.flat):
                if i_a >= plot.n_plot:
                    continue
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useOffset=False))
                ax.xaxis.get_major_formatter().set_scientific(False)
                ax.tick_params(axis="x", labelrotation=0, labelsize=7)
                ax.locator_params(axis="x", nbins=4)

            # Axis labels
            n_row = (plot.n_plot + n_col - 1) // n_col
            for i_a, ax in enumerate(plot.axes.flat):
                if i_a >= plot.n_plot:
                    continue
                if i_a % n_col == 0:
                    ax.set_ylabel("Intensity [K]")
                row = i_a // n_col
                is_last = (row == n_row - 1) or \
                          (row == n_row - 2 and i_a + n_col >= plot.n_plot)
                if is_last:
                    ax.set_xlabel("Frequency [MHz]")

            plt.subplots_adjust(wspace=0.15, hspace=0.1, top=0.98)

            fig = plot.fig
            fig.suptitle(
                f"{name}  T_ex={T_ex:.1f}K  "
                f"N_tot={N_tot:.2e}cm⁻²  "
                f"θ={theta:.2f}″  Δv={delta_v_p:.2f}km/s  "
                f"v_off={v_offset:.2f}km/s  score={score_val:.2f}",
                fontsize=10, y=0.995
            )

            safe_name = name.replace(";", "_").replace("=", "")
            fpath = os.path.join(SAVE_DIR, f"peakplot_{safe_name}.png")
            fig.savefig(fpath, dpi=150, bbox_inches="tight")
            print(f"  Saved: {fpath}")
            plt.close(fig)

        except Exception as e:
            print(f"  PeakPlot error for {name}: {e}")
            import traceback
            traceback.print_exc()

print("\nDone! All plots saved to:", SAVE_DIR)

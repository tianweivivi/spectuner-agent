#!/usr/bin/env python3
"""Spectuner CH3CHO identification for IRAS 16293-2422 B."""

import os
import sys
import time
import glob
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

import spectuner
from spectuner import PeakPlot, SpectralPlot
from spectuner import load_preprocess, get_freq_data, get_T_data
from spectuner.sl_model import ParameterManager, create_spectral_line_db
from datetime import datetime

DATA_DIR = "/Users/sdkkk/Desktop/spectra_skills/IRAS16293_spectra"
CDMS_DB = "/Users/sdkkk/spectuner_data/cdms_sqlite__official-version__2024-01-01.db"
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
BMAJ = 0.30 / 3600.0
BMIN = 0.20 / 3600.0


def build_config():
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
    config.set_ident_species(species=["CH3CHO"], collect_iso=True,
                             combine_iso=False, combine_state=False)
    config.set_n_process(10)
    config.set_optimizer("pso", n_swarm=28, n_trial=1, n_draw=50)
    return config, files


def run_fit(config):
    start = time.time()
    print("\nPhase 1: Individual line identification (CH3CHO)")
    spectuner.run_individual_line_id(config, SAVE_DIR)
    t1 = time.time() - start
    print(f"Phase 1 done in {int(t1//60)}m {t1%60:.1f}s")

    print("\nPhase 2: Combining line identification")
    t2_start = time.time()
    spectuner.run_combining_line_id(config, SAVE_DIR)
    t2 = time.time() - t2_start
    total = time.time() - start
    print(f"Phase 2 done in {int(t2//60)}m {t2%60:.1f}s")
    print(f"Total: {int(total//60)}m {total%60:.1f}s")
    return total, t1, t2


def generate_summary(config, total, t1, t2):
    freq_ranges = [(float(o["spec"][0,0]), float(o["spec"][-1,0]))
                   for o in config["obs_info"]]
    total_freq = sum(f2 - f1 for f1, f2 in freq_ranges)

    # Collect single-phase results
    results = []
    import h5py
    with h5py.File(os.path.join(SAVE_DIR, "identify_results_single.h5"), "r") as f:
        for key in sorted(f.keys(), key=lambda k: int(k.split("_")[0])):
            grp = f[key]
            if "specie_data" in grp:
                raw = grp["specie_data"][()]
                data = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                sd = json.loads(data)
                for idx, species_dict in sd.items():
                    for name, info in species_dict.items():
                        results.append({
                            "id": idx, "name": name,
                            "score": info.get("score", 0),
                            "num_tp_i": info.get("num_tp_i", 0),
                            "num_fp": info.get("num_fp", 0),
                            "T_ex": info.get("T_ex", float("nan")),
                            "N_tot": info.get("N_tot", float("nan")),
                            "delta_v": info.get("delta_v", float("nan")),
                            "v_offset": info.get("v_offset", float("nan")),
                            "theta": info.get("theta", float("nan")),
                        })
    results.sort(key=lambda x: x["score"], reverse=True)

    # Collect combine-phase results
    combine_results = {}
    with h5py.File(os.path.join(SAVE_DIR, "identify_results_combine.h5"), "r") as f:
        if "combine" in f and "specie_data" in f["combine"]:
            raw = f["combine"]["specie_data"][()]
            sd = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            for idx, species_dict in sd.items():
                for name, info in species_dict.items():
                    combine_results[name] = info

    # Print summary
    passing = [r for r in results if r["score"] >= 2.7]
    print(f"\n=== Fit Summary ===")
    print(f"Species passing (score>=2.7): {len(passing)}/{len(results)}")
    print(f"\n{'name':<27s} {'score':>7s} {'lines':>5s} {'T_ex':>7s} {'N_tot':>12s} {'dv':>6s} {'voff':>6s} {'status':>6s}")
    print("-" * 85)
    for r in results:
        st = "PASS" if r["score"] >= 2.7 else "FAIL"
        print(f"  {r['name']:<25s} {r['score']:>7.2f} {r['num_tp_i']:>5d} "
              f"{r['T_ex']:>7.1f} {r['N_tot']:>12.3e} {r['delta_v']:>6.2f} {r['v_offset']:>6.2f} {st:>6s}")

    # Write summary.txt
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SAVE_DIR, "summary.txt")
    with open(path, "w") as f:
        f.write(f"# Spectuner CH3CHO Identification Summary\n# Generated: {ts}\n")
        f.write(f"Source: IRAS 16293-2422 B\nOutput: {SAVE_DIR}\n\n")
        f.write(f"## Spectral coverage\n")
        for i, (f1, f2) in enumerate(freq_ranges):
            f.write(f"  Window {i+1:2d}: {f1:.3f} - {f2:.3f} MHz\n")
        f.write(f"  Total: {total_freq/1e3:.3f} GHz\n\n")
        f.write(f"## Fit performance\n")
        f.write(f"  Phase 1: {int(t1//60)}m {t1%60:.1f}s\n")
        f.write(f"  Phase 2: {int(t2//60)}m {t2%60:.1f}s\n")
        f.write(f"  Total:   {int(total//60)}m {total%60:.1f}s\n\n")
        f.write(f"## Single-phase results\n\n")
        f.write(f"{'name':<27s} {'score':>7s} {'lines':>5s} {'fp':>3s} "
                f"{'theta':>7s} {'T_ex':>7s} {'N_tot':>12s} {'dv':>7s} {'voff':>7s} {'status':>6s}\n")
        f.write("-" * 105 + "\n")
        for r in results:
            st = "PASS" if r["score"] >= 2.7 else "FAIL"
            f.write(f"{r['name']:<27s} {r['score']:>7.2f} {r['num_tp_i']:>5d} {r['num_fp']:>3d} "
                    f"{r['theta']:>7.3f} {r['T_ex']:>7.1f} {r['N_tot']:>12.3e} "
                    f"{r['delta_v']:>7.2f} {r['v_offset']:>7.2f} {st:>6s}\n")
        f.write(f"\n## Combine-phase confirmed\n\n")
        for name, info in combine_results.items():
            f.write(f"  {name:<25s} score={info['score']:.2f} lines={info['num_tp']} "
                    f"T_ex={info['T_ex']:.1f}K N_tot={info['N_tot']:.2e}\n")
    print(f"\nSummary: {path}")
    return results, combine_results, freq_ranges


def plot_results(config, freq_ranges):
    import h5py
    obs_data = load_preprocess(config["obs_info"], clip=False)
    freq_data_obs = get_freq_data(obs_data)
    T_data_obs = get_T_data(obs_data)
    noise_avg = np.mean([item["noise"] for item in config["obs_info"]])

    # Load combine result
    ident_result = spectuner.load_previous_ident_result(
        os.path.join(SAVE_DIR, "identify_results_combine.h5"))
    df_mol = ident_result.derive_df_mol(max_order=3)
    param_mgr = ParameterManager.from_config(ident_result.specie_list, config)
    params_mol = param_mgr.derive_params(ident_result.x)

    name_to_params = {}
    idx = 0
    for mol_item in ident_result.specie_list:
        for name in mol_item["species"]:
            name_to_params[name] = params_mol[idx]
            idx += 1

    # SpectralPlot
    print("\nCreating SpectralPlot...")
    try:
        plot = SpectralPlot.from_config(
            config, freq_per_row=1000., width=20., height=3., color="k")
        plot.plot_ident_result(ident_result, show_lines=True,
                              color="r", color_blen="orange", color_fp="b", fontsize=9)
        plot.set_ylim(-10. * noise_avg, 80. * noise_avg)
        fig = plot.axes[0].figure
        fig.suptitle(f"IRAS 16293-2422 B — CH3CHO Family Overview (ALMA Band 6)", fontsize=12, y=0.998)
        fig.subplots_adjust(top=0.97)
        fig.savefig(os.path.join(SAVE_DIR, "spectral_overview.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: spectral_overview.png")
        plt.close(fig)
    except Exception as e:
        print(f"  SpectralPlot error: {e}")
        import traceback; traceback.print_exc()

    # PeakPlot for each species
    sl_db = create_spectral_line_db(CDMS_DB)
    for specie_idx, mol_item in enumerate(ident_result.specie_list):
        for name in mol_item["species"]:
            print(f"Creating PeakPlot for {name}...")
            try:
                key = mol_item["id"]
                ident_sub = ident_result.extract(key)
                lt = ident_sub.line_table
                mask = np.array([n is not None for n in lt.name])
                freqs = lt.freq[mask]
                if len(freqs) == 0:
                    print(f"  No peaks, skipping")
                    continue
                print(f"  {len(freqs)} peaks")

                p = name_to_params.get(name, params_mol[specie_idx])
                T_ex, N_tot, theta, delta_v_p, v_offset = p[1], p[2], p[0], p[3], p[4]
                score_val = df_mol.iloc[specie_idx]["score"]

                n_col = min(4, len(freqs))
                plot = PeakPlot(freqs, delta_v=20., n_col=n_col, plot_width=4, plot_height=3)
                plot.plot_spec(freq_data_obs, T_data_obs, step_plot=True, ylim_factor=1.5, color="k")
                T_pred = ident_sub.get_T_pred()
                if T_pred is not None:
                    plot.plot_spec(ident_sub.freq_data, T_pred, color="r")
                plot.vlines(freqs, linestyle="--", color="r")

                try:
                    props = ident_sub.query_sl_dict(sl_db, key, name)
                    if "E_up" in props:
                        n_a = min(len(freqs), len(props["E_up"]))
                        texts = np.array([f"{props['E_up'][j]:.1f}K" for j in range(n_a)])
                        plot.vtexts(freqs[:n_a], texts, h_txt_offset=0.02, v_txt_offset=0.92,
                                   fontsize=9, color="darkgreen")
                except Exception:
                    pass

                for i_a, ax in enumerate(plot.axes.flat):
                    if i_a >= plot.n_plot: continue
                    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
                    ax.xaxis.get_major_formatter().set_scientific(False)
                    ax.tick_params(axis="x", labelrotation=0, labelsize=7)
                    ax.locator_params(axis="x", nbins=4)

                n_row = (plot.n_plot + n_col - 1) // n_col
                for i_a, ax in enumerate(plot.axes.flat):
                    if i_a >= plot.n_plot: continue
                    if i_a % n_col == 0: ax.set_ylabel("Intensity [K]")
                    row = i_a // n_col
                    if (row == n_row-1) or (row == n_row-2 and i_a+n_col >= plot.n_plot):
                        ax.set_xlabel("Frequency [MHz]")
                plt.subplots_adjust(wspace=0.15, hspace=0.1, top=0.98)

                fig = plot.fig
                fig.suptitle(
                    f"{name}  T_ex={T_ex:.1f}K  N_tot={N_tot:.2e}cm⁻²  "
                    f"θ={theta:.2f}″  Δv={delta_v_p:.2f}km/s  v_off={v_offset:.2f}km/s  score={score_val:.2f}",
                    fontsize=10, y=0.995)
                safe = name.replace(";","_").replace("=","")
                fig.savefig(os.path.join(SAVE_DIR, f"peakplot_{safe}.png"), dpi=150, bbox_inches="tight")
                print(f"  Saved: peakplot_{safe}.png")
                plt.close(fig)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback; traceback.print_exc()


def main():
    print(f"Output: {SAVE_DIR}")
    print(f"Target: CH3CHO (acetaldehyde)")
    print(f"Source: IRAS 16293-2422 B")

    config, files = build_config()
    print(f"Loaded {len(files)} spectral windows, "
          f"coverage: {config['obs_info'][0]['spec'][0,0]/1e3:.3f} - "
          f"{config['obs_info'][-1]['spec'][-1,0]/1e3:.3f} GHz")

    total, t1, t2 = run_fit(config)
    generate_summary(config, total, t1, t2)
    plot_results(config, None)

    print("\nDone! Opening results...")
    os.system(f"open {os.path.join(SAVE_DIR, 'spectral_overview.png')}")
    for f in sorted(glob.glob(os.path.join(SAVE_DIR, "peakplot_*.png"))):
        os.system(f"open {f}")
    os.system(f"open {SAVE_DIR}")


if __name__ == "__main__":
    main()

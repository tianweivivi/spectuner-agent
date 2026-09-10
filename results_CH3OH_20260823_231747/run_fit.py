#!/usr/bin/env python3
"""Spectuner CH3OH identification for IRAS 16293-2422 B.

Fits only CH3OH to the 20 ALMA spectral windows (~232-241 GHz)
to determine whether methanol is present in the data.
"""

import os
import sys
import time
import glob
import numpy as np
import spectuner
from spectuner.sl_model import ParameterManager
from datetime import datetime


def main():
    # --- 1. Configuration ---
    DATA_DIR = "/Users/sdkkk/Desktop/spectra_skills/IRAS16293_spectra"
    CDMS_DB = "/Users/sdkkk/spectuner_data/cdms_sqlite__official-version__2024-01-01.db"
    SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

    # ALMA synthesized beam (typical for Band 6 ~1.3mm observations)
    # BMAJ and BMIN in degrees  (~0.3" x 0.2")
    BMAJ = 0.30 / 3600.0  # arcsec -> degrees
    BMIN = 0.20 / 3600.0

    T_BG = 0.0       # background temp (CMB added separately via need_cmb)
    NEED_CMB = True   # add 2.726 K CMB

    print(f"Output directory: {SAVE_DIR}")
    print(f"Data directory:   {DATA_DIR}")
    print(f"CDMS database:    {CDMS_DB}")

    # --- 2. Load all spectral windows, sorted by frequency ---
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.dat")),
        key=lambda f: float(np.loadtxt(f, max_rows=1)[0])
    )
    print(f"\nLoading {len(files)} spectral windows...")

    config = spectuner.load_default_config()
    config.set_fname_db(CDMS_DB)

    for i, f in enumerate(files):
        spw = f.split("spw")[1].split(".")[0]
        spec = np.loadtxt(f)
        freq_start, freq_end = spec[0, 0], spec[-1, 0]

        # Robust noise estimate via MAD
        intensity = spec[:, 1]
        mad = np.median(np.abs(intensity - np.median(intensity)))
        noise = float(mad * 1.4826)

        config.append_spectral_window(
            spec,
            beam_info=(BMAJ, BMIN),
            noise=noise,
            T_bg=T_BG,
            need_cmb=NEED_CMB,
        )
        print(f"  [{i+1:2d}/20] spw{spw:>3s}: {freq_start:.1f} - {freq_end:.1f} MHz, "
              f"noise = {noise:.3f} K")

    # --- 3. Set parameter bounds (appropriate for hot corino IRAS 16293-2422 B) ---
    config.set_param_info("theta",    is_log=False, bound=(0.0, 5.0))    # arcsec
    config.set_param_info("T_ex",     is_log=False, bound=(10.0, 500.0)) # K
    config.set_param_info("N_tot",    is_log=True,  bound=(12.0, 19.0))  # log cm^-2
    config.set_param_info("delta_v",  is_log=False, bound=(0.5, 10.0))   # km/s
    config.set_param_info("v_offset", is_log=False, bound=(-10.0, 10.0)) # km/s

    # --- 4. Configure species and optimizer ---
    config.set_ident_species(
        species=["CH3OH"],
        collect_iso=True,
        combine_iso=False,
        combine_state=False,
    )
    config.set_n_process(10)

    config.set_optimizer("pso", n_swarm=28, n_trial=1, n_draw=50)

    # Frequency coverage summary
    freq_ranges = []
    for obs in config["obs_info"]:
        s = obs["spec"]
        freq_ranges.append((float(s[0, 0]), float(s[-1, 0])))
    total_freq = sum(f2 - f1 for f1, f2 in freq_ranges)
    print(f"\nTotal frequency coverage: {freq_ranges[0][0]/1e3:.3f} - "
          f"{freq_ranges[-1][1]/1e3:.3f} GHz ({total_freq/1e3:.3f} GHz)")

    # --- 5. Run fitting with timing ---
    print("\n" + "=" * 60)
    print("Starting Phase 1: Individual line identification (CH3OH)")
    print("=" * 60)
    start_time = time.time()

    spectuner.run_individual_line_id(config, SAVE_DIR)

    t_phase1 = time.time() - start_time
    print(f"\nPhase 1 completed in {int(t_phase1//60)} min {t_phase1%60:.1f} s")

    print("\n" + "=" * 60)
    print("Starting Phase 2: Combining line identification")
    print("=" * 60)
    t2_start = time.time()

    spectuner.run_combining_line_id(config, SAVE_DIR)

    t_phase2 = time.time() - t2_start
    wall_time = time.time() - start_time
    print(f"\nPhase 2 completed in {int(t_phase2//60)} min {t_phase2%60:.1f} s")
    print(f"Total fit duration: {int(wall_time//60)} min {wall_time%60:.1f} s")

    # --- 6. Load results and generate summary ---
    print("\n" + "=" * 60)
    print("Generating summary...")
    print("=" * 60)

    try:
        ident_result = spectuner.load_previous_ident_result(SAVE_DIR)
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
                df_mol.at[i, "theta"] = p[0]
                df_mol.at[i, "T_ex"] = p[1]
                df_mol.at[i, "N_tot"] = p[2]
                df_mol.at[i, "delta_v"] = p[3]
                df_mol.at[i, "v_offset"] = p[4]

        # Print summary table
        print(f"\n=== Fit Summary ===")
        print(f"Output directory: {SAVE_DIR}")
        print(f"Total frequency coverage: {freq_ranges[0][0]/1e3:.3f} - "
              f"{freq_ranges[-1][1]/1e3:.3f} GHz ({total_freq/1e3:.3f} GHz)")
        print(f"Fit duration: {int(wall_time // 60)} min {wall_time % 60:.1f} s")
        print(f"\nSpecies identified: {stats['n_mol']}")
        print(f"Master groups: {stats['n_master']}")

        print(f"\n{'name':<25} {'lines':>5} {'score':>7} "
              f"{'theta':>7} {'T_ex':>7} {'N_tot':>12} {'delta_v':>7} {'v_off':>7}")
        print("-" * 90)
        for _, row in df_mol.iterrows():
            n_lines = row.get('num_tp_i', row.get('num_tp_c', 0))
            print(f"  {row['name']:<23} {n_lines:>5} {row['score']:>7.4f} "
                  f"{row.get('theta', float('nan')):>7.2f} "
                  f"{row.get('T_ex', float('nan')):>7.1f} "
                  f"{row.get('N_tot', float('nan')):>12.3e} "
                  f"{row.get('delta_v', float('nan')):>7.2f} "
                  f"{row.get('v_offset', float('nan')):>7.2f}")

        # --- 7. Write summary.txt ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(SAVE_DIR, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"# Spectuner Fit Summary\n")
            f.write(f"# Generated: {timestamp}\n")
            f.write(f"Output directory: {SAVE_DIR}\n\n")
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

        print(f"\nSummary written to: {summary_path}")

    except Exception as e:
        print(f"\nError generating summary: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone!")


if __name__ == '__main__':
    main()

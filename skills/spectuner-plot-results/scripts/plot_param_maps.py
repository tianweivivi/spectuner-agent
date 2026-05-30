"""
Helper script for converting pixel-level HDF fitting results to FITS files
and plotting 2D parameter distribution maps.

This script automates the conversion of HDF5 cube fitting results to FITS
format and generates publication-ready parameter maps for each fitted molecule.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path


PARAM_NAMES = ("theta", "T_ex", "N_tot", "delta_v", "v_offset")
PARAM_UNITS = ("arcsec", "K", "cm-2", "km/s", "km/s")
# Parameters that are typically stored in log scale in Spectuner
DEFAULT_LOG_PARAMS = {"N_tot", "theta"}


def convert_and_plot_params(
    cube_file,
    result_file,
    save_dir="output_plots",
    species=None,
    log_params=None,
    show_cb_label=True,
    cmap="viridis",
    fig_width=18,
    fig_height=12,
    dpi=150,
):
    """Convert HDF fitting results to FITS and plot parameter maps.

    Args:
        cube_file: Path to the HDF5 cube observation file.
        result_file: Path to the HDF5 pixel-level fitting result file.
        save_dir: Directory to save output plots and FITS files.
        species: List of species names to plot. If None, plot all species
            found in the result file.
        log_params: Set of parameter names to plot with logarithmic scale.
            Defaults to {"N_tot", "theta"}.
        show_cb_label: Whether to show colorbar labels with units.
        cmap: Matplotlib colormap name.
        fig_width: Figure width in inches.
        fig_height: Figure height in inches.
        dpi: DPI for saved figures.

    Returns:
        dict: Mapping from species name to saved figure path.
    """
    from spectuner import HDFCubeManager
    from astropy.io import fits

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cube_mgr = HDFCubeManager(cube_file)

    # Convert to FITS
    fits_dir = save_dir / "fits"
    fits_dir.mkdir(parents=True, exist_ok=True)
    cube_mgr.pred_data_to_fits(
        result_file,
        save_dir=str(fits_dir),
        add_v_LSR=True,
        overwrite=True
    )

    if log_params is None:
        log_params = DEFAULT_LOG_PARAMS

    # Determine which species to plot
    if species is None:
        import h5py
        with h5py.File(result_file, "r") as fp:
            species = [k for k in fp.keys() if k != "score"]

    fig_paths = {}

    for sp in species:
        sp_dir = fits_dir / sp
        if not sp_dir.exists():
            print(f"Warning: no FITS data found for species '{sp}', skipping.")
            continue

        fig, axes = plt.subplots(2, 3, figsize=(fig_width, fig_height))
        axes = axes.flatten()

        for idx, (param, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
            ax = axes[idx]
            fits_path = sp_dir / f"{param}.fits"

            if not fits_path.exists():
                ax.text(0.5, 0.5, f"{param}\nnot found",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"{param}")
                continue

            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            # Handle NaN/inf
            data = np.where(np.isfinite(data), data, np.nan)

            # Determine vmin/vmax robustly
            finite = data[np.isfinite(data)]
            if len(finite) == 0:
                ax.text(0.5, 0.5, f"{param}\nno data",
                        transform=ax.transAxes, ha="center", va="center")
                ax.set_title(f"{param}")
                continue

            # Log scale for certain parameters
            if param in log_params:
                pos = finite[finite > 0]
                if len(pos) > 0:
                    vmin, vmax = np.nanpercentile(pos, [1, 99])
                    norm = LogNorm(vmin=vmin, vmax=vmax)
                else:
                    vmin, vmax = np.nanpercentile(finite, [1, 99])
                    norm = None
            else:
                vmin, vmax = np.nanpercentile(finite, [1, 99])
                norm = None

            im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm,
                           vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{param}")
            ax.set_xlabel("Pixel")
            ax.set_ylabel("Pixel")

            # Colorbar
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if show_cb_label:
                label = f"{param} [{unit}]"
                cb.set_label(label)
            if param in log_params:
                cb.formatter.set_powerlimits((0, 0))
                cb.update_ticks()

        # Remove empty subplot
        for ax in axes[len(PARAM_NAMES):]:
            ax.remove()

        plt.suptitle(f"Parameter Maps: {sp}", fontsize=14)
        plt.tight_layout()

        fig_path = save_dir / f"{sp}_params_map.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        fig_paths[sp] = str(fig_path)
        print(f"Saved parameter map for {sp} to {fig_path}")

    return fig_paths


def plot_single_param_map(
    fits_file,
    param_name,
    unit="",
    log_scale=False,
    cmap="viridis",
    vmin=None,
    vmax=None,
    title=None,
    save_path=None,
    show=True,
):
    """Plot a single parameter map from a FITS file.

    Args:
        fits_file: Path to the FITS file.
        param_name: Name of the parameter (for title).
        unit: Unit string for colorbar label.
        log_scale: Whether to use logarithmic normalization.
        cmap: Matplotlib colormap name.
        vmin: Minimum value for colormap. If None, use 1st percentile.
        vmax: Maximum value for colormap. If None, use 99th percentile.
        title: Figure title. If None, use param_name.
        save_path: Path to save the figure.
        show: Whether to call plt.show().

    Returns:
        tuple: (fig, ax) matplotlib figure and axis.
    """
    from astropy.io import fits

    with fits.open(fits_file) as hdul:
        data = hdul[0].data

    data = np.where(np.isfinite(data), data, np.nan)
    finite = data[np.isfinite(data)]

    if vmin is None:
        vmin = np.nanpercentile(finite, 1)
    if vmax is None:
        vmax = np.nanpercentile(finite, 99)

    if log_scale:
        pos = finite[finite > 0]
        if len(pos) > 0 and vmin <= 0:
            vmin = np.nanpercentile(pos, 1)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, origin="lower", cmap=cmap, norm=norm,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Pixel")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if unit:
        cb.set_label(f"{param_name} [{unit}]")
    if log_scale:
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

    if title is None:
        title = param_name
    ax.set_title(title)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

#!/usr/bin/env python3
"""
Extract spectrum from astronomical FITS cubes or plain-text spectra,
compute noise and baseline, and prepare data for spectuner analysis.

Supports:
  - Single-pixel extraction from FITS cubes
  - Circular or rectangular aperture extraction (averaged spectrum)
  - Direct processing of existing .txt/.dat spectral files
  - Automatic noise estimation from line-free regions

Usage (FITS cube):
    python extract_spectrum.py <fits_files...>
        --ra <RA> --dec <DEC>
        --aperture-type {point,circle,box}
        --aperture-size <SIZE>
        --outdir <OUTDIR>

Usage (plain-text spectrum):
    python extract_spectrum.py <txt_files...>
        --from-txt
        --beam-bmaj <DEG> --beam-bmin <DEG>
        --outdir <OUTDIR>

Arguments:
    fits_files / txt_files   Input file(s), wildcards supported

Required:
    --outdir DIR             Output directory

FITS-specific required:
    --ra RA                  Target RA (deg or hh:mm:ss)
    --dec DEC                Target DEC (deg or dd:mm:ss)

FITS-specific optional:
    --aperture-type TYPE     Extraction aperture: point, circle, box (default: point)
    --aperture-size SIZE     Aperture radius / half-width in arcsec (default: 0)
                             For circle: radius of the circular aperture
                             For box: half-width of the square box
                             For point: ignored (single pixel)
    --noise-ref-ra RA        RA of noise reference region (default: auto-edge)
    --noise-ref-dec DEC      DEC of noise reference region
    --noise-auto-offset-asec OFFSET  Auto noise offset from target (default: 10)
    --continuum-fits FITS    Continuum FITS for T_bg (auto-detect if not given)
    --reverse-freq           Reverse frequency axis if needed (default: True)
    --cut-channels N         Cut first N channels (default: 0)
    --beam-bmaj DEG          Override BMAJ from header (deg)
    --beam-bmin DEG          Override BMIN from header (deg)

TXT-specific:
    --from-txt               Input files are plain-text spectra, not FITS
    --beam-bmaj DEG          Beam major axis in degrees (required for txt)
    --beam-bmin DEG          Beam minor axis in degrees (required for txt)
    --freq-scale SCALE       Frequency unit: mhz, ghz, hz (default: mhz)
    --intensity-scale SCALE  Intensity unit: kelvin, jybeam (default: kelvin)

General optional:
    --suffix SUFFIX          Suffix for output filenames
    --write-pickle           Also write a spectuner pickle config file
    --config-db PATH         CDMS database path for pickle config
"""

from astropy.io import fits
from spectral_cube import SpectralCube
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from astropy import units as u
import numpy as np
import os
import sys
import glob
import argparse
import re


def parse_angle(s):
    """Parse angle: return decimal degrees. Accepts decimal or sexagesimal."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        # Try to detect if RA-like (has more colons or hour-like)
        c = SkyCoord(s, unit=(u.hourangle, u.deg))
        return c.ra.deg


def parse_dec(s):
    """Parse declination, return decimal degrees."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        c = SkyCoord(ra=0*u.deg, dec=s, unit=(u.deg, u.deg))
        return c.dec.deg


def jybeam_to_kelvin(intensity_jy, freq_hz, bmaj_arcsec, bmin_arcsec):
    """Convert Jy/beam to brightness temperature (K)."""
    return intensity_jy * 1220000.0 / bmaj_arcsec / bmin_arcsec / (freq_hz / 1e9) ** 2


def get_beam_info(header, override_bmaj=None, override_bmin=None):
    """Extract beam from FITS header. Override if provided. Returns (bmaj_deg, bmin_deg, bmaj_arcsec, bmin_arcsec)."""
    if override_bmaj is not None and override_bmin is not None:
        bmaj_deg = float(override_bmaj)
        bmin_deg = float(override_bmin)
        return bmaj_deg, bmin_deg, bmaj_deg * 3600.0, bmin_deg * 3600.0

    bmaj = header.get('BMAJ', None)
    bmin = header.get('BMIN', None)

    if bmaj is None or bmin is None:
        # Try alternative header keys
        bmaj = header.get('BMAJ', header.get('CLEANBMJ', None))
        bmin = header.get('BMIN', header.get('CLEANBMN', None))

    if bmaj is None or bmin is None:
        return None  # Signal that beam info is missing

    bmaj_deg = float(bmaj)
    bmin_deg = float(bmin)
    return bmaj_deg, bmin_deg, bmaj_deg * 3600.0, bmin_deg * 3600.0


def create_circular_mask(ny, nx, cy, cx, radius_pix):
    """Create a boolean circular mask."""
    y, x = np.ogrid[:ny, :nx]
    mask = (x - cx)**2 + (y - cy)**2 <= radius_pix**2
    return mask


def create_box_mask(ny, nx, cy, cx, half_width_pix):
    """Create a boolean rectangular mask."""
    mask = np.zeros((ny, nx), dtype=bool)
    y_min = max(0, int(round(cy - half_width_pix)))
    y_max = min(ny, int(round(cy + half_width_pix)) + 1)
    x_min = max(0, int(round(cx - half_width_pix)))
    x_max = min(nx, int(round(cx + half_width_pix)) + 1)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def extract_spectrum_fits(cube_path, ra_deg, dec_deg,
                          aperture_type='point', aperture_size_asec=0,
                          noise_ref_ra=None, noise_ref_dec=None,
                          noise_auto_offset_asec=10,
                          continuum_path=None,
                          reverse_freq=True, cut_channels=0,
                          beam_bmaj=None, beam_bmin=None):
    """Extract spectrum from a single FITS cube.

    Returns a dict with extraction results.
    """
    result = {
        'cube_path': cube_path,
        'status': 'ok',
        'message': '',
        'frequency_mhz': None,
        'intensity_k': None,
        'noise': None,
        'T_bg': None,
        'bmaj_deg': None, 'bmin_deg': None,
        'bmaj_arcsec': None, 'bmin_arcsec': None,
        'freq_start_mhz': None, 'freq_end_mhz': None,
        'n_channels': None,
        'n_pixels_aperture': 1,
        'continuum_name': None,
        'aperture_type': aperture_type,
    }

    # Open FITS
    try:
        hdu = fits.open(cube_path)[0]
        header = hdu.header
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f'Cannot open FITS: {e}'
        return result

    # Beam info
    beam = get_beam_info(header, beam_bmaj, beam_bmin)
    if beam is None:
        result['status'] = 'error'
        result['message'] = (
            'Missing BMAJ/BMIN in FITS header. Please provide beam size '
            'via --beam-bmaj and --beam-bmin (in degrees).'
        )
        return result

    bmaj_deg, bmin_deg, bmaj_arcsec, bmin_arcsec = beam
    result['bmaj_deg'] = bmaj_deg
    result['bmin_deg'] = bmin_deg
    result['bmaj_arcsec'] = bmaj_arcsec
    result['bmin_arcsec'] = bmin_arcsec

    # WCS coordinate conversion
    wcs = WCS(header, naxis=2)
    x, y = wcs.all_world2pix(ra_deg, dec_deg, 1)
    x, y = float(x), float(y)

    # Read cube
    try:
        cube = SpectralCube.read(cube_path)
        cube.allow_huge_operations = True
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f'Cannot read cube: {e}'
        return result

    frequency = cube.spectral_axis.value  # Hz
    nchan = len(frequency)
    result['n_channels'] = nchan
    ny, nx = header['NAXIS2'], header['NAXIS1']

    # --- Extract target spectrum via aperture ---
    if aperture_type == 'point' or aperture_size_asec <= 0:
        # Single pixel
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < nx and 0 <= yi < ny):
            result['status'] = 'error'
            result['message'] = f'Target pixel ({xi},{yi}) out of bounds ({nx},{ny})'
            return result
        try:
            intensity = cube[:, yi, xi]
            result['n_pixels_aperture'] = 1
        except Exception as e:
            result['status'] = 'error'
            result['message'] = f'Pixel extraction failed: {e}'
            return result
    else:
        # Circular or rectangular aperture
        pixel_scale = np.abs(header.get('CDELT1', header.get('CD1_1', 1.0)))  # deg/pix
        pixel_scale_arcsec = pixel_scale * 3600.0

        if aperture_type == 'circle':
            radius_pix = aperture_size_asec / pixel_scale_arcsec
            mask = create_circular_mask(ny, nx, y, x, radius_pix)
        elif aperture_type == 'box':
            half_width_pix = aperture_size_asec / pixel_scale_arcsec
            mask = create_box_mask(ny, nx, y, x, half_width_pix)
        else:
            result['status'] = 'error'
            result['message'] = f'Unknown aperture type: {aperture_type}'
            return result

        if not np.any(mask):
            result['status'] = 'error'
            result['message'] = 'Aperture contains no valid pixels'
            return result

        masked_cube = cube.with_mask(mask)
        intensity = masked_cube.mean(axis=(1, 2))
        result['n_pixels_aperture'] = int(np.sum(mask))

    # --- Noise: extract from line-free region ---
    # Strategy: use explicit noise ref coordinate, or auto-offset from target
    if noise_ref_ra is not None and noise_ref_dec is not None:
        x_n, y_n = wcs.all_world2pix(noise_ref_ra, noise_ref_dec, 1)
        x_n, y_n = int(round(float(x_n))), int(round(float(y_n)))
        if not (0 <= x_n < nx and 0 <= y_n < ny):
            result['status'] = 'warning'
            result['message'] += f' Noise ref pixel ({x_n},{y_n}) out of bounds, using auto. '
            x_n, y_n = None, None
    else:
        x_n, y_n = None, None

    if x_n is None:
        # Auto: offset from target toward the image edge
        pixel_scale = np.abs(header.get('CDELT1', header.get('CD1_1', 1.0)))
        offset_pix = int(round(noise_auto_offset_asec / 3600.0 / pixel_scale))
        # Try several positions: right, down-right, down, down-left, left, up-left, up, up-right
        candidates = [
            (x + offset_pix, y),
            (x + offset_pix, y + offset_pix),
            (x, y + offset_pix),
            (x - offset_pix, y + offset_pix),
            (x - offset_pix, y),
            (x - offset_pix, y - offset_pix),
            (x, y - offset_pix),
            (x + offset_pix, y - offset_pix),
        ]
        for cx, cy in candidates:
            cxi, cyi = int(round(cx)), int(round(cy))
            if 0 <= cxi < nx and 0 <= cyi < ny:
                x_n, y_n = cxi, cyi
                break

    if x_n is None or y_n is None:
        result['status'] = 'error'
        result['message'] += 'Could not find a valid noise reference pixel. '
        return result

    try:
        noise_line = cube[:, y_n, x_n]
    except Exception as e:
        result['status'] = 'error'
        result['message'] += f'Noise extraction failed: {e} '
        return result

    result['noise_pixel'] = (x_n, y_n)
    result['target_pixel'] = (int(round(x)), int(round(y)))

    # --- Unit conversion to Kelvin ---
    freq_hz = frequency
    intensity_k = jybeam_to_kelvin(intensity.value, freq_hz, bmaj_arcsec, bmin_arcsec)
    noise_k = jybeam_to_kelvin(noise_line.value, freq_hz, bmaj_arcsec, bmin_arcsec)

    # Compute noise via sigma-clipped stats on noise spectrum (line-free region)
    clipped = sigma_clipped_stats(noise_k, sigma=3, maxiters=5)
    noise_rms = float(np.std(clipped))

    # --- Baseline / continuum (T_bg) ---
    T_bg = None
    if continuum_path is not None and os.path.exists(continuum_path):
        try:
            cont_hdu = fits.open(continuum_path)[0]
            cont_data = cont_hdu.data
            cont_header = cont_hdu.header
            cont_wcs = WCS(cont_header, naxis=2)
            cx, cy = cont_wcs.all_world2pix(ra_deg, dec_deg, 1)
            cx, cy = int(round(float(cx))), int(round(float(cy)))
            if cont_data.ndim == 4:
                cont_val = cont_data[0, 0, cy, cx]
            elif cont_data.ndim == 2:
                cont_val = cont_data[cy, cx]
            else:
                cont_val = cont_data.flatten()[0]
            cont_freq = cont_header.get('RESTFRQ', freq_hz[0] if len(freq_hz) else 2.3e11)
            T_bg = jybeam_to_kelvin(cont_val, cont_freq, bmaj_arcsec, bmin_arcsec)
            result['continuum_name'] = os.path.basename(continuum_path)
        except Exception:
            T_bg = None

    # Fallback: median of sigma-clipped spectrum
    if T_bg is None:
        clipped_spec = sigma_clipped_stats(intensity_k, sigma=3, maxiters=5)
        T_bg = float(np.median(clipped_spec))

    # --- Post-processing ---
    freq_mhz = freq_hz / 1e6
    if reverse_freq and len(freq_mhz) > 1 and freq_mhz[0] > freq_mhz[-1]:
        freq_mhz = freq_mhz[::-1]
        intensity_k = intensity_k[::-1]

    if cut_channels > 0:
        freq_mhz = freq_mhz[cut_channels:]
        intensity_k = intensity_k[cut_channels:]

    result['frequency_mhz'] = freq_mhz
    result['intensity_k'] = intensity_k
    result['noise'] = noise_rms
    result['T_bg'] = float(T_bg)
    result['freq_start_mhz'] = float(freq_mhz[0]) if len(freq_mhz) else None
    result['freq_end_mhz'] = float(freq_mhz[-1]) if len(freq_mhz) else None

    return result


def process_txt_file(txt_path, beam_bmaj=None, beam_bmin=None,
                     freq_scale='mhz', intensity_scale='kelvin'):
    """Process a plain-text spectral file.

    Returns a dict similar to extract_spectrum_fits.
    """
    result = {
        'cube_path': txt_path,
        'status': 'ok',
        'message': '',
        'frequency_mhz': None,
        'intensity_k': None,
        'noise': None,
        'T_bg': None,
        'bmaj_deg': None, 'bmin_deg': None,
        'bmaj_arcsec': None, 'bmin_arcsec': None,
        'freq_start_mhz': None, 'freq_end_mhz': None,
        'n_channels': None,
        'n_pixels_aperture': None,
        'continuum_name': None,
        'aperture_type': 'txt_file',
    }

    # Validate beam info
    if beam_bmaj is None or beam_bmin is None:
        result['status'] = 'error'
        result['message'] = (
            'TXT files lack beam information. Please provide '
            '--beam-bmaj and --beam-bmin (in degrees).'
        )
        return result

    bmaj_deg = float(beam_bmaj)
    bmin_deg = float(beam_bmin)
    result['bmaj_deg'] = bmaj_deg
    result['bmin_deg'] = bmin_deg
    result['bmaj_arcsec'] = bmaj_deg * 3600.0
    result['bmin_arcsec'] = bmin_deg * 3600.0

    # Read data
    try:
        data = np.loadtxt(txt_path)
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f'Cannot read TXT file: {e}'
        return result

    if data.ndim != 2 or data.shape[1] < 2:
        result['status'] = 'error'
        result['message'] = 'TXT file must have at least 2 columns (freq, intensity)'
        return result

    freq = data[:, 0]
    intensity = data[:, 1]

    # Convert frequency to MHz
    freq_scale = freq_scale.lower()
    if freq_scale == 'ghz':
        freq_mhz = freq * 1000.0
    elif freq_scale == 'hz':
        freq_mhz = freq / 1e6
    elif freq_scale == 'mhz':
        freq_mhz = freq
    else:
        result['status'] = 'error'
        result['message'] = f'Unknown freq_scale: {freq_scale}'
        return result

    # Convert intensity to Kelvin
    intensity_scale = intensity_scale.lower()
    if intensity_scale == 'jybeam':
        # Need frequency for conversion; use median freq
        freq_hz = np.median(freq_mhz) * 1e6
        intensity_k = jybeam_to_kelvin(
            intensity, freq_hz, result['bmaj_arcsec'], result['bmin_arcsec']
        )
    elif intensity_scale == 'kelvin' or intensity_scale == 'k':
        intensity_k = intensity
    else:
        result['status'] = 'error'
        result['message'] = f'Unknown intensity_scale: {intensity_scale}'
        return result

    # Sigma-clipped stats for noise and baseline
    clipped = sigma_clipped_stats(intensity_k, sigma=3, maxiters=5)
    noise_rms = float(np.std(clipped))
    T_bg = float(np.median(clipped))

    result['frequency_mhz'] = freq_mhz
    result['intensity_k'] = intensity_k
    result['noise'] = noise_rms
    result['T_bg'] = T_bg
    result['n_channels'] = len(freq_mhz)
    result['freq_start_mhz'] = float(freq_mhz[0])
    result['freq_end_mhz'] = float(freq_mhz[-1])

    return result


def find_continuum_fits(cube_path):
    """Auto-detect continuum FITS file matching a cube."""
    dir_path = os.path.dirname(cube_path)
    base = os.path.basename(cube_path)
    patterns = [
        base.replace('.cube.I.pbcor.', '.mfs.I.pbcor.').replace('.cube.', '.mfs.'),
        base.replace('.cube.', '.cont.').replace('.image.', '.cont.'),
        '*mfs*I.pbcor.fits',
        '*continuum*.fits',
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(dir_path, pat))
        if matches:
            return matches[0]

    continuum_dir = os.path.join(dir_path, 'continuum')
    if os.path.isdir(continuum_dir):
        spw_match = re.search(r'spw(\d+)', base)
        if spw_match:
            spw_num = spw_match.group(1)
            for f in os.listdir(continuum_dir):
                if f.endswith('.fits') and f'spw{spw_num}' in f:
                    return os.path.join(continuum_dir, f)
        # No SPW match: try any continuum file
        fits_files = [f for f in os.listdir(continuum_dir) if f.endswith('.fits')]
        if fits_files:
            return os.path.join(continuum_dir, fits_files[0])
    return None


def write_spectrum_dat(result, outdir, suffix=''):
    """Write extracted spectrum to .dat file. Returns output path."""
    base_name = os.path.basename(result['cube_path'])
    if base_name.endswith('.fits'):
        base_name = base_name[:-5]
    elif base_name.endswith('.dat'):
        base_name = base_name[:-4]
    elif base_name.endswith('.txt'):
        base_name = base_name[:-4]

    if suffix:
        out_name = f'{base_name}_{suffix}.dat'
    else:
        out_name = f'{base_name}.dat'
    out_path = os.path.join(outdir, out_name)

    freq = result['frequency_mhz']
    intens = result['intensity_k']
    np.savetxt(out_path, np.c_[freq, intens],
               fmt='%.6f  %.6f',
               header='Frequency (MHz)  Tb (K)')
    return out_path


def write_summary(results, outdir, out_name='extract_summary.txt'):
    """Write summary text file with per-file info."""
    out_path = os.path.join(outdir, out_name)
    lines = []
    lines.append('# Spectral Extraction Summary')
    lines.append('# Columns: name freq_start(MHz) freq_end(MHz) T_bg(K) noise(K) bmaj(deg) bmin(deg) n_pixels aperture status')
    lines.append('')
    for r in results:
        name = os.path.basename(r['cube_path'])
        for ext in ['.fits', '.dat', '.txt']:
            if name.endswith(ext):
                name = name[:-len(ext)]
                break
        freq1 = f"{r['freq_start_mhz']:.6f}" if r['freq_start_mhz'] is not None else 'NA'
        freq2 = f"{r['freq_end_mhz']:.6f}" if r['freq_end_mhz'] is not None else 'NA'
        T_bg = f"{r['T_bg']:.6f}" if r['T_bg'] is not None else 'NA'
        noise = f"{r['noise']:.6f}" if r['noise'] is not None else 'NA'
        bmaj = f"{r['bmaj_deg']:.10f}" if r['bmaj_deg'] is not None else 'NA'
        bmin = f"{r['bmin_deg']:.10f}" if r['bmin_deg'] is not None else 'NA'
        n_pix = r.get('n_pixels_aperture', 'NA')
        aperture = r.get('aperture_type', 'NA')
        status = r['status']
        msg = f" # {r['message']}" if r['message'] else ''
        lines.append(f'{name} {freq1} {freq2} {T_bg} {noise} {bmaj} {bmin} {n_pix} {aperture} {status}{msg}')
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return out_path


def write_spectuner_pickle(results, outdir, db_path='cdms.db'):
    """Write a spectuner pickle config file."""
    try:
        import spectuner
    except ImportError:
        print('WARNING: spectuner not installed, skipping pickle output', file=sys.stderr)
        return None

    config = spectuner.load_default_config()
    config.set_fname_db(db_path)

    for r in results:
        if r['status'] != 'ok':
            continue
        spec = np.c_[r['frequency_mhz'], r['intensity_k']]
        beam = (r['bmaj_deg'], r['bmin_deg'])
        config.append_spectral_window(
            spec, beam_info=beam,
            noise=r['noise'], T_bg=r['T_bg'], need_cmb=True
        )

    out_path = os.path.join(outdir, 'spectuner_config.pkl')
    spectuner.save_config(config, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Extract spectrum from FITS cubes or text files')
    parser.add_argument('input_paths', nargs='+', help='Input file(s) (FITS or txt)')
    parser.add_argument('--outdir', required=True, help='Output directory')

    # Source coordinates (FITS only)
    parser.add_argument('--ra', default=None, help='Target RA (deg or hh:mm:ss)')
    parser.add_argument('--dec', default=None, help='Target DEC (deg or dd:mm:ss)')

    # Aperture
    parser.add_argument('--aperture-type', choices=['point', 'circle', 'box'],
                        default='point', help='Extraction aperture type')
    parser.add_argument('--aperture-size', type=float, default=0,
                        help='Aperture radius/box half-width in arcsec (0=single pixel)')

    # Noise reference
    parser.add_argument('--noise-ref-ra', default=None, help='Noise reference RA')
    parser.add_argument('--noise-ref-dec', default=None, help='Noise reference DEC')
    parser.add_argument('--noise-auto-offset-asec', type=float, default=10.0,
                        help='Auto noise offset from target (arcsec)')

    # Continuum
    parser.add_argument('--continuum-fits', default=None,
                        help='Continuum FITS for T_bg (auto-detect if not given)')

    # Post-processing
    parser.add_argument('--reverse-freq', type=lambda x: x.lower() == 'true',
                        default=True, help='Reverse frequency axis if descending')
    parser.add_argument('--cut-channels', type=int, default=0,
                        help='Cut first N channels')
    parser.add_argument('--suffix', default='', help='Output filename suffix')

    # TXT file options
    parser.add_argument('--from-txt', action='store_true',
                        help='Input files are plain-text spectra')
    parser.add_argument('--beam-bmaj', type=float, default=None,
                        help='Beam major axis in degrees (required for txt or missing FITS header)')
    parser.add_argument('--beam-bmin', type=float, default=None,
                        help='Beam minor axis in degrees')
    parser.add_argument('--freq-scale', default='mhz',
                        choices=['mhz', 'ghz', 'hz'],
                        help='Frequency unit in txt file')
    parser.add_argument('--intensity-scale', default='kelvin',
                        choices=['kelvin', 'jybeam'],
                        help='Intensity unit in txt file')

    # Output
    parser.add_argument('--write-pickle', action='store_true',
                        help='Write spectuner pickle config')
    parser.add_argument('--config-db', default='cdms.db',
                        help='CDMS DB path for pickle config')

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Expand glob patterns
    input_files = []
    for fp in args.input_paths:
        if '*' in fp or '?' in fp:
            input_files.extend(sorted(glob.glob(fp)))
        else:
            input_files.append(fp)

    # Determine mode
    from_txt = args.from_txt
    if not from_txt:
        # Auto-detect: if all files are .txt/.dat, assume txt mode
        all_txt = all(
            f.endswith('.txt') or f.endswith('.dat') or f.endswith('.spec')
            for f in input_files
        )
        if all_txt:
            from_txt = True
            print('Auto-detected txt input mode (all files are .txt/.dat)')

    # Validate required args per mode
    if not from_txt:
        if args.ra is None or args.dec is None:
            parser.error('--ra and --dec are required for FITS cube input')
        ra_deg = parse_angle(args.ra)
        dec_deg = parse_dec(args.dec)
        noise_ref_ra = parse_angle(args.noise_ref_ra) if args.noise_ref_ra else None
        noise_ref_dec = parse_dec(args.noise_ref_dec) if args.noise_ref_dec else None

    results = []
    for in_path in input_files:
        if not os.path.exists(in_path):
            print(f'WARNING: file not found: {in_path}', file=sys.stderr)
            continue

        print(f'Processing: {in_path}')

        if from_txt:
            result = process_txt_file(
                in_path,
                beam_bmaj=args.beam_bmaj,
                beam_bmin=args.beam_bmin,
                freq_scale=args.freq_scale,
                intensity_scale=args.intensity_scale,
            )
        else:
            # Auto-detect continuum
            cont_path = args.continuum_fits
            if cont_path is None:
                cont_path = find_continuum_fits(in_path)
                if cont_path:
                    print(f'  Auto-detected continuum: {cont_path}')

            result = extract_spectrum_fits(
                in_path, ra_deg, dec_deg,
                aperture_type=args.aperture_type,
                aperture_size_asec=args.aperture_size,
                noise_ref_ra=noise_ref_ra,
                noise_ref_dec=noise_ref_dec,
                noise_auto_offset_asec=args.noise_auto_offset_asec,
                continuum_path=cont_path,
                reverse_freq=args.reverse_freq,
                cut_channels=args.cut_channels,
                beam_bmaj=args.beam_bmaj,
                beam_bmin=args.beam_bmin,
            )

        results.append(result)

        if result['status'] == 'ok':
            out_path = write_spectrum_dat(result, args.outdir, args.suffix)
            print(f'  -> Spectrum: {out_path}')
            print(f'  Freq range: {result["freq_start_mhz"]:.3f} - {result["freq_end_mhz"]:.3f} MHz')
            print(f'  Noise: {result["noise"]:.4f} K, T_bg: {result["T_bg"]:.4f} K')
            print(f'  Beam: {result["bmaj_arcsec"]:.3f}" x {result["bmin_arcsec"]:.3f}"')
            if not from_txt:
                print(f'  Aperture: {result["aperture_type"]} ({result["n_pixels_aperture"]} pixels)')
                print(f'  Target pixel: {result["target_pixel"]}, Noise pixel: {result["noise_pixel"]}')
        else:
            print(f'  ERROR: {result["message"]}')

    # Write summary
    summary_path = write_summary(results, args.outdir)
    print(f'\nSummary written to: {summary_path}')

    # Write pickle config
    if args.write_pickle:
        pickle_path = write_spectuner_pickle(results, args.outdir, args.config_db)
        if pickle_path:
            print(f'Pickle config written to: {pickle_path}')

    # Report any failures
    failures = [r for r in results if r['status'] != 'ok']
    if failures:
        print(f'\nWARNING: {len(failures)} / {len(results)} files failed:')
        for r in failures:
            print(f'  - {os.path.basename(r["cube_path"])}: {r["message"]}')


if __name__ == '__main__':
    main()

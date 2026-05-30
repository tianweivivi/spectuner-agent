---
name: spectuner-extract-spectrum
description: Extract spectra from astronomical FITS cubes or plain-text files for spectuner line identification and fitting. Use this skill whenever the user wants to extract a spectrum from a FITS spectral cube at a given sky coordinate (single pixel, circular aperture, or rectangular region average), convert Jy/beam to Kelvin, estimate noise (RMS) from a line-free region and baseline/continuum level, read spatial resolution (beam size), process existing .txt/.dat spectral files, and save the results as text files ready for spectuner analysis. Trigger on phrases like "extract spectrum from FITS", "read FITS cube", "get spectrum from ALMA data", "prepare data for spectuner", "FITS to txt/dat", "spectral extraction", "calculate noise from cube", "beam size from FITS", "average spectrum over region", "circular aperture extraction", "process txt spectrum", or any workflow involving FITS cubes or text spectra → spectuner input files.
---

# Spectuner Spectrum Extractor

Extract spectra from astronomical FITS spectral cubes or plain-text spectral files, and prepare them as input for the [spectuner](spectuner-run) automated line identification pipeline.

## What this skill does

1. **Read FITS spectral cubes or text spectra** — one or many
2. **Extract spectrum at a target coordinate** — single pixel, circular aperture, or rectangular box average
3. **Unit conversion** — Jy/beam → brightness temperature (K) using beam size and frequency
4. **Estimate noise (RMS)** — sigma-clipped statistics on a spectrum from a **line-free off-source region**
5. **Estimate baseline / continuum level** — reads companion continuum FITS (auto-detected) or estimates from spectrum median
6. **Read spatial resolution** — extracts BMAJ / BMIN from FITS header; **prompts user if missing**
7. **Process existing text spectra** — reads `.txt`/`.dat` files directly, computes noise via sigma-clip
8. **Write outputs** — plain-text `.dat` spectra + summary info file, optionally a spectuner pickle config

## The extraction script

A bundled Python script at `scripts/extract_spectrum.py` handles the core pipeline. Use it directly for batch operations, or follow the step-by-step workflow below for interactive work.

### Quick usage (FITS cubes)

**Single pixel:**
```bash
python scripts/extract_spectrum.py \
  cubes/54_fits_edit/*.fits \
  --ra 248.0940833 --dec -24.4757778 \
  --outdir data/
```

**Circular aperture (0.5 arcsec radius):**
```bash
python scripts/extract_spectrum.py \
  cubes/54_fits_edit/*.fits \
  --ra 248.0940833 --dec -24.4757778 \
  --aperture-type circle --aperture-size 0.5 \
  --outdir data/
```

**Rectangular box (1.0 arcsec half-width):**
```bash
python scripts/extract_spectrum.py \
  cubes/54_fits_edit/*.fits \
  --ra 248.0940833 --dec -24.4757778 \
  --aperture-type box --aperture-size 1.0 \
  --outdir data/
```

### Quick usage (plain-text spectra)

```bash
python scripts/extract_spectrum.py \
  data/*.dat \
  --from-txt \
  --beam-bmaj 0.000173 --beam-bmin 0.000123 \
  --outdir processed/
```

### Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `input_paths` | Input file(s), wildcards OK | **required** |
| `--outdir` | Output directory | **required** |
| `--ra` | Target RA (deg or `hh:mm:ss`) | FITS only, **required** |
| `--dec` | Target DEC (deg or `dd:mm:ss`) | FITS only, **required** |
| `--aperture-type` | `point`, `circle`, or `box` | `point` |
| `--aperture-size` | Radius / half-width in **arcsec** | `0` |
| `--noise-ref-ra` / `--noise-ref-dec` | Explicit line-free noise region | auto-offset from target |
| `--noise-auto-offset-asec` | Auto noise offset from target | `10.0` |
| `--continuum-fits` | Continuum FITS for T_bg | auto-detect |
| `--beam-bmaj` / `--beam-bmin` | Override beam (degrees) | from FITS header |
| `--reverse-freq` | Reverse if freq runs high→low | `true` |
| `--cut-channels` | Cut first N channels | `0` |
| `--from-txt` | Input is plain-text spectra | auto-detect |
| `--freq-scale` | `mhz` / `ghz` / `hz` | `mhz` |
| `--intensity-scale` | `kelvin` / `jybeam` | `kelvin` |
| `--write-pickle` | Write spectuner pickle config | false |
| `--config-db` | CDMS DB path for pickle | `cdms.db` |

### Auto-detected continuum FITS patterns

The script tries these patterns to find a matching continuum image:

1. Replace `.cube.I.pbcor.` → `.mfs.I.pbcor.` in cube filename
2. Replace `.cube.` → `.mfs.` or `.cont.`
3. Search directory for `*mfs*I.pbcor.fits`
4. Search `continuum/` subdirectory for matching SPW number
5. Fallback: use any `.fits` in the `continuum/` subdirectory

## Step-by-step interactive workflow

### Step 1: Gather inputs

Ask the user for (or infer from context):

- **Input files** — FITS cubes or `.txt`/`.dat` spectra
- **Target coordinate** (FITS only) — RA and DEC
- **Extraction aperture** — point (default), circle radius, or box half-width
- **Noise reference** — explicit line-free coordinate, or accept auto-offset (default 10 arcsec from target)
- **Output directory** — where to save `.dat` files
- **Beam size** — if FITS header lacks BMAJ/BMIN, **ask the user to provide it**

### Step 2: Determine extraction mode

- If all input files end with `.txt`/`.dat`/`.spec` → **text mode**
- Otherwise → **FITS mode**

In text mode, the user **must** provide beam size (`--beam-bmaj`, `--beam-bmin` in degrees) since text files have no header.

### Step 3A: FITS mode — Inspect header and check beam info

Read the FITS header to confirm:

- `NAXIS3` (or `NAXIS4`) — number of spectral channels
- `CRVAL3`, `CDELT3`, `CRPIX3` — frequency axis definition
- `BMAJ`, `BMIN` — synthesized beam (degrees)
- `BUNIT` — should be `Jy/beam`
- Data shape — may be `(Stokes, Freq, Y, X)` = 4D

**If BMAJ/BMIN are missing from the header**, stop and ask the user:

> The FITS header is missing beam size information (BMAJ/BMIN). Please provide the synthesized beam size so I can convert Jy/beam to Kelvin. What are the BMAJ and BMIN values (in arcseconds or degrees)?

Once the user provides the beam size, pass it via `--beam-bmaj` and `--beam-bmin` (in **degrees**).

### Step 3B: Coordinate conversion (FITS only)

```python
from astropy.wcs import WCS
wcs = WCS(header, naxis=2)
x, y = wcs.all_world2pix(ra_deg, dec_deg, 1)
```

Round to nearest integer pixel for the center. Verify it is inside image bounds.

### Step 4: Extract spectrum via aperture

**Point (single pixel):**
```python
intensity = cube[:, int(y), int(x)]  # Jy/beam
```

**Circular aperture:**
```python
pixel_scale_deg = abs(header['CDELT1'])
pixel_scale_arcsec = pixel_scale_deg * 3600.0
radius_pix = aperture_size_arcsec / pixel_scale_arcsec

y_grid, x_grid = np.ogrid[:ny, :nx]
mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius_pix**2
masked_cube = cube.with_mask(mask)
intensity = masked_cube.mean(axis=(1, 2))
```

**Rectangular box:**
```python
half_width_pix = aperture_size_arcsec / pixel_scale_arcsec
mask = np.zeros((ny, nx), dtype=bool)
mask[int(y-hw):int(y+hw)+1, int(x-hw):int(x+hw)+1] = True
masked_cube = cube.with_mask(mask)
intensity = masked_cube.mean(axis=(1, 2))
```

### Step 5: Convert units

```python
# Jy/beam -> K
flux_kelvin = intensity * 1220000. / bmaj_arcsec / bmin_arcsec / (frequency/1e9)**2
```

Where `bmaj_arcsec = BMAJ * 3600` (BMAJ from FITS header or user-provided).

### Step 6: Estimate noise from line-free region

The noise must come from a **line-free off-source region** — a spectrum that contains only continuum and thermal noise, not emission lines.

**With explicit noise reference coordinate:**
```python
x_n, y_n = wcs.all_world2pix(noise_ra_deg, noise_dec_deg, 1)
noise_line = cube[:, int(y_n), int(x_n)]
```

**Auto mode (default):** offset from target toward the image edge by ~10 arcsec. Try multiple directions (right, down-right, down, etc.) and pick the first in-bounds pixel:

```python
offset_pix = int(round(noise_auto_offset_asec / 3600.0 / pixel_scale_deg))
candidates = [
    (x + offset_pix, y), (x + offset_pix, y + offset_pix),
    (x, y + offset_pix), (x - offset_pix, y + offset_pix),
    (x - offset_pix, y), (x - offset_pix, y - offset_pix),
    (x, y - offset_pix), (x + offset_pix, y - offset_pix),
]
```

Then compute sigma-clipped RMS:

```python
from astropy.stats import sigma_clipped_stats
noise_k = noise_line * 1220000. / bmaj_arcsec / bmin_arcsec / (frequency/1e9)**2
clipped = sigma_clipped_stats(noise_k.value, sigma=3, maxiters=5)
rms = np.std(clipped)
```

### Step 7: Estimate baseline / continuum (T_bg)

**Preferred method:** read companion continuum FITS:

```python
cont_hdu = fits.open(continuum_path)[0]
cont_data = cont_hdu.data  # shape may be (1,1,Y,X) or (Y,X)
cont_val = cont_data[0, 0, int(y), int(x)]  # Jy/beam
cont_freq = cont_hdu.header['RESTFRQ']
T_bg = cont_val * 1220000. / bmaj_arcsec / bmin_arcsec / (cont_freq/1e9)**2
```

**Fallback method:** median of the sigma-clipped target spectrum:

```python
clipped_spec = sigma_clipped_stats(flux_kelvin.value, sigma=3, maxiters=5)
T_bg = np.median(clipped_spec)
```

### Step 8A: Process plain-text spectra (txt mode)

If the input is already a `.txt`/`.dat` spectrum (no FITS cube):

1. Read the two-column file (frequency, intensity)
2. Convert frequency to MHz if needed (based on `--freq-scale`)
3. Convert intensity to Kelvin if in Jy/beam (based on `--intensity-scale` and provided beam size)
4. Compute noise and baseline via sigma-clipped stats on the entire spectrum:

```python
clipped = sigma_clipped_stats(intensity_k, sigma=3, maxiters=5)
noise_rms = np.std(clipped)
T_bg = np.median(clipped)
```

**In txt mode, beam size is mandatory** — ask the user if not provided:

> These are plain-text spectral files without beam information. To prepare them for spectuner, I need the synthesized beam size (BMAJ and BMIN). What are the values (in arcseconds)?

### Step 9: Post-process and save

**Reverse frequency axis** (if freq runs high→low):

```python
if frequency_mhz[0] > frequency_mhz[-1]:
    frequency_mhz = frequency_mhz[::-1]
    intensity_k = intensity_k[::-1]
```

**Cut overlapping channels** (ALMA cubes often have 256 overlapping channels):

```python
frequency_mhz = frequency_mhz[256:]
intensity_k = intensity_k[256:]
```

**Write `.dat` file** (two columns, frequency in MHz, intensity in K):

```python
np.savetxt('output.dat', np.c_[frequency_mhz, intensity_k], fmt='%.6f  %.6f')
```

### Step 10: Write summary file

Write a text summary with one line per file:

```
# Columns: name freq_start(MHz) freq_end(MHz) T_bg(K) noise(K) bmaj(deg) bmin(deg) n_pixels aperture status

54_spw25.cube.I.pbcor 232046.413 232483.580 19.383976 0.376227 0.0001728982 0.0001226759 1 point ok
54_spw27.cube.I.pbcor 232483.941 232921.109 24.592639 0.277434 0.0001727833 0.0001231900 12 circle ok
```

### Step 11: (Optional) Write spectuner pickle config

The latest spectuner stores config as Python pickle files, not YAML. Generate one with:

```python
import spectuner

config = spectuner.load_default_config()
config.set_fname_db("/path/to/cdms.db")

for each spectrum:
    spec = np.c_[frequency_mhz, intensity_k]
    config.append_spectral_window(
        spec,
        beam_info=(bmaj_deg, bmin_deg),
        noise=noise_rms,
        T_bg=T_bg,
        need_cmb=True
    )

spectuner.save_config(config, "spectuner_config.pkl")
```

Then the user can load it directly:

```python
config = spectuner.load_config("spectuner_config.pkl")
```

## Important notes

- **Beam size units**: FITS headers store BMAJ/BMIN in **degrees**. Pass user-provided values in **degrees** too (e.g. `0.000173` deg ≈ `0.622` arcsec).
- **Frequency units**: The `.dat` output uses **MHz** (frequency / 1e6). This matches spectuner's default.
- **Aperture size**: `--aperture-size` is in **arcseconds** for both circle (radius) and box (half-width).
- **Line-free noise region**: The default auto-offset (`--noise-auto-offset-asec 10.0`) places the noise pixel ~10 arcsec from the target. For crowded fields, the user should provide an explicit `--noise-ref-ra/--noise-ref-dec` in a truly line-free region.
- **Txt mode auto-detection**: If all input files end with `.txt`/`.dat`/`.spec`, the script automatically switches to txt mode. The user can force it with `--from-txt`.
- **Missing beam info**: If the FITS header lacks BMAJ/BMIN and the user does not provide `--beam-bmaj/--beam-bmin`, the script reports an error with a clear message asking for the beam size. Do not silently guess.
- **Large cubes**: `SpectralCube.read()` can consume significant memory. For very large cubes, the script sets `allow_huge_operations = True`. If memory is still an issue, consider extracting with `astropy.io.fits` directly.
- **4D cubes**: Some ALMA cubes have shape `(Stokes, Freq, Y, X)`. `SpectralCube` handles this automatically.
- **NaN handling**: If the target pixel or aperture contains NaN, the masked cube average automatically ignores them. Warn the user if many pixels in the aperture are masked.

## Output file structure

```
outdir/
├── *.dat                      # spectrum: freq(MHz)  Tb(K)
├── extract_summary.txt        # per-file metadata
└── spectuner_config.pkl       # (optional) spectuner pickle config
```

## Workflow integration with spectuner

After extraction, the `.dat` files and summary are ready for spectuner. Typical next steps (handled by the spectuner-run skill):

1. Load the pickle config (or build one via Python API)
2. Set `sl_model.fname_db` to the CDMS SQLite database path
3. Choose species list and fitting bounds
4. Run `exec_fit` or the Python API for line identification

When the user finishes extraction and asks to proceed with spectuner, automatically invoke the spectuner-run skill.

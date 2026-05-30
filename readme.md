# Spectuner-Agent

An AI agent built on top of [Spectuner](https://github.com/yqiuu/spectuner), enabling automated, natural-language-driven spectral line analysis for interstellar molecules.

## 📌 Project Overview

- **Base Framework**: This repository is built upon [yqiuu/spectuner](https://github.com/yqiuu/spectuner), which provides core spectral fitting algorithms including the LTE radiative transfer model, peak-matching loss function, particle swarm optimization, and deep reinforcement learning acceleration.
- **Added Feature**: A `skills/` module is added to package Spectuner as an AI agent that can be invoked by AI coding assistants (e.g., Claude Code, Cursor, Continue). This enables conversational, agentic workflows where users describe their analysis goals in natural language and the agent autonomously executes the necessary steps.

## 🛠️ Agent Skills

The agent implements three modular skills that together cover the complete spectral analysis workflow:

![The workflow](https://github.com/user-attachments/assets/442ed0cf-7542-446d-8cae-bf9d1e069f38)

### 1. `spectuner-extract-spectrum` — Data Extraction

Prepares observational data for analysis. Given a natural language command like *"extract the spectrum at RA=16h32m22s.58, DEC=-24°28'32.8'' with a 0.5" circular aperture"*, this skill:
- Reads FITS headers to extract frequency axis and beam parameters (BMAJ, BMIN)
- Extracts spectra using point, circular aperture, or rectangular box averaging
- Converts units from Jy/beam to brightness temperature (K)
- Estimates RMS noise from line-free off-source regions using sigma-clipped statistics
- Estimates continuum baseline from companion FITS or spectrum median
- Outputs standardized .dat files (frequency in MHz, intensity in K) with metadata summaries
- Also supports plain-text input files (.txt, .dat)

### 2. `spectuner-run` — Automated Fitting and Identification

Performs LTE spectral line fitting and molecule identification. Given a command like *"fit methanol and methyl formate in this spectrum"*, this skill:
- Loads configuration and queries the CDMS spectroscopic database for relevant transitions
- Executes Phase 1: individual molecule fitting using PSO with the robust peak-matching loss function
- Executes Phase 2: greedy combination fitting to resolve line blending across multiple species
- For spectral cubes: performs pixel-by-pixel fitting with optional deep reinforcement learning-based initial guesses (Spectuner-D1), generating spatially resolved maps of five physical parameters: source size ($\theta$), excitation temperature ($T_{\mathrm{ex}}$), column density ($N_{\mathrm{tot}}$), velocity width ($\Delta v$), and velocity offset ($v_{\mathrm{LSR}}$)
- Outputs results to HDF5 files with fitted parameters and identified line tables

### 3. `spectuner-plot-results` — Visualization

Transforms fitting results into publication-quality visualizations. Given a command like *"show me the peak plot for methanol with E_u labels"*, this skill generates:
- **PeakPlot**: Grid of subplots centered on each identified peak, showing observed (black) and fitted (red) spectra with $E_u$ annotations in green
- **SpectralPlot**: Full spectrum overview across all windows with color-coded peak classifications (red = matched, orange = blended, blue = false positives)
- **Parameter Maps**: 2D spatial distribution maps of the five fitted parameters from pixel-by-pixel cube fitting, with logarithmic normalization for log-scale parameters and preserved WCS coordinates
- **Peak Details Table**: Exports detailed transition tables including rest frequency, shifted frequency, $E_u$, $E_l$, $A_{ul}$, $g_u$, and quantum numbers ($J_u$, $J_l$) queried directly from CDMS

## 🚀 Quick Start

1. Follow the [original project's documentation](https://github.com/yqiuu/spectuner) to install dependencies and configure the CDMS database.
2. Clone this repository:
   ```bash
   git clone https://github.com/tianweivivi/spectuner-agent.git
   ```
3. Open the project in an AI editor that supports Skills (such as Claude Code, Cursor, or Continue). The `skills/` directory should be loaded automatically.
4. Interact with the agent using natural language. For example:
   > *"Extract the spectrum from data/my_cube.fits at the continuum peak, then fit methanol and methyl formate, and finally plot the peak plot for methanol."*

## 💡 Key Differentiators

Unlike prior AI spectral agents (e.g., Egent for optical stellar spectra, Spec-o3 for rare object classification), Spectuner Agent targets **interstellar molecular rotational spectra** in the **radio regime**, where the core challenges are:
- Matching observed lines to quantum-mechanical transition databases (CDMS/JPL)
- Fitting LTE physical parameters from radiative transfer models
- Handling severe line blending where dozens of molecular species overlap
- Scaling to pixel-by-pixel fitting of ALMA spectral cubes (10,000+ pixels) to generate spatially resolved parameter maps

## 🙏 Attribution & Citation

The core algorithms and code are the work of the original Spectuner team. If you find this agent useful in your research, please cite the original Spectuner papers:

```bibtex
@ARTICLE{2025ApJS..277...21Q,
       author = {{Qiu}, Yisheng and {Zhang}, Tianwei and {M{\"o}ller}, Thomas and {Jiang}, Xue-Jian and {Song}, Zihao and {Chen}, Huaxi and {Quan}, Donghui},
        title = "{Specturner: A Framework for Automated Line Identification of Interstellar Molecules}",
      journal = {\apjs},
         year = 2025,
        month = mar,
       volume = {277},
       number = {1},
          eid = {21},
        pages = {21},
          doi = {10.3847/1538-4365/adaeba}
}

@ARTICLE{2026ApJS..283....1Q,
       author = {{Qiu}, Yisheng and {Zhang}, Tianwei and {Liu}, Tie and {Zhu}, Fengyao and {Meng}, Dezhao and {Chen}, Huaxi and {M{\"o}ller}, Thomas and {Schilke}, Peter and {Quan}, Donghui},
        title = "{Spectuner-D1: Spectral Line Fitting of Interstellar Molecules Using Deep Reinforcement Learning}",
      journal = {\apjs},
         year = 2026,
        month = mar,
       volume = {283},
       number = {1},
          eid = {1},
        pages = {1},
          doi = {10.3847/1538-4365/ae3742}
}
```

## 📄 License

This project retains the same license as the original Spectuner. Please see the [LICENSE](./LICENSE) file for details.

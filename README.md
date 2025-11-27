# BMRS Energy Forecast Analysis

This repository contains a small analysis tool that visualizes BMRS day-ahead forecasts and (simulated or live) actuals for wind and solar generation, and plots the evolution of the indicated imbalance forecast across settlement periods.

Repository contents
- `bmrs_analysis.py` — the primary/original script. By default it uses simulated data; set `USE_REAL_BMRS = True` and provide a `BMRS_API_KEY` to fetch live BMRS data.

Requirements
- Python 3.8+
- Python packages: `pandas`, `numpy`, `matplotlib`, `pytz`, `requests`, `reportlab`, `Pillow`

Quick start

1. Change to the project directory:

```bash
cd /workspaces/bmrs-energy-forecast-analysis
```

2. (Optional) Create and activate a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. (Headless/server) If running without a display set Matplotlib backend:

```bash
export MPLBACKEND=Agg
```

5. Run the analysis script:

```bash
python bmrs_analysis.py
```

The script supports the following CLI flags:

```bash
python bmrs_analysis.py --help
```

**Available options:**
- `--date YYYY-MM-DD` (or `-d`) — Analysis date (default: 2025-11-20)
- `--use-real` — Fetch live BMRS data (requires `BMRS_API_KEY` environment variable)
- `--outdir PATH` — Output directory for PNG files and PDF report (default: `outputs/`)
- `--no-pdf` — Skip PDF report generation (only generate PNG files)

**Examples:**

```bash
# Default: simulated data for 2025-11-20, save to outputs/ with PDF
python bmrs_analysis.py

# Custom date with simulated data and PDF
python bmrs_analysis.py --date 2025-11-15 --outdir my_results/

# Use real BMRS data (requires API key) and generate PDF
export BMRS_API_KEY="YOUR_API_KEY"
python bmrs_analysis.py --date 2025-11-20 --use-real --outdir outputs/

# Generate PNGs only, skip PDF
python bmrs_analysis.py --no-pdf
```

What the script produces
- `indicated_imbalance_evolution_YYYY-MM-DD.png` — comparison of previous vs current indicated imbalance forecast across settlement periods.
- `indicated_imbalance_current_YYYY-MM-DD.png` — current/latest indicated imbalance forecast only.
- `wind_forecast_vs_actual_YYYY-MM-DD.png` and `solar_forecast_vs_actual_YYYY-MM-DD.png` — wind/solar forecast vs actuals.
- `wind_difference_table.png` and `solar_difference_table.png` — summary tables with forecast errors.
- **`BMRS_Report_YYYY-MM-DD.pdf`** — comprehensive PDF report containing:
  - All graphs (imbalance evolution, current imbalance, wind/solar forecast vs actuals)
  - Error tables with summary statistics (Average Error, Mean Absolute Error)
  - Expert commentary on forecast errors and system impact

All outputs are saved in the directory specified by `--outdir` (default: `outputs/`) with ISO date suffixes.

Using real BMRS data

To fetch live BMRS data you need an API key from Elexon/BMRS:

```bash
export BMRS_API_KEY="YOUR_API_KEY"
python bmrs_analysis.py --use-real --date 2025-11-20
```

Notes
- The script falls back to deterministic simulated data when BMRS data is unavailable or when `--use-real` is not set.
- Output files include an ISO date stamp (e.g., `2025-11-20`) for easy tracking across multiple runs.

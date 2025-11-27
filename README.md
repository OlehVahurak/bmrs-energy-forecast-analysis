# BMRS Energy Forecast Analysis

This repository contains a small analysis tool that visualizes BMRS day-ahead forecasts and (simulated or live) actuals for wind and solar generation, and plots the evolution of the indicated imbalance forecast across settlement periods.

Repository contents
- `bmrs_analysis.py` — the primary/original script. By default it uses simulated data; set `USE_REAL_BMRS = True` and provide a `BMRS_API_KEY` to fetch live BMRS data.

Requirements
- Python 3.8+
- Python packages: `pandas`, `numpy`, `matplotlib`, `pytz`, `requests`

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
pip install pandas numpy matplotlib pytz requests
```

4. (Headless/server) If running without a display set Matplotlib backend:

```bash
export MPLBACKEND=Agg
```

5. Run the analysis script:

```bash
python bmrs_analysis.py
```

What the script produces
- `indicated_imbalance_evolution.png` — comparison of previous vs current indicated imbalance forecast across settlement periods.
- `indicated_imbalance.png` — current/latest indicated imbalance forecast only.
- (If extended) `*_forecast_vs_actual.png` and `*_difference_table.png` — for wind/solar forecast vs actuals and summary tables.

Using real BMRS data

To fetch live BMRS data you need an API key from Elexon/BMRS. Set it in the environment and enable live mode in the script:

```bash
export BMRS_API_KEY="YOUR_API_KEY"
# then in bmrs_analysis.py set USE_REAL_BMRS = True
```

Notes
- The script is written to fall back to deterministic simulated data when BMRS data is unavailable or when `USE_REAL_BMRS` is `False`.
- Modify `SELECTED_DATE` inside the script to change the analysis day, or add CLI handling if you prefer.

Suggested next steps (optional)
- Add a `requirements.txt` file: `pip freeze > requirements.txt`.
- Add CLI flags (`--date`, `--use-real`, `--outdir`) using `argparse`.
- Save outputs in an `outputs/` directory with datestamps.

If you want, I can add any of the above optional improvements — tell me which one to implement next.

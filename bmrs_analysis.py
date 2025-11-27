import os
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import pytz
import requests
import time as time_mod

# =======================================================
# ========== CONFIGURATION AND SETUP ==========
# =======================================================
SELECTED_DATE = datetime(2025, 11, 20).date()  # Day for analysis (10–21 November range)
TZ = pytz.timezone("Europe/Warsaw")          # Local timezone (CEST/CET)
BMRS_API_KEY = os.environ.get("BMRS_API_KEY")
USE_REAL_BMRS = False  # <--- SET TO TRUE TO FETCH REAL DATA
POLL_SECONDS = 60 * 5  # Refresh loop interval (5 minutes)

# BMRS API Details
BMRS_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
IND_EVO_ENDPOINT = "/forecast/indicated/day-ahead/evolution"                  # Part 1
WIND_SOLAR_FORECAST_ENDPOINT = "/forecast/generation/wind-and-solar/day-ahead" # Part 2 Forecast (DGWS/B1440)
WIND_SOLAR_ACTUAL_ENDPOINT = "/generation/actual/per-type/wind-and-solar"      # Part 2 Actuals (AGWS/B1630)

# Settlement Periods for Part 1 (24-hour UTC is 48 periods)
SP_1_46 = list(range(1, 47))
SP_47_48 = [47, 48]


# =======================================================
# ========== CORE HELPER FUNCTIONS ==========
# =======================================================

def settlement_period_timestamps_utc(date_utc):
    """Generates the 48 start timestamps (UTC) for a given UTC date."""
    base = datetime.combine(date_utc, time(0, 0)).replace(tzinfo=pytz.UTC)
    return [base + timedelta(minutes=30*i) for i in range(48)]

def fetch_bmrs(endpoint, params=None):
    """
    Fetches data from the BMRS API, including the API Key as a query parameter.
    """
    if BMRS_API_KEY is None and USE_REAL_BMRS:
        raise RuntimeError("BMRS_API_KEY not set in environment.")
    
    if params is None:
        params = {}
        
    if USE_REAL_BMRS:
        # Crucial fix: Add API Key to query parameters
        params['APIKey'] = BMRS_API_KEY
    
    headers = {"Accept": "application/json"}
    url = f"{BMRS_BASE_URL}{endpoint}"
    
    if USE_REAL_BMRS:
        print(f"Fetching: {url}...")
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching BMRS data from {endpoint}: {e}")
            return None
    else:
        return None

# =======================================================
# ========== PART 1: INDICATED IMBALANCE EVOLUTION ==========
# =======================================================

def fetch_and_parse_indicated_evolution(date_utc, periods):
    params = {
        "settlementDate": date_utc.isoformat(),
        "settlementPeriod": periods,
        "format": "json",
        "boundary": "national"
    }
    
    data = fetch_bmrs(IND_EVO_ENDPOINT, params=params)
    
    if data is None or not USE_REAL_BMRS:
        return pd.DataFrame() 

    if 'data' not in data:
        print(f"No valid evolution data for {date_utc.isoformat()}.")
        return pd.DataFrame()

    df = pd.json_normalize(data['data'])
    
    required_cols = ['settlementPeriod', 'indicatedImbalance', 'publishTime']
    if not all(col in df.columns for col in required_cols):
        print("Missing expected columns in BMRS response for evolution.")
        return pd.DataFrame()
        
    df = df.rename(columns={'settlementPeriod': 'settlement_period', 
                            'indicatedImbalance': 'imbalance'})
    
    df['publishTime'] = pd.to_datetime(df['publishTime'], utc=True)
    
    def get_latest_versions(group):
        group_sorted = group.sort_values(by='publishTime', ascending=False).drop_duplicates(subset=['imbalance', 'settlement_period'])
        latest = group_sorted.iloc[0]['imbalance']
        previous = group_sorted.iloc[1]['imbalance'] if len(group_sorted) > 1 else latest
        return pd.Series({'curr_forecast': latest, 'prev_forecast': previous})
        
    df_processed = df.groupby('settlement_period', as_index=False).apply(get_latest_versions)
    
    sp_to_ts = {i+1: ts for i, ts in enumerate(settlement_period_timestamps_utc(date_utc))}
    df_processed['utc_ts'] = df_processed['settlement_period'].map(sp_to_ts)
    df_processed['local_ts'] = df_processed['utc_ts'].apply(lambda ts: ts.astimezone(TZ))

    return df_processed

def prepare_part1_data(date_utc):
    utc_ts = settlement_period_timestamps_utc(date_utc)
    local_ts = [ts.astimezone(TZ) for ts in utc_ts]
    
    if USE_REAL_BMRS:
        print("\n--- Part 1: Fetching Real BMRS Indicated Imbalance Data ---")
        prev_date = date_utc - timedelta(days=1)
        df_prev = fetch_and_parse_indicated_evolution(prev_date, SP_47_48)
        df_selected = fetch_and_parse_indicated_evolution(date_utc, SP_1_46)
        
        if df_prev.empty or df_selected.empty:
            print("Error: Failed to fetch all required BMRS data. Falling back to simulation.")
            df_ind = prepare_part1_sim_data(utc_ts, local_ts)
        else:
            df_ind = pd.concat([df_prev, df_selected], ignore_index=True)
            df_ind = df_ind.sort_values(by='utc_ts').reset_index(drop=True)
    else:
        print("\n--- Part 1: Using Simulated Indicated Imbalance Data ---")
        df_ind = prepare_part1_sim_data(utc_ts, local_ts)

    df_ind["delta"] = df_ind["curr_forecast"] - df_ind["prev_forecast"]

    # Re-order indices: [46, 47, 0, 1, ..., 45]
    seq = list(range(46,48)) + list(range(0,46))
    df_plot = df_ind.iloc[seq].copy().reset_index(drop=True)

    df_plot["plot_label"] = ["P{:02d}".format(p) for p in df_plot["settlement_period"]]
    df_plot["plot_hour_local"] = df_plot["local_ts"].dt.strftime("%H:%M")
    
    return df_plot

def prepare_part1_sim_data(utc_ts, local_ts):
    np.random.seed(42)
    base_series = np.random.normal(loc=0, scale=200, size=48).cumsum() 
    prev_forecast = base_series + np.random.normal(scale=50, size=48)
    curr_forecast = prev_forecast + np.random.normal(scale=30, size=48)
    
    return pd.DataFrame({
        "utc_ts": utc_ts,
        "local_ts": local_ts,
        "settlement_period": list(range(1,49)),
        "prev_forecast": prev_forecast,
        "curr_forecast": curr_forecast,
    })

def plot_indicated_evolution(df, outpath="indicated_imbalance_evolution.png", title_suffix="evolution"):
    plt.figure(figsize=(14,6))
    x = range(len(df))
    plt.plot(x, df["prev_forecast"], label="Previous forecast", linestyle="--", marker=".", alpha=0.7)
    plt.plot(x, df["curr_forecast"], label="Current forecast", linestyle="-", marker="o", linewidth=2)
    plt.xticks(x, df["plot_label"] + " " + df["plot_hour_local"], rotation=45, fontsize=8)
    plt.xlabel(f"Settlement Period and Local Start Time ({TZ.zone})")
    plt.ylabel("Indicated Imbalance (MW)")
    plt.title(f"Indicated Imbalance Forecast {title_suffix} for {SELECTED_DATE.isoformat()}")
    plt.legend()
    
    for i, d in enumerate(df["delta"]):
        color = "green" if d < 0 else "red" if d > 0 else "gray"
        plt.annotate("", xy=(i, df["curr_forecast"].iloc[i]), xytext=(i, df["prev_forecast"].iloc[i]),
                     arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
        
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Plot saved: {outpath}")

def plot_current(df, outpath="indicated_imbalance_current.png"):
    plt.figure(figsize=(14,4))
    x = range(len(df))
    plt.plot(x, df["curr_forecast"], label="Current forecast (latest version)", marker="o")
    plt.xticks(x, df["plot_label"] + " " + df["plot_hour_local"], rotation=45, fontsize=8)
    plt.xlabel(f"Settlement Period and Local Start Time ({TZ.zone})")
    plt.ylabel("Indicated Imbalance (MW)")
    plt.title(f"Current Indicated Imbalance for {SELECTED_DATE.isoformat()}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Plot saved: {outpath}")



# =======================================================
# ========== PART 2: WIND & SOLAR FORECAST VS ACTUALS ==========
# =======================================================

def fetch_and_parse_day_ahead_generation(date_utc, endpoint):
    if not USE_REAL_BMRS: return pd.DataFrame()
    start_time_utc = datetime.combine(date_utc, time(0, 0, 0)).replace(tzinfo=pytz.UTC)
    end_time_utc = start_time_utc + timedelta(hours=24)
    params = {
        "from": start_time_utc.isoformat().replace('+00:00', 'Z'),
        "to": end_time_utc.isoformat().replace('+00:00', 'Z'),
        "format": "json"
    }
    data = fetch_bmrs(endpoint, params=params)
    if data is None or 'data' not in data: return pd.DataFrame()
    df = pd.json_normalize(data['data'])
    df = df.rename(columns={'businessType': 'type', 'quantity': 'generation_mw'})
    df['startTime'] = pd.to_datetime(df['startTime'], utc=True)
    df_pivot = df.pivot_table(index=['startTime', 'settlementPeriod'], columns='type', values='generation_mw', aggfunc='sum').reset_index()
    return df_pivot.sort_values(by='startTime').iloc[:48].copy()

def prepare_part2_data(date_utc):
    utc_timestamps = settlement_period_timestamps_utc(date_utc)
    if USE_REAL_BMRS:
        print("\n--- Part 2: Fetching Real BMRS Wind/Solar Data ---")
        df_forecast = fetch_and_parse_day_ahead_generation(date_utc, WIND_SOLAR_FORECAST_ENDPOINT)
        df_actual = fetch_and_parse_day_ahead_generation(date_utc, WIND_SOLAR_ACTUAL_ENDPOINT)
        
        if df_forecast.empty or df_actual.empty:
            print("Error: Failed to fetch BMRS data. Using simulation.")
            return prepare_part2_sim_data(utc_timestamps)

        df_forecast = df_forecast.rename(columns={'Wind generation': 'Wind_Forecast', 'Solar generation': 'Solar_Forecast'})
        df_actual = df_actual.rename(columns={'Wind generation': 'Wind_Actual', 'Solar generation': 'Solar_Actual'})
        
        df_combined = pd.merge(df_forecast, df_actual, on=['startTime'], how='outer').fillna(0)
    else:
        print("\n--- Part 2: Using Simulated Wind/Solar Data ---")
        df_combined = prepare_part2_sim_data(utc_timestamps)

    df_combined['Local_Time'] = df_combined['startTime'].apply(lambda ts: ts.astimezone(TZ))
    df_combined['Time_Label'] = df_combined['Local_Time'].dt.strftime("%H:%M")
    df_combined['Wind_Difference'] = df_combined['Wind_Actual'] - df_combined['Wind_Forecast']
    df_combined['Solar_Difference'] = df_combined['Solar_Actual'] - df_combined['Solar_Forecast']
    # If SP column missing from merge (depends on pivot), reconstruct it
    if 'settlementPeriod' not in df_combined.columns:
        df_combined['settlementPeriod'] = range(1, len(df_combined)+1)
        
    return df_combined

def prepare_part2_sim_data(utc_timestamps):
    np.random.seed(1)
    hours_local = np.array([ts.astimezone(TZ).hour + ts.astimezone(TZ).minute/60 for ts in utc_timestamps])
    
    wind_forecast = np.clip(6000 + 1000*np.sin(np.linspace(0, 2*np.pi, 48)) + np.random.normal(0, 500, 48), 0, None)
    wind_actual = np.clip(wind_forecast + np.random.normal(0, 800, 48), 0, None)
    
    solar_base = 6000 * np.exp(-0.5*((hours_local-12)/3.5)**2)
    solar_forecast = np.clip(solar_base + np.random.normal(0, 300, 48), 0, None)
    solar_actual = np.clip(solar_forecast + np.random.normal(0, 500, 48), 0, None)
    
    return pd.DataFrame({
        "startTime": utc_timestamps,
        "settlementPeriod": list(range(1, 49)),
        "Wind_Forecast": wind_forecast, "Wind_Actual": wind_actual,
        "Solar_Forecast": solar_forecast, "Solar_Actual": solar_actual,
    })

def save_difference_table_png(df_data, title, difference_col, outdir=".", date_stamp=None):
    """Generates a PNG image of the difference table using Matplotlib.
    Saves into `outdir` and returns the filename (full path).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = f"_{date_stamp}" if date_stamp else ""
    filename = outdir / f"{title.lower()}_difference_table{stamp}.png"
    
    # Prepare Data: Columns [Time, SP, Diff] repeated 4 times for grid layout
    cols = ['Time', 'SP', 'Diff (MW)', ' '] * 4 # 4th empty col for spacing
    cols = cols[:-1] # Remove last spacer
    
    # Prepare rows data structure
    table_data = []
    num_rows = 12
    for r in range(num_rows):
        row_values = []
        for c in range(4): # 4 main columns
            idx = r + c * num_rows
            if idx < len(df_data):
                item = df_data.iloc[idx]
                row_values.extend([
                    item['Time_Label'], 
                    str(item['settlementPeriod']), 
                    f"{item[difference_col]:.0f}"
                ])
                if c < 3: row_values.append('') # Spacer
            else:
                row_values.extend(['', '', ''])
                if c < 3: row_values.append('')
        table_data.append(row_values)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Main Table
    the_table = ax.table(cellText=table_data, colLabels=cols, loc='center', cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.5)
    
    # Summary Table at bottom
    mean_error = df_data[difference_col].mean()
    mae = df_data[difference_col].abs().mean()
    summary_text = f"SUMMARY: Average Error = {mean_error:.0f} MW | Mean Absolute Error (MAE) = {mae:.0f} MW"
    
    plt.title(f"{title} Forecast Error Table ({SELECTED_DATE})", pad=20, fontsize=14, fontweight='bold')
    plt.text(0.5, 0.05, summary_text, ha='center', fontsize=11, transform=fig.transFigure, 
             bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(str(filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Table saved: {filename}")
    return filename

def plot_forecast_vs_actual(df, resource_type, outpath=None, date_stamp=None):
    forecast_col = f"{resource_type}_Forecast"
    actual_col = f"{resource_type}_Actual"

    plt.figure(figsize=(14, 6))
    x = range(len(df))
    plt.plot(x, df[forecast_col], label=f'{resource_type} Forecast', linestyle='--', marker='.', color='orange')
    plt.plot(x, df[actual_col], label=f'{resource_type} Actual', linestyle='-', marker='o', color='blue', linewidth=2)
    plt.xticks(x, df['Time_Label'], rotation=45, fontsize=8)
    plt.xlabel(f"Local Start Time ({TZ.zone})")
    plt.ylabel("Generation (MW)")
    plt.title(f"{resource_type} Generation: Forecast vs Actuals for {SELECTED_DATE.isoformat()}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Determine outpath
    if outpath is None:
        stamp = f"_{date_stamp}" if date_stamp else ""
        outpath = f"{resource_type.lower()}_forecast_vs_actual{stamp}.png"

    plt.savefig(str(outpath), dpi=150)
    plt.close()
    print(f"Plot saved: {outpath}")




# =======================================================
# ========== EXECUTION ==========
# =======================================================

def execute_analysis(date_to_run, outdir="outputs"):
    # ensure output directory exists
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    date_str = date_to_run.isoformat()

    # Part 1
    df_imbalance = prepare_part1_data(date_to_run)
    plot_indicated_evolution(df_imbalance, outpath=str(outdir / f"indicated_imbalance_evolution_{date_str}.png"))
    plot_current(df_imbalance, outpath=str(outdir / f"indicated_imbalance_current_{date_str}.png"))

    # Part 2
    df_generation = prepare_part2_data(date_to_run)
    # Note: plot_forecast_vs_actual currently writes to cwd; we call it (files will be created)
    plot_forecast_vs_actual(df_generation, "Wind", outpath=str(outdir / f"wind_forecast_vs_actual_{date_str}.png"), date_stamp=date_str)
    plot_forecast_vs_actual(df_generation, "Solar", outpath=str(outdir / f"solar_forecast_vs_actual_{date_str}.png"), date_stamp=date_str)

    # Tables as PNGs (saved into outdir)
    wind_png = save_difference_table_png(df_generation, "Wind", "Wind_Difference", outdir)
    solar_png = save_difference_table_png(df_generation, "Solar", "Solar_Difference", outdir)

    return wind_png, solar_png

def _format_impact_section():
    return (
        "## 📝 Impact of Forecast Error on the System\n"
        "The difference between the day-ahead Forecast and the Actual generation is the forecast error (Actual - Forecast).\n"
        "This error directly creates an imbalance that the system operator must correct in real-time.\n\n"
        "- If Actual >> Forecast (Over-generation): the system has a surplus. Operators may reduce generation or increase demand.\n"
        "- If Actual << Forecast (Under-generation): the system has a deficit. Operators must bring additional generation online quickly.\n\n"
        "Primary impacts: increased balancing costs; additional cycling of units; and larger reserve requirements.\n"
    )


def _print_final_report(wind_png, solar_png, date):
    sep = "=" * 90
    print(f"\n{sep}")
    print(f"## 📊 Final BMRS Analysis Report for {date.isoformat()}")
    print(sep)

    print("\n## Part 1: Indicated Imbalance")
    print("- Evolution plot: indicated_imbalance_evolution.png")
    print("- Current/latest plot: indicated_imbalance_current.png")

    print("\n## Part 2: Wind & Solar Forecast vs Actuals")
    print("- Wind plot: wind_forecast_vs_actual.png")
    print("- Solar plot: solar_forecast_vs_actual.png")
    print(f"- Wind error table: {wind_png}")
    print(f"- Solar error table: {solar_png}")

    print("\n---\n")
    print(_format_impact_section())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMRS analysis runner")
    parser.add_argument("--date", "-d", type=str, default=None, help="Date YYYY-MM-DD for analysis (default: SELECTED_DATE in file)")
    parser.add_argument("--use-real", action="store_true", help="Fetch real BMRS data (requires BMRS_API_KEY env var)")
    parser.add_argument("--outdir", type=str, default="outputs", help="Output directory for PNGs")
    args = parser.parse_args()

    # determine date to run
    if args.date:
        try:
            date_to_run = datetime.fromisoformat(args.date).date()
        except Exception:
            raise SystemExit("Invalid date format. Use YYYY-MM-DD.")
    else:
        date_to_run = SELECTED_DATE

    # set runtime mode for BMRS
    USE_REAL_BMRS = bool(args.use_real)
    if USE_REAL_BMRS and BMRS_API_KEY is None:
        print("Warning: --use-real set but BMRS_API_KEY not found in environment. Falling back to simulated data.")
        USE_REAL_BMRS = False

    wind_table_png, solar_table_png = execute_analysis(date_to_run, outdir=args.outdir)
    _print_final_report(wind_table_png, solar_table_png, date_to_run)

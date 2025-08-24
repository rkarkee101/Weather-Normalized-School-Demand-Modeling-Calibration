# Solar + BESS optimization for a school with 2‑week (dummy) weather forecast
# This cell generates a self-contained demo with plots and CSV outputs.
# You can download the generated CSV/PNG files from the links I’ll provide below.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from datetime import datetime, timedelta
import os

# ----------------------
# 1) USER-EDITABLE CONFIG
# ----------------------
CONFIG = {
    # Simulation window
    "start_date": "2025-09-01",     # YYYY-MM-DD (local time)
    "days": 14,                      # simulate next 2 weeks
    
    # School schedule: weekdays open, weekends closed
    "weekend_closed": True,
    # Optionally list closure dates (YYYY-MM-DD) as strings (e.g., holidays)
    "extra_closures": [],
    
    # Tariff
    "import_price_flat": 0.12,       # $/kWh (flat) — can override by providing TOU below
    # If you want TOU, provide a list of 24 prices; it will repeat daily
    "tou_24": None,                  # e.g., [0.08]*6 + [0.15]*10 + [0.32]*5 + [0.12]*3
    "export_price": 0.05,            # $/kWh feed-in tariff
    "demand_charge_rate": 12.0,      # $/kW over the simulated peak
    
    # PV system
    "pv_dc_kw": 150.0,               # DC size (kWdc)
    "pv_dc_to_ac_ratio": 1.1,        # DC/AC ratio -> limits peak AC output
    "pv_ac_limit_kw": None,          # If None, computed from pv_dc_kw / ratio
    
    # Battery
    "bess_capacity_kwh": 200.0,
    "bess_charge_kw": 50.0,
    "bess_discharge_kw": 50.0,
    "bess_initial_soc_kwh": 0.0,
    "allow_grid_charging": True,
    "allow_grid_export": True,
    
    # Load model (baseline profile and weather sensitivity)
    "baseline_weekday_profile_kw": (
        [12]*6 + [28,35,40,45] + [30,28,26,24] + [26,38,48,58] + [70,70,60,45] + [22,18]
    ),
    "baseline_weekend_kw": 6.0,      # constant baseload on weekends/closures
    # Weather sensitivity (simple linear model around 22°C neutral)
    "neutral_temp_C": 22.0,
    "cooling_slope_kw_per_C": 1.2,   # increase per degree above neutral during 9-20h
    "heating_slope_kw_per_C": 0.6,   # increase per degree below neutral during 6-9h & 17-22h
    
    # Weather (dummy forecast file path; if file exists, it will be loaded; else created)
    "weather_csv_path": "/mnt/data/dummy_weather_2w.csv",
    # If creating dummy: mean temps and GHI envelopes for the 2 weeks
    "dummy_mean_temp_C": 27.0,       # overall warm period
    "dummy_temp_daily_amp": 6.0,     # day/night swing
    "dummy_ghi_peak_Wm2": 800.0,     # midday peak GHI (clear day)
    "dummy_cloudiness_scale": 0.85,  # 0..1 multiplier for clouds (1 = clear)
    
    # Outputs
    "out_dir": "/mnt/data/school_bess_outputs"
}

os.makedirs(CONFIG["out_dir"], exist_ok=True)

# ----------------------
# 2) WEATHER: CREATE OR LOAD 2-WEEK HOURLY FORECAST (DUMMY)
# ----------------------
start_dt = datetime.fromisoformat(CONFIG["start_date"])
hours = CONFIG["days"] * 24
time_index = [start_dt + timedelta(hours=i) for i in range(hours)]

weather_path = CONFIG["weather_csv_path"]
if not os.path.exists(weather_path):
    # Create a simple sinusoidal daily temperature + bell-shaped GHI per day
    rows = []
    for i in range(hours):
        t = time_index[i]
        hour = t.hour
        # Temperature: daily sinusoid around dummy_mean_temp_C
        temp = CONFIG["dummy_mean_temp_C"] + CONFIG["dummy_temp_daily_amp"] * np.sin(2*np.pi*(hour-6)/24.0)
        # GHI: daytime bell (rough)
        if 6 <= hour <= 18:
            x = (hour - 12) / 6.0  # -1..+1 across the solar window
            ghi = CONFIG["dummy_cloudiness_scale"] * CONFIG["dummy_ghi_peak_Wm2"] * np.exp(-x*x)
        else:
            ghi = 0.0
        rows.append((t.isoformat(), hour, float(temp), float(ghi)))
    wdf = pd.DataFrame(rows, columns=["timestamp", "hour", "temp_C", "GHI_Wm2"])
    wdf.to_csv(weather_path, index=False)
else:
    wdf = pd.read_csv(weather_path)

# Ensure length matches
wdf = wdf.iloc[:hours].copy()
wdf["timestamp"] = pd.to_datetime(wdf["timestamp"] if "timestamp" in wdf else time_index)
wdf["hour"] = wdf["hour"] if "hour" in wdf else [t.hour for t in time_index]

# ----------------------
# 3) BUILD LOAD (DEMAND) AND PV USING WEATHER
# ----------------------
# Weekday/weekend/closure flags
dates = [dt.date().isoformat() for dt in time_index]
is_weekend = [(dt.weekday() >= 5) for dt in time_index] if CONFIG["weekend_closed"] else [False]*hours
is_closed = [d in CONFIG["extra_closures"] for d in dates]

# Baseline (weekday) profile repeated daily
weekday_profile = np.array(CONFIG["baseline_weekday_profile_kw"], dtype=float)
assert len(weekday_profile) == 24

demand = np.zeros(hours, dtype=float)
for i, t in enumerate(time_index):
    if is_weekend[i] or is_closed[i]:
        demand[i] = CONFIG["baseline_weekend_kw"]
    else:
        demand[i] = weekday_profile[t.hour]

# Weather sensitivity: adjust demand based on temperature and hour windows
temp = wdf["temp_C"].to_numpy()
neutral = CONFIG["neutral_temp_C"]
cooling_excess = np.maximum(0.0, temp - neutral)  # degrees above neutral
heating_excess = np.maximum(0.0, neutral - temp)  # degrees below neutral

cooling_mask = np.array([9 <= t.hour <= 20 for t in time_index])   # typical school occupied hours
heating_mask = np.array([(6 <= t.hour <= 9) or (17 <= t.hour <= 22) for t in time_index])

demand += cooling_mask * CONFIG["cooling_slope_kw_per_C"] * cooling_excess
demand += heating_mask * CONFIG["heating_slope_kw_per_C"] * heating_excess

# PV from GHI (very simple proxy): scale GHI to PV AC output subject to AC limit
ghi = wdf["GHI_Wm2"].to_numpy()  # W/m2
# Convert to kW/m2 and multiply by an effective area/scaling to reach approx pv_dc_kw at peak
# This is a crude proportional model; in practice use PVLib or SAM for accuracy.
pv_dc_kw = CONFIG["pv_dc_kw"]
dc_to_ac = CONFIG["pv_dc_to_ac_ratio"]
pv_ac_limit = CONFIG["pv_ac_limit_kw"] or (pv_dc_kw / dc_to_ac)
# We choose a scale so that clear-sky peak ~ pv_dc_kw
scale = pv_dc_kw / max(CONFIG["dummy_ghi_peak_Wm2"], 1.0)
pv_dc = (ghi * scale) / 1000.0  # kWdc
pv_ac = np.minimum(pv_dc, pv_ac_limit)  # clip by AC limit

# ----------------------
# 4) TARIFFS (IMPORT/EXPORT PRICES)
# ----------------------
if CONFIG["tou_24"] is not None:
    assert len(CONFIG["tou_24"]) == 24
    import_price = np.tile(np.array(CONFIG["tou_24"], dtype=float), CONFIG["days"])
else:
    import_price = np.full(hours, CONFIG["import_price_flat"], dtype=float)
export_price = np.full(hours, CONFIG["export_price"], dtype=float)
demand_charge_rate = CONFIG["demand_charge_rate"]

# ----------------------
# 5) OPTIMIZATION (LP)
# ----------------------
cap = CONFIG["bess_capacity_kwh"]
chg_max = CONFIG["bess_charge_kw"]
dis_max = CONFIG["bess_discharge_kw"]
soc0 = CONFIG["bess_initial_soc_kwh"]
allow_grid_chg = CONFIG["allow_grid_charging"]
allow_export = CONFIG["allow_grid_export"]

h = hours
idx_grid_imp       = lambda t: t
idx_grid_exp       = lambda t: h + t
idx_batt_chg_grid  = lambda t: 2*h + t
idx_batt_chg_solar = lambda t: 3*h + t
idx_batt_dis       = lambda t: 4*h + t
idx_soc            = lambda t: 5*h + t
n_vars = 5*h + (h + 1) + 1
peak_idx = n_vars - 1

c = np.zeros(n_vars)
for t in range(h):
    c[idx_grid_imp(t)] = import_price[t]
    c[idx_grid_exp(t)] = -export_price[t]
c[peak_idx] = demand_charge_rate

A_eq = []
b_eq = []
# Power balance: grid_imp + batt_dis + pv_ac = demand + batt_chg_grid + batt_chg_solar + grid_exp
for t in range(h):
    row = np.zeros(n_vars)
    row[idx_grid_imp(t)] =  1
    row[idx_batt_dis(t)] =  1
    row[idx_batt_chg_grid(t)]  = -1
    row[idx_batt_chg_solar(t)] = -1
    row[idx_grid_exp(t)]       = -1
    A_eq.append(row)
    b_eq.append(demand[t] - pv_ac[t])
# SOC continuity
for t in range(h):
    row = np.zeros(n_vars)
    row[idx_soc(t)] = -1
    row[idx_soc(t+1)] = 1
    row[idx_batt_chg_grid(t)]  = 1
    row[idx_batt_chg_solar(t)] = 1
    row[idx_batt_dis(t)]       = -1
    A_eq.append(row)
    b_eq.append(0.0)
# Initial SOC
row = np.zeros(n_vars); row[idx_soc(0)] = 1
A_eq.append(row); b_eq.append(soc0)
# Final SOC equals initial
row = np.zeros(n_vars); row[idx_soc(h)] = 1; row[idx_soc(0)] = -1
A_eq.append(row); b_eq.append(0.0)

A_ub = []
b_ub = []
# Peak: grid_import[t] <= P_peak
for t in range(h):
    row = np.zeros(n_vars)
    row[idx_grid_imp(t)] = 1
    row[peak_idx] = -1
    A_ub.append(row); b_ub.append(0.0)
# Charge rate limit
for t in range(h):
    row = np.zeros(n_vars)
    row[idx_batt_chg_grid(t)]  = 1
    row[idx_batt_chg_solar(t)] = 1
    A_ub.append(row); b_ub.append(chg_max)

# Bounds
bounds = [(0, None)] * n_vars
for t in range(h):
    bounds[idx_grid_imp(t)] = (0, None)
    bounds[idx_grid_exp(t)] = (0, None) if allow_export else (0, 0)
    bounds[idx_batt_chg_grid(t)]  = (0, chg_max) if allow_grid_chg else (0, 0)
    bounds[idx_batt_chg_solar(t)] = (0, chg_max)
    bounds[idx_batt_dis(t)]       = (0, dis_max)
for t in range(h+1):
    bounds[idx_soc(t)] = (0, cap)
bounds[idx_soc(0)] = (soc0, soc0)
bounds[peak_idx]   = (0, None)

A_eq = np.array(A_eq); b_eq = np.array(b_eq)
A_ub = np.array(A_ub); b_ub = np.array(b_ub)

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

if not res.success:
    print("Optimization failed:", res.message)
else:
    print("Optimization successful:", res.message)

x = res.x
grid_imp   = x[idx_grid_imp(0): idx_grid_imp(0)+h]
grid_exp   = x[idx_grid_exp(0): idx_grid_exp(0)+h]
bchg_grid  = x[idx_batt_chg_grid(0): idx_batt_chg_grid(0)+h]
bchg_solar = x[idx_batt_chg_solar(0): idx_batt_chg_solar(0)+h]
bdis       = x[idx_batt_dis(0): idx_batt_dis(0)+h]
soc        = x[idx_soc(0): idx_soc(0)+h+1]
p_peak     = x[peak_idx]

energy_cost = float(np.dot(grid_imp, import_price) - np.dot(grid_exp, export_price))
demand_cost = float(p_peak * demand_charge_rate)
total_cost  = energy_cost + demand_cost

# ----------------------
# 6) SAVE CSV OUTPUTS
# ----------------------
df = pd.DataFrame({
    "timestamp": time_index,
    "Demand_kW": demand,
    "PV_AC_kW": pv_ac,
    "Grid_Import_kW": grid_imp,
    "Grid_Export_kW": grid_exp,
    "Batt_Charge_from_Grid_kW": bchg_grid,
    "Batt_Charge_from_Solar_kW": bchg_solar,
    "Batt_Discharge_kW": bdis,
    "SOC_kWh": soc[:-1],
    "Import_Price_$perkWh": import_price,
    "Export_Price_$perkWh": export_price
})
sched_csv = os.path.join(CONFIG["out_dir"], "optimized_schedule_2w.csv")
df.to_csv(sched_csv, index=False)

summary = pd.DataFrame({
    "Metric": ["Peak Grid Import (kW)", "Energy Cost ($)", "Demand Charge ($)", "Total Cost ($)"],
    "Value": [p_peak, energy_cost, demand_cost, total_cost]
})
summary_csv = os.path.join(CONFIG["out_dir"], "cost_summary.csv")
summary.to_csv(summary_csv, index=False)

# ----------------------
# 7) PLOTS (each as its own figure; no explicit colors set)
# ----------------------

# Helper: make day separators for readability
def add_day_grid(ax):
    days_unique = sorted(list(set([dt.date() for dt in time_index])))
    for d in days_unique:
        # vertical line at midnight for each day boundary
        idxs = [i for i, t in enumerate(time_index) if t.hour == 0 and t.date() == d]
        for i in idxs:
            ax.axvline(i, linewidth=0.5, linestyle="--")

# Plot 1: Demand vs PV vs Grid Import and Battery Discharge (stack-ish view)
plt.figure(figsize=(12, 4))
plt.plot(demand, label="Demand (kW)")
plt.plot(pv_ac, label="PV AC (kW)")
plt.plot(grid_imp, label="Grid Import (kW)")
plt.plot(bdis, label="Battery Discharge (kW)")
plt.title("Demand, PV, Grid Import, and Battery Discharge (2 weeks)")
plt.xlabel("Hour")
plt.ylabel("kW")
plt.legend(loc="upper right")
add_day_grid(plt.gca())
plot1_path = os.path.join(CONFIG["out_dir"], "plot_timeseries_power.png")
plt.tight_layout()
plt.savefig(plot1_path, dpi=150)
plt.show()

# Plot 2: Battery SoC profile
plt.figure(figsize=(12, 3.5))
plt.plot(soc[:-1])
plt.title("Battery State of Charge (kWh)")
plt.xlabel("Hour")
plt.ylabel("kWh")
add_day_grid(plt.gca())
plot2_path = os.path.join(CONFIG["out_dir"], "plot_soc.png")
plt.tight_layout()
plt.savefig(plot2_path, dpi=150)
plt.show()

# Plot 3: Grid import vs peak line
plt.figure(figsize=(12, 3.5))
plt.plot(grid_imp, label="Grid Import (kW)")
plt.axhline(p_peak, linestyle="--", label="Peak Demand (kW)")
plt.title("Grid Import with Peak Demand")
plt.xlabel("Hour")
plt.ylabel("kW")
plt.legend(loc="upper right")
add_day_grid(plt.gca())
plot3_path = os.path.join(CONFIG["out_dir"], "plot_grid_peak.png")
plt.tight_layout()
plt.savefig(plot3_path, dpi=150)
plt.show()

# Plot 4: Daily maxima of grid import
daily_max = (
    pd.Series(grid_imp)
    .groupby([(i//24) for i in range(h)]).max()
)
plt.figure(figsize=(8, 3.5))
plt.bar(np.arange(len(daily_max)), daily_max.values)
plt.title("Daily Peak Grid Import (kW)")
plt.xlabel("Day index")
plt.ylabel("kW")
plot4_path = os.path.join(CONFIG["out_dir"], "plot_daily_peaks.png")
plt.tight_layout()
plt.savefig(plot4_path, dpi=150)
plt.show()

# ----------------------
# 8) PRINT KEY OUTPUT PATHS
# ----------------------
print("CSV outputs:")
print(" - Schedule:", sched_csv)
print(" - Cost summary:", summary_csv)
print("Plot images:")
print(" -", plot1_path)
print(" -", plot2_path)
print(" -", plot3_path)
print(" -", plot4_path)

{"sched_csv": sched_csv, "summary_csv": summary_csv, "plots": [plot1_path, plot2_path, plot3_path, plot4_path], "p_peak": p_peak, "total_cost": total_cost}

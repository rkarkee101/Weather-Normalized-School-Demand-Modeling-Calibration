# School Load Disaggregation & Hourly Profiling

A Python script that converts a **monthly utility bill** for a school into **hourly end‑use profiles** (HVAC, interior/exterior lighting with daylighting, plug loads, kitchen, optional PV). It **pulls historical weather** for any past date range, applies **schedule logic**, **daylight‑responsive lighting**, **night/weekend fans**, and **scales** the result to match the **billing kWh**. Exports **CSV tables** and a clean **overlay plot**.

---

## Features

- **Historical weather** by latitude/longitude for any *past* date range (Open‑Meteo archive or Meteostat).
- **Schedules** for school/staff/cleaning hours, weekends, and explicit **holiday** dates.
- **Interior lighting** with full/partial/cleaning states **by hour**, plus **daylight dimming** using GHI (can go to zero at bright midday).
- **Exterior lighting** by irradiance threshold or fixed on/off hours.
- **HVAC (cooling)** with simple load model proportional to (T_out − setpoint), **night/weekend fans** when AC is off (no double counting), optional minimal weekend cooling.
- **Plug loads** (day vs night) and **kitchen** (breakfast/lunch/prep windows).
- **PV netting** (optional) using irradiance (or a sine fallback).
- **Bill calibration:** scales hourly result to match your **target_total_kwh**.
- **Outputs:** 
  - `school_load_profile.csv` (24×N, hourly kW)
  - `school_daily_summary.csv` (date, daily kWh, daily peak kW)
  - `school_overlay.png` (Day 1 light → Day N dark; dashed = closed/holiday)

---

## Installation

```bash
pip install -r requirements.txt
# or
pip install requests pandas matplotlib meteostat pytz
```

> Works with Matplotlib 3.x (the script handles colormap API differences).

---

## Quick Start

1. **Put files in a folder** (e.g., this repo structure below).
2. **Open** `school_load_model.py` and edit the **Inputs** block (top of file). Minimal items:

```python
# Site & timezone
latitude  = 39.7392
longitude = -104.9903
timezone_str = "America/Denver"

# Date window (PAST, inclusive)
start_date = "2025-02-01"
end_date   = "2025-02-28"

# Billing kWh target for this window
target_total_kwh = 50000

# Weather source (either)
WEATHER_SOURCE = "open-meteo"  # or "meteostat"
```

3. Optionally adjust **schedules** and **end‑uses** (see “Inputs & What to Edit” below).
4. **Run** the script:
   ```bash
   python school_load_model.py
   ```
5. You’ll get:
   - `school_load_profile.csv`
   - `school_daily_summary.csv`
   - `school_overlay.png`

---

## Methods (How the model computes things)

### 1) Weather ingestion & normalization
- Fetches hourly **air temperature** (°C) and (if available) **shortwave radiation** (GHI, W/m²) for `start_date..end_date` in **local time**.
- Normalizes each **calendar day to 24 hours** (handles DST):
  - Duplicate hours (fall‑back) → **averaged**
  - Missing hours (spring‑forward) → **interpolated** (irradiance defaults to 0 at night)
- Arrays produced: `temps_by_day[day][hour]`, `ghi_by_day[day][hour]`.

### 2) End‑use load construction (kW per hour)
For each day/hour:
- **Base & always‑on**: `base_load_kw + server_room_kw + lab_always_on_kw`.
- **Exterior lighting**:
  - If `use_ghi_for_exterior=True`: ON when `GHI < night_ghi_threshold`.
  - Else: ON in `[ext_light_on_hour .. 24)` and `[0 .. ext_light_off_hour)`.
- **Plug loads**: `plug_kw_day` during staff hours on open days; `plug_kw_night` otherwise.
- **Interior lighting** (only open days):
  - State precedence **FULL > PARTIAL > CLEAN > OFF** using hour windows:
    - `full_light_hours` → `interior_light_full_kw`
    - `partial_light_hours` → `interior_light_partial_kw`
    - `clean_light_hours` → `interior_light_clean_kw`
  - **Daylight dimming** (if GHI available & `use_daylight_controls=True`):
    - If `GHI ≥ daylight_ghi_off` → **0 kW** (lights off)
    - If `GHI ≤ daylight_ghi_dim` → **100%** of the state capacity
    - Else linearly interpolate between 100%→0%
- **HVAC (cooling)**:
  - Only on open days **and** `hvac_on_start ≤ hour < hvac_on_end`.
  - If `T_out > cool_setpoint_C`:
    - `frac = (T_out − setpoint) / design_temp_diffC` (clamped to [0,1])
    - HVAC kW = `HVAC_capacity_kw * frac`
  - Otherwise **0** (no cooling).  
  - (Optional) **weekend_min_cooling** adds some cooling on closed days in hottest hours.
- **Night/weekend fans** (when AC is **off**, to avoid double counting):
  - If `night_fan_kw > 0` and AC isn’t running:
    - On **closed** days: if `hour in night_fan_hours_closed` → add `night_fan_kw`
    - On **open** days: if `hour in night_fan_hours_open` → add `night_fan_kw`
  - Tip: set `night_min_vent_kw = 0.0` when using `night_fan_kw` (legacy fixed minimal vent).
- **Kitchen** (open days only):
  - `kitchen_lunch_hours` → add `kitchen_peak_kw`
  - `kitchen_prep_cleanup` → add `kitchen_peak_kw * kitchen_prep_frac`
  - `kitchen_breakfast_hours` → add `kitchen_peak_kw * kitchen_breakfast_frac`
- **PV netting** (optional):
  - If `PV_use_radiation=True` & GHI available:
    - PV kW = `PV_capacity_kw * PV_performance_ratio * clamp(GHI/1000, 0..1)`
  - Else fallback sine between 06:00–18:00.
  - Net load = `max(0, load − PV)`

### 3) Bill calibration (energy scaling)
- Compute unscaled total: `sum(load_kW_hourly)` over the date range.
- **Scale factor** = `target_total_kwh / unscaled_total` (if `target_total_kwh` is provided).
- Multiply **every hour** by this scale factor to match billed energy exactly.
- Note: this **preserves shape**; if you must match a **peak kW** too, adjust inputs (e.g., `HVAC_capacity_kw` for hottest day) and rerun.

---

## Inputs & What to Edit

All inputs are **at the top** of `school_load_model.py`. Common edits:

### Site / Period / Billing
- `latitude`, `longitude`, `timezone_str`
- `start_date`, `end_date` (YYYY‑MM‑DD, inclusive; must be **past**)
- `target_total_kwh` (set to **None** to skip scaling)
- `WEATHER_SOURCE = "open-meteo"` or `"meteostat"`
- `holiday_dates = ["YYYY-MM-DD", ...]`

### Schedules
- `days_open_per_week = 5`  (Mon–Fri open)
- Hours (24h clock):
  - `school_start`, `school_end`
  - `staff_start`, `staff_end`
  - `cleaning_end`

### Lighting
- Interior capacities (kW) per state:
  - `interior_light_full_kw`, `interior_light_partial_kw`, `interior_light_clean_kw`
- Hour windows:
  - `full_light_hours`, `partial_light_hours`, `clean_light_hours`
- Daylighting:
  - `use_daylight_controls = True/False`
  - `daylight_ghi_dim`, `daylight_ghi_off` (W/m² thresholds)
- Exterior:
  - `exterior_light_kw`
  - `use_ghi_for_exterior = True/False`
  - If False: `ext_light_on_hour`, `ext_light_off_hour`
  - If True: `night_ghi_threshold`

### Plug / Always‑on
- `plug_kw_day`, `plug_kw_night`
- `base_load_kw`, optional `server_room_kw`, `lab_always_on_kw`

### HVAC / Fans
- `cool_setpoint_C`, `HVAC_capacity_kw`, `design_temp_diffC`
- `hvac_on_start`, `hvac_on_end`
- **Fans when AC is off**:
  - `night_fan_kw` (kW), `night_fan_hours_open`, `night_fan_hours_closed`
  - Keep `night_min_vent_kw=0.0` when using night_fan_kw (avoid double counting)
- Optional weekend cooling: `weekend_min_cooling`, `weekend_cooling_fraction`

### Kitchen
- Set `kitchen_peak_kw = 0.0` to **disable**.
- Otherwise configure `kitchen_*_hours` and fractions.

### PV
- `PV_on = True/False`
- `PV_capacity_kw`, `PV_performance_ratio`, `PV_use_radiation`

### Outputs
- `csv_filename_hourly`, `csv_filename_daily`, `plot_filename`, `make_heatmap`

---

## Outputs

1) **Hourly CSV** — `school_load_profile.csv`  
   - **Rows**: Hour `0..23`  
   - **Columns**: `Day1_YYYY-MM-DD` … `DayN_YYYY-MM-DD`  
   - Values: **kW** each hour (already scaled if `target_total_kwh` is set)

2) **Daily summary CSV** — `school_daily_summary.csv`  
   - Columns: `date`, `daily_kWh`, `daily_peak_kW`  
   - Useful for quick QA or comparing day‑level totals.

3) **Overlay plot** — `school_overlay.png`  
   - **Blue gradient**: Day 1 (light) → Day N (dark)  
   - **Dashed lines** = closed/holiday days  
   - Legend explains line style; thin horizontal colorbar shows day index.

---

## Validation & Comparison

- Compare **daily kWh** and **peak kW** with interval data (if available).
- For model‑to‑sim comparisons (e.g., EnergyPlus/eQUEST):
  - Use the **same weather period** and **schedules**.
  - Align **PV capacity** and derates.
  - Expect similar **shape**; exact peaks depend on HVAC sizing and internal gains.

---

## Troubleshooting

- **“Axes not compatible with tight_layout” warning**  
  Caused by inset colorbars. The script uses manual margins instead of `tight_layout` to avoid this.

- **Colorbar/legend overlapping**  
  The script positions the legend inside and the colorbar under the plot. Adjust `bbox_to_anchor` or `fig.subplots_adjust(...)` if you change layout.

- **Matplotlib colormap API differences**  
  Handled via a helper (`get_resampled_cmap`) to support older/newer versions.

- **Weather gaps**  
  The day normalization interpolates within a day and sets night irradiance to 0. If a whole day is missing (rare), it falls back to mean values.

- **Peak doesn’t match bill’s demand charge**  
  Increase `HVAC_capacity_kw` (or shift schedules) on the hottest weekday(s) until the modeled peak aligns.

---

## Suggested Repo Structure

```
school-load-model/
├─ README.md
├─ school_load_model.py
├─ requirements.txt
├─ examples/
│  └─ example_config_notes.md
└─ outputs/
   ├─ school_load_profile.csv
   ├─ school_daily_summary.csv
   └─ school_overlay.png
```

`requirements.txt`
```
requests
pandas
matplotlib
meteostat
pytz
```

---



# * Author: Purui Yi (Perry), Claude Code
# * Update date: 2026-09-27
# * Credit: Data.gov, OpenEI, NREL, USURDB


import pandas as pd
import numpy as np
import json
from collections import Counter
from pathlib import Path

### This is a script to clean and process the utility rates data from the US Utility Rate Database (USURDB) for commercial sector analysis.
### It reads the data, filters for approved and effective rates, calculates coverage statistics, and saves the current commercial rates to a CSV file. 
### Additionally, it includes functions to estimate monthly utility usage based on charger type and to calculate average energy prices based on rate structures.

### To run this script, ensure that the required CSV files ("usurdb.json.gz", "iou_zipcodes_2024.csv", and "non_iou_zipcodes_2024.csv") are present in the Input folder next to this script.
### Outputs are written to the Output folder next to this script.
### usurdb.json.gz can be found on https://openei.org/apps/USURDB/download/usurdb.json.gz or from https://openei.org/wiki/Utility_Rate_Database
### iou_zipcodes_2024.csv can be found on https://data.openei.org/files/8563/iou_zipcodes_2024.csv
### non_iou_zipcodes_2024.csv can be found on https://data.openei.org/files/8563/non_iou_zipcodes_2024.csv
### or more generally on https://catalog.data.gov/dataset/u-s-electric-utility-companies-and-rates-look-up-by-zip-code-2024


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
OUTPUT_DIR.mkdir(exist_ok=True)

rates = pd.read_json(INPUT_DIR / "usurdb.json.gz", lines=False)
TODAY = pd.Timestamp.now(tz="UTC")
MONTH_DAYS = 30  

CHARGER_CONFIG = {
    'Level 1': {'name': 'Level 1', 'kw': 20},
    'Level 2': {'name': 'Level 2', 'kw': 50},
    'Level 3': {'name': 'Level 3', 'kw': 120},
    'DCFC': {'name': 'DCFC', 'kw': 350}
}

def get_utility_usage(charger_type: str, n_chargers: int, working_hours: float) -> tuple:
    """Estimate the monthly utility usage in kWh for a given charging site.
    returns: tuple: (monthly_kwh, peak_kw)
    """
    config = CHARGER_CONFIG.get(charger_type)
    if config:
        return (config['kw'] * n_chargers * working_hours * MONTH_DAYS, config['kw'] * n_chargers)
    else:
        raise ValueError(f"Unknown charger type: {charger_type}")
# print(rates.shape)
print(rates.columns.tolist())


# Utility pricing for the whole US, approved and as effective today, commercial sector only
rates["effectiveDate"] = pd.to_datetime(rates["effectiveDate"].str.get('$date'), errors="coerce", utc=True)
rates["endDate"] = pd.to_datetime(rates["endDate"].str.get('$date'), errors="coerce", utc=True)
rates = rates[rates['approved'] == True]    # only keep approved rates
current = rates[(rates["effectiveDate"] <= TODAY) & (rates["endDate"].isna() | (rates["endDate"] > TODAY))]
current = (current.sort_values("effectiveDate")
                  .drop_duplicates(subset=["eiaId", "rateName"], keep="last"))
current = current[current["sector"] == "Commercial"]
current = current[current['serviceType'] == 'Bundled']


# Tradeoff: price recency vs geographic coverage.
# Go back one month at a time from today and report how much of the US the rates effective since then cover;
# stop once at least 69% of zip codes are covered. Zip -> utility (eiaid) -> state comes from NREL's 2024 zip lookup.
MIN_ZIP_COVERAGE = 0.69
zips = pd.concat([pd.read_csv(INPUT_DIR / "iou_zipcodes_2024.csv"), pd.read_csv(INPUT_DIR / "non_iou_zipcodes_2024.csv")])
zips = zips.drop_duplicates(["zip", "eiaid"])
current["eid"] = pd.to_numeric(current["eiaId"], errors="coerce")
zips = zips[zips["eiaid"].isin(current["eid"].dropna())]   # only zips served by a utility with a Bundled rate
eff_month = current["effectiveDate"].dt.tz_localize(None).dt.to_period("M")

rows = []
month = pd.Timestamp(TODAY.date()).to_period("M")
while True:
    window = current[eff_month >= month]
    served = zips[zips["eiaid"].isin(window["eid"].dropna())]
    rows.append({
        "Effective from": month.strftime("%b %Y"),
        "Rates": len(window),
        "Utilities": window["eid"].nunique(),
        "States touched": served["state"].nunique(),
        "Zip coverage": served["zip"].nunique() / zips["zip"].nunique(),
    })
    if rows[-1]["Zip coverage"] >= MIN_ZIP_COVERAGE or month < eff_month.min():
        break
    month -= 1

coverage = pd.DataFrame(rows)
print(coverage.to_string(index=False, formatters={"Zip coverage": "{:.1%}".format}))
current = current[eff_month >= month]


# Save cleaned as csv
# out1 = current.copy()
# out1["energyRateStrux"] = out1["energyRateStrux"].apply(
#     lambda v: json.dumps(v, default=str) if isinstance(v, list) else v
# )
# try:
#     out1.to_csv(OUTPUT_DIR / "current_commercial_rates.csv", index=False)
# except Exception as e:  
#     print(f"Error occurred while saving CSV: {e}")

# print(len(rates), "total rates ->", len(current), "current commercial rates")
# print(current[["eiaId", "utilityName", "rateName", "effectiveDate", "endDate"]].head(10))


# Energy bill calculation functions
def _normalize_rate_structure(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def energy_bill(strux, period, kwh):
    strux = _normalize_rate_structure(strux)
    if not isinstance(strux, (list, tuple)) or not strux:
        return 0.0

    period_data = strux[period]
    total, prev = 0.0, 0.0
    # Calculate tiered energy charges
    for t in period_data.get("energyRateTiers", []):
        cap = t.get("max", float("inf"))
        total += max(0, min(kwh, cap) - prev) * (t.get("rate", 0) + t.get("adj", 0))
        if kwh <= cap:
            break
        prev = cap
    return total

# Demand charge bill calculation functions
def _flatten_months(months):
    """Coerce flatDemandMonths into a flat list of month identifiers."""
    if months is None:
        return []
    if isinstance(months, (list, tuple, set)):
        flattened = []
        for item in months:
            if isinstance(item, (list, tuple, set)):
                flattened.extend(list(item))
            else:
                flattened.append(item)
        return flattened
    return [months]


def flat_demand_rate_for_kw(flat_demand_strux, peak_kw):
    """Return the applicable flat-demand rate based on the peak kW threshold."""
    if not flat_demand_strux or peak_kw <= 0:
        return 0.0

    strux = _normalize_rate_structure(flat_demand_strux)
    if not isinstance(strux, (list, tuple)):
        return 0.0

    for period in strux:
        if not isinstance(period, dict):
            continue

        tiers = period.get("flatDemandTiers", [])
        for tier in tiers:
            max_kw = tier.get("max")
            rate = float(tier.get("rate", 0) or 0) + float(tier.get("adj", 0) or 0)

            if max_kw is None:
                return rate
            if peak_kw <= float(max_kw):
                return rate

    return 0.0


def flat_demand_bill(flat_demand_strux, flat_demand_months, peak_kw):
    """Calculate the flat demand charge as peak kW times the applicable demand rate."""
    demand_rate = flat_demand_rate_for_kw(flat_demand_strux, peak_kw)
    return peak_kw * demand_rate if demand_rate > 0 else 0.0


def first_meter_fixed_charge_monthly(row):
    """Convert the first-meter fixed charge to an average monthly amount."""
    charge = row.get("fixedChargeFirstMeter", 0)
    charge = float(charge) if pd.notna(charge) else 0.0
    units = str(row.get("fixedChargeUnits", "$/month")).strip().lower()
    if "day" in units:
        return charge * 365 / 12
    return charge


# Energy, demand, and first-meter charges by selected period
def avg_price(strux, row, kwh, energyWeekdaySched, peak_kw):
    """Return average prices in dollars per kWh as a tuple (total, energy, demand, meter).

    ``kwh`` is representative monthly consumption.
    """
    strux = _normalize_rate_structure(strux)
    if isinstance(energyWeekdaySched, str):
        try:
            energyWeekdaySched = json.loads(energyWeekdaySched)
        except (TypeError, ValueError):
            energyWeekdaySched = []
    if not isinstance(energyWeekdaySched, (list, tuple)) or not energyWeekdaySched:
        energyWeekdaySched = [[0]] * 12   # no schedule: use energy period 0 every month

    months = [Counter(m).most_common(1)[0][0] for m in energyWeekdaySched if m]
    energy_bills = [energy_bill(strux, p, kwh) for p in months]
    annual_energy_charge = sum(energy_bills) * (12 / len(energy_bills)) if energy_bills else 0.0
    annual_meter_charge = first_meter_fixed_charge_monthly(row) * 12

    flat_demand = flat_demand_bill(
        row.get("flatDemandStrux"),
        row.get("flatDemandMonths"),
        peak_kw or 0.0,
    )
    demand_months = _flatten_months(row.get("flatDemandMonths"))
    demand_month_count = len(demand_months) if demand_months else 12
    annual_demand_charge = flat_demand * demand_month_count

    annual_total = annual_energy_charge + annual_demand_charge + annual_meter_charge
    annual_kwh = 12 * kwh
    if not annual_kwh:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        annual_total / annual_kwh,
        annual_energy_charge / annual_kwh,
        annual_demand_charge / annual_kwh,
        annual_meter_charge / annual_kwh,
    )



kwh, peak_kw = get_utility_usage(charger_type='Level 3', n_chargers=2, working_hours=10)


# save a copy with bill and average price calculations
out2 = current.copy()
out2["energyRateStrux"] = out2["energyRateStrux"].map(_normalize_rate_structure)
out2["energyWeekdaySched"] = out2["energyWeekdaySched"].map(
    lambda value: json.loads(value) if isinstance(value, str) else value
)

out2["est_monthly_bill"] = out2.apply(
    lambda row: energy_bill(row["energyRateStrux"], 0, kwh)
    + flat_demand_bill(row.get("flatDemandStrux"), row.get("flatDemandMonths"), peak_kw),
    axis=1,
)

out2[["avg_price", "avg_energy_price", "avg_demand_price", "avg_meter_price"]] = out2.apply(
    lambda row: avg_price(row["energyRateStrux"], row, kwh, row["energyWeekdaySched"], peak_kw),
    axis=1,
    result_type="expand",
)

try:
    out2.to_csv(OUTPUT_DIR / "current_utility_rates.csv", index=False)
except Exception as e:  
    print(f"Error occurred while saving CSV: {e}")

print(kwh)
print(peak_kw)
print(out2[["utilityName", "rateName", "est_monthly_bill", "avg_price", "avg_energy_price", "avg_demand_price", "avg_meter_price"]].head())


# Save utility rates by zip code
# NREL's 2024 lookup gives every zip's utility, ownership, and EIA average commercial price (comm_rate).
zip_codes = pd.concat(
    [
        pd.read_csv(INPUT_DIR / "iou_zipcodes_2024.csv"),
        pd.read_csv(INPUT_DIR / "non_iou_zipcodes_2024.csv"),
    ],
    ignore_index=True,
)
zip_codes = zip_codes[zip_codes["service_type"] == "Bundled"]
zip_codes = zip_codes[["zip", "state", "eiaid", "utility_name", "ownership", "comm_rate"]].drop_duplicates(["zip", "eiaid"])
zip_codes["eiaid"] = pd.to_numeric(zip_codes["eiaid"], errors="coerce")
zip_codes["comm_rate"] = zip_codes["comm_rate"].where(zip_codes["comm_rate"] > 0)   # 0 means no EIA data

rate_columns = out2[
    [
        "eid",
        "eiaId",
        "utilityName",
        "rateName",
        "effectiveDate",
        "avg_energy_price",
        "avg_demand_price",
        "avg_meter_price",
        "avg_price",
    ]
].copy()
rate_columns["eid"] = pd.to_numeric(rate_columns["eid"], errors="coerce")
rate_columns["pricingYear"] = pd.to_datetime(rate_columns["effectiveDate"], utc=True).dt.year

zip_rate_output = zip_codes.merge(
    rate_columns,
    left_on="eiaid",
    right_on="eid",
    how="inner",
)
zip_rate_output = zip_rate_output.rename(
    columns={
        "zip": "zipCode",
        "avg_energy_price": "energyRate",
        "avg_demand_price": "demandRate",
        "avg_meter_price": "fixedRate",
        "avg_price": "totalRate",
    }
)[
    [
        "eiaId",
        "zipCode",
        "state",
        "ownership",
        "utilityName",
        "rateName",
        "comm_rate",
        "energyRate",
        "demandRate",
        "fixedRate",
        "totalRate",
        "pricingYear",
    ]
]
price_columns = ["comm_rate", "energyRate", "demandRate", "fixedRate", "totalRate"]
zip_rate_output[price_columns] = zip_rate_output[price_columns].apply(
    pd.to_numeric, errors="coerce"
).astype(float)
zip_rate_output["source"] = "USURDB"

# Zips with no USURDB rate: fill totalRate with the utility's EIA average commercial rate (comm_rate).
covered_pairs = pd.MultiIndex.from_arrays([zip_rate_output["zipCode"], pd.to_numeric(zip_rate_output["eiaId"], errors="coerce")])
covered = pd.MultiIndex.from_frame(zip_codes[["zip", "eiaid"]]).isin(covered_pairs)
estimated = (zip_codes[~covered & zip_codes["comm_rate"].notna()]
             .rename(columns={"zip": "zipCode", "eiaid": "eiaId", "utility_name": "utilityName"}))
estimated["totalRate"] = estimated["comm_rate"]
estimated["source"] = "EIA estimate"
estimated["pricingYear"] = 2024   # NREL's 2024 zip lookup
newly_filled = set(estimated["zipCode"]) - set(zip_rate_output["zipCode"])
print(f"{len(newly_filled)} zips with no USURDB rate filled with EIA comm_rate; "
      f"{estimated['zipCode'].nunique() - len(newly_filled)} more zips got comm_rate for an extra utility "
      f"({len(estimated)} zip-utility rows in total)")

zip_rate_output = pd.concat([zip_rate_output, estimated], ignore_index=True)[zip_rate_output.columns]
zip_rate_output["pricingYear"] = zip_rate_output["pricingYear"].astype("Int64")
print(zip_rate_output["source"].value_counts().to_string())
zip_rate_output.to_csv(OUTPUT_DIR / "utility_rates_by_zip.csv", index=False, float_format="%.10f")

# TODO: Add Confidnece level and Interval for each zip code's totalRate based on the number of rates available for that zip code and the variability of those rates. 




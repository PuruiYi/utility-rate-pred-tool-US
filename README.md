# US Commercial Electricity Rate Estimator (KNN)

Estimates the commercial electricity price ($/kWh) for an EV charging site anywhere in the US. Click a point on a
map and a k-nearest-neighbor (KNN) regressor predicts the rate there from nearby zip codes with known prices.

Known prices come from two public sources:

| Source (as shown in the app) | What it provides | Link |
|---|---|---|
| **OpenEI Database** | Tariff-level commercial rate schedules from the US Utility Rate Database (USURDB), priced for the modeled site | [openei.org/wiki/Utility_Rate_Database](https://openei.org/wiki/Utility_Rate_Database) |
| **EIA Estimation** | Each utility's average commercial price (`comm_rate`, from U.S. Energy Information Administration data via NREL's zip lookup), used where USURDB has no rate | [catalog.data.gov: rates look-up by zip code (2024)](https://catalog.data.gov/dataset/u-s-electric-utility-companies-and-rates-look-up-by-zip-code-2024) |

## Project layout

```
Jule/
├── CommercialRateAnalysis.py   # Step 1: build per-zip rates  -> Output/
├── InteractiveViz.py           # Step 2: Streamlit map + KNN regressor
├── Input/                      # Source data (download, see below)
│   ├── usurdb.json.gz
│   ├── iou_zipcodes_2024.csv
│   ├── non_iou_zipcodes_2024.csv
│   └── 2020_Gaz_zcta_national.txt
├── Output/                     # Generated
│   ├── current_utility_rates.csv
│   └── utility_rates_by_zip.csv
└── archive/                    # Earlier scripts and outputs, not used by the pipeline
```

Paths are resolved relative to the scripts, so both can be run from any working directory.

## Data sources

Put these files in `Input/`:

| File | Source |
|---|---|
| `usurdb.json.gz` | [OpenEI USURDB download](https://openei.org/apps/USURDB/download/usurdb.json.gz) |
| `iou_zipcodes_2024.csv` | [data.openei.org/files/8563/iou_zipcodes_2024.csv](https://data.openei.org/files/8563/iou_zipcodes_2024.csv) |
| `non_iou_zipcodes_2024.csv` | [data.openei.org/files/8563/non_iou_zipcodes_2024.csv](https://data.openei.org/files/8563/non_iou_zipcodes_2024.csv) |
| `2020_Gaz_zcta_national.txt` | [Census ZCTA Gazetteer (zip)](https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_zcta_national.zip), unzipped; gives each zip code's center point |

## Setup

Tested with Python 3.8.2.

```bash
pip install pandas numpy scikit-learn streamlit folium streamlit-folium "xyzservices<2025"
```

`xyzservices` 2025+ (installed by folium) requires Python 3.9+. Pin it as above on Python 3.8.

## Usage

```bash
python CommercialRateAnalysis.py     # a few minutes; writes Output/utility_rates_by_zip.csv
streamlit run InteractiveViz.py      # opens the map in a browser
```

## How it works

### 1. Rate pipeline (`CommercialRateAnalysis.py`)

1. **Filter USURDB** to rates that are approved, in effect today, commercial sector, and `serviceType == "Bundled"`
   (energy and delivery priced together). Keep the latest version of each utility + rate name.
2. **Pick a recency cutoff.** Step back one month at a time until the rates cover at least 69% of the zip codes
   served by a utility with a Bundled rate (`MIN_ZIP_COVERAGE`).
3. **Price each rate for a modeled site**, by default 2 × Level 3 chargers (120 kW each) running 10 h/day:
   240 kW peak demand and 72,000 kWh/month. Change this in the `get_utility_usage(...)` call.
   `avg_price` returns $/kWh for the whole year, split into energy, demand (flat demand charges, with
   adjustments), and fixed (meter) charges.
4. **Map rates to zip codes** with NREL's 2024 zip → utility lookup (Bundled rows only).
5. **Fill the gaps.** Zip + utility pairs with no USURDB rate get `totalRate = comm_rate` from the same lookup,
   marked `source = "EIA estimate"` and `pricingYear = 2024`.

**`Output/utility_rates_by_zip.csv`** (one row per zip × utility × rate schedule):

| Column | Meaning |
|---|---|
| `eiaId`, `utilityName`, `ownership`, `state`, `zipCode` | Utility and location |
| `rateName` | USURDB rate schedule (empty for EIA rows) |
| `comm_rate` | EIA average commercial $/kWh for the utility |
| `energyRate`, `demandRate`, `fixedRate` | $/kWh parts of the modeled site's bill (USURDB rows only) |
| `totalRate` | Total $/kWh: sum of the three above (USURDB) or `comm_rate` (EIA) |
| `pricingYear` | Year of the rate's `effectiveDate` (USURDB), or 2024 (EIA) |
| `source` | `USURDB` or `EIA estimate` |

### 2. Interactive map (`InteractiveViz.py`)

- Collapses the CSV to one value per zip: the **median `totalRate`** across that zip's rate schedules, with
  outliers above a chosen quantile clipped.
- Places each zip at its Census ZCTA center point and fits
  `KNeighborsRegressor(metric="haversine", algorithm="ball_tree")` on latitude/longitude.
- **Click the map** to see:
  - the predicted $/kWh
  - the nearest zip and the utilities serving it
  - that zip's known rate, if any
  - the k neighbors used, with utility, rate, year, distance, number of rate schedules (`nRates`), and source

**Sidebar settings:**

| Setting | Default |
|---|---|
| k | 5 (range 1–30) |
| Neighbor weighting | distance or uniform |
| Only use neighbors in the same state | on |
| Outlier clip quantile | 0.99 |
| Show all zips with known rates | off (slower when on) |

**Accuracy** is shown as a 5-fold `GroupKFold` mean absolute error that **holds out whole utilities**, next to a
national-median baseline. A plain random split would score each zip against other zips of the same utility, which
share its price, and make the error look near zero.

## Limitations

- **Prices change at utility boundaries, not smoothly across distance.** Neighbors from a different utility can be
  far off. When only USURDB rates were used, location-based KNN did *not* beat the national median under the
  utility-held-out test. Including EIA rows lowers the reported error, partly because their values are uniform
  within each utility.
- **EIA rows are utility-wide averages over all commercial customers**, not the price for this specific
  site. The `source` column keeps them separate from modeled USURDB prices.
- **Rate schedule choice.** A zip's value is the median over *all* of its utility's current Bundled commercial
  schedules. That includes schedules the modeled site might not qualify for, and the operator would likely pick the
  cheapest eligible one.
- **Coverage.** About 41% of zip codes have a USURDB-based rate. The rest rely on EIA averages.
- **Same-state filter.** The "same state" option applies to map clicks but not to the cross-validation score.

## Credits

Data: [OpenEI / USURDB](https://openei.org/wiki/Utility_Rate_Database),
[NREL & EIA via data.gov](https://catalog.data.gov/dataset/u-s-electric-utility-companies-and-rates-look-up-by-zip-code-2024),
[U.S. Census Bureau ZCTA Gazetteer](https://www.census.gov/geographies/reference-files/time-series/geo/gazetteer-files.html).

Author: Purui Yi (Perry), with Claude Code.

# * Author: Purui Yi (Perry), Claude Code
# * Update date: 2026-09-27
# * Credit: Census Bureau (ZCTA Gazetteer), OpenEI, NREL, USURDB

from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from streamlit_folium import st_folium

### Click anywhere on the map to predict the commercial electricity rate ($/kWh) there with a
### k-nearest-neighbor regressor fit on zip codes that have a known Bundled commercial rate.
###
### Run with:  streamlit run InteractiveViz.py
### Requires Output/utility_rates_by_zip.csv (from CommercialRateAnalysis.py) and, in Input/, the NREL zip lookups
### ("iou_zipcodes_2024.csv", "non_iou_zipcodes_2024.csv"), and the Census ZCTA Gazetteer
### "2020_Gaz_zcta_national.txt" from
### https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2020_Gazetteer/2020_Gaz_zcta_national.zip

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
EARTH_RADIUS_MI = 3958.8
RATE_COL = "totalRate"
# Display name and link for each value of the CSV's "source" column
SOURCES = {
    "USURDB": ("OpenEI Database", "https://openei.org/wiki/Utility_Rate_Database"),
    "EIA estimate": ("EIA Estimation",
                     "https://catalog.data.gov/dataset/u-s-electric-utility-companies-and-rates-look-up-by-zip-code-2024"),
}


@st.cache_data
def load_zip_points():
    """Every zip code with a centroid, plus its state and utility from the NREL lookup."""
    gaz = pd.read_csv(INPUT_DIR / "2020_Gaz_zcta_national.txt", sep="\t", dtype={"GEOID": str})
    gaz.columns = gaz.columns.str.strip()
    gaz = gaz.rename(columns={"GEOID": "zip", "INTPTLAT": "lat", "INTPTLONG": "lon"})[["zip", "lat", "lon"]]

    lookup = pd.concat([pd.read_csv(INPUT_DIR / "iou_zipcodes_2024.csv"), pd.read_csv(INPUT_DIR / "non_iou_zipcodes_2024.csv")])
    lookup["zip"] = lookup["zip"].astype(str).str.zfill(5)
    lookup = (lookup.groupby("zip")
                    .agg(state=("state", "first"),
                         servingUtilities=("utility_name", lambda s: ", ".join(sorted(set(s)))))
                    .reset_index())
    return gaz.merge(lookup, on="zip", how="left")


@st.cache_data
def load_known_rates(clip_quantile):
    """One row per zip: median total rate over the zip's commercial rates, outliers clipped."""
    rates = pd.read_csv(OUTPUT_DIR / "utility_rates_by_zip.csv", dtype={"zipCode": str})
    rates["zip"] = rates["zipCode"].str.zfill(5)
    rates["source"] = rates["source"].replace({key: name for key, (name, _) in SOURCES.items()})
    rates[RATE_COL] = rates[RATE_COL].clip(upper=rates[RATE_COL].quantile(clip_quantile))

    per_zip = (rates.groupby("zip")
                    .agg(**{RATE_COL: (RATE_COL, "median")},
                         nRates=("rateName", "count"),
                         eiaId=("eiaId", "first"),
                         utilityName=("utilityName", "first"),
                         pricingYear=("pricingYear", _distinct),
                         source=("source", _distinct))
                    .reset_index())
    return per_zip.merge(load_zip_points(), on="zip", how="inner")


def _distinct(values):
    """Distinct values of a zip's rates as text, e.g. '2025, 2026' or 'EIA Estimation, OpenEI Database'."""
    values = values.dropna()
    values = values.astype(int) if pd.api.types.is_numeric_dtype(values) else values
    return ", ".join(str(v) for v in sorted(set(values)))


def fit_knn(known, k, weights):
    model = KNeighborsRegressor(n_neighbors=min(k, len(known)), weights=weights,
                                metric="haversine", algorithm="ball_tree")
    model.fit(np.radians(known[["lat", "lon"]].to_numpy()), known[RATE_COL].to_numpy())
    return model


@st.cache_data
def grouped_cv(clip_quantile, k, weights):
    """Hold out whole utilities, so a zip is never predicted from other zips of its own utility."""
    known = load_known_rates(clip_quantile)
    X = np.radians(known[["lat", "lon"]].to_numpy())
    y = known[RATE_COL].to_numpy()
    errors, baseline = [], []
    for train, test in GroupKFold(n_splits=5).split(X, y, groups=known["eiaId"]):
        model = KNeighborsRegressor(n_neighbors=k, weights=weights, metric="haversine", algorithm="ball_tree")
        model.fit(X[train], y[train])
        errors.append(np.abs(model.predict(X[test]) - y[test]))
        baseline.append(np.abs(np.median(y[train]) - y[test]))
    return float(np.concatenate(errors).mean()), float(np.concatenate(baseline).mean())


def nearest_zip(points, lat, lon):
    """Closest zip centroid to a clicked point (haversine), as a Series."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    lat2, lon2 = np.radians(points["lat"].to_numpy()), np.radians(points["lon"].to_numpy())
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return points.iloc[int(np.argmin(a))]


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="Commercial Rate KNN", layout="wide")
st.title("Commercial electricity rate: k-nearest-neighbor estimate")

with st.sidebar:
    k = st.slider("k (neighbors)", 1, 30, 5)
    weights = st.radio("Neighbor weighting", ["distance", "uniform"], horizontal=True)
    same_state = st.checkbox("Only use neighbors in the same state", value=True)
    clip_quantile = st.slider("Clip rates above quantile", 0.90, 1.00, 0.99, 0.01,
                              help="Caps outlier rates before fitting.")
    show_known = st.checkbox("Show all zips with known rates", value=False,
                             help="Draws ~15k points; makes the map slower.")

known = load_known_rates(clip_quantile)
points = load_zip_points()

with st.sidebar:
    st.divider()
    mae, base_mae = grouped_cv(clip_quantile, k, weights)
    st.metric("Grouped CV mean abs. error", f"${mae:.4f}/kWh",
              delta=f"{mae - base_mae:+.4f} vs. national median", delta_color="inverse")
    st.caption(f"5-fold, holding out whole utilities. {len(known):,} zips with known rates "
               f"from {known['eiaId'].nunique()} utilities.")

if "click" not in st.session_state:
    st.session_state.click = None
if "view" not in st.session_state:
    st.session_state.view = {"center": [39.5, -98.35], "zoom": 4}

m = folium.Map(location=st.session_state.view["center"], zoom_start=st.session_state.view["zoom"],
               tiles="OpenStreetMap")

if show_known:
    lo, hi = known[RATE_COL].quantile([0.05, 0.95])
    cmap = folium.LinearColormap(["#2c7bb6", "#ffffbf", "#d7191c"], vmin=lo, vmax=hi,
                                 caption="Total rate ($/kWh)")
    for r in known.itertuples():
        folium.CircleMarker([r.lat, r.lon], radius=2, weight=0, fill=True, fill_opacity=0.7,
                            fill_color=cmap(min(max(getattr(r, RATE_COL), lo), hi))).add_to(m)
    cmap.add_to(m)

result = None
if st.session_state.click:
    lat, lon = st.session_state.click
    here = nearest_zip(points, lat, lon)
    pool = known[known["state"] == here["state"]] if same_state and pd.notna(here["state"]) else known
    if pool.empty:
        pool = known

    model = fit_knn(pool, k, weights)
    query = np.radians([[lat, lon]])
    dist, idx = model.kneighbors(query)
    neighbors = pool.iloc[idx[0]].assign(distance_mi=dist[0] * EARTH_RADIUS_MI)
    result = (here, float(model.predict(query)[0]), neighbors)

    folium.Marker([lat, lon], tooltip=f"Prediction: ${result[1]:.4f}/kWh",
                  icon=folium.Icon(color="red", icon="flash")).add_to(m)
    for r in neighbors.itertuples():
        folium.PolyLine([[lat, lon], [r.lat, r.lon]], weight=1, color="#555", dash_array="4").add_to(m)
        folium.CircleMarker([r.lat, r.lon], radius=6, color="#1f4e79", fill=True, fill_opacity=0.9,
                            tooltip=f"{r.zip} · {r.utilityName} · ${getattr(r, RATE_COL):.4f}/kWh "
                                    f"· {r.pricingYear} · {r.source} · {r.distance_mi:.1f} mi").add_to(m)

map_col, info_col = st.columns([3, 2])
with map_col:
    out = st_folium(m, height=620, use_container_width=True, key="map",
                    returned_objects=["last_clicked", "center", "zoom"])

if out and out.get("last_clicked"):
    new_click = (out["last_clicked"]["lat"], out["last_clicked"]["lng"])
    if new_click != st.session_state.click:
        st.session_state.click = new_click
        if out.get("center"):
            st.session_state.view = {"center": [out["center"]["lat"], out["center"]["lng"]],
                                     "zoom": out.get("zoom", 4)}
        st.rerun()

with info_col:
    if result is None:
        st.info("Click anywhere on the map to estimate the rate at that point.")
    else:
        here, pred, neighbors = result
        st.metric("Estimated total rate", f"${pred:.4f}/kWh")
        st.write(f"**Nearest zip:** {here['zip']} ({here['state'] if pd.notna(here['state']) else 'no state'})  \n"
                 f"**Served by:** {here['servingUtilities'] if pd.notna(here['servingUtilities']) else 'unknown'}")
        actual = known.loc[known["zip"] == here["zip"], RATE_COL]
        if not actual.empty:
            st.write(f"**Known rate in this zip:** ${actual.iloc[0]:.4f}/kWh (median of its rates)")
        else:
            st.write("**No known rate in this zip.** The estimate comes only from neighbors.")
        if neighbors["utilityName"].nunique() > 1:
            st.warning("Neighbors span more than one utility; prices can jump at utility boundaries.")
        st.dataframe(
            neighbors[["zip", "utilityName", RATE_COL, "pricingYear", "distance_mi", "nRates", "source"]]
            .rename(columns={RATE_COL: "$/kWh", "utilityName": "utility", "pricingYear": "year",
                             "distance_mi": "miles"})
            .style.format({"$/kWh": "{:.4f}", "miles": "{:.1f}"}),
            hide_index=True, use_container_width=True,
        )
        st.caption("Sources: " + " · ".join(f"[{name}]({url})" for name, url in SOURCES.values()))

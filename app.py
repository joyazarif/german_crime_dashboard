import numpy as np
import geopandas as gpd
import dash_bootstrap_components as dbc
import json
import re
from urllib.request import urlopen


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback_context
import logging
logger = logging.getLogger(__name__)

print("Lade Daten und initialisiere Dashboard...")


# Use a light Plotly template
pio.templates.default = "plotly_white"

# --------- POPULATION (for rates per 100k) ---------
POP_FILE = "data/population_states.csv"


def load_population():
    pop = pd.read_csv(POP_FILE, sep=";", encoding="utf-8")
    pop.columns = [c.strip() for c in pop.columns]
    pop["Bundesland"] = pop["Bundesland"].astype(str).str.strip()
    pop["Jahr"] = pop["Jahr"].astype(int)
    pop["Population"] = pop["Population"].astype(float)
    return pop


try:
    POP = load_population()
    print(" Population loaded:", POP.shape)
except Exception as e:
    POP = pd.DataFrame(columns=["Bundesland", "Jahr", "Population"])
    print("⚠️ Population NOT loaded:", e)


def get_population(bundesland: str, year: int):
    if POP.empty:
        return None
    sub = POP[(POP["Bundesland"] == bundesland) & (POP["Jahr"] == int(year))]
    if not sub.empty:
        return float(sub["Population"].iloc[0])
    return None


def per_100k(value, population):
    if population is None or population <= 0:
        return 0.0
    return 100000.0 * float(value) / float(population)


# --------- THEME COLORS ---------
HEADER_BG = "#0F1A2A"
HEADER_BORDER = "#1F2A3A"
HEADER_TEXT_MAIN = "#FFFFFF"
HEADER_TEXT_SUB = "#D0D6E2"

SIDEBAR_BG = "#eef2ff"
SIDEBAR_BORDER = "#c7d2fe"

# --------- LAYOUT STYLES ---------
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "130px",     # below header
    "left": 0,
    "bottom": 0,
    "width": "300px",
    "padding": "20px 15px",
    "backgroundColor": SIDEBAR_BG,
    "color": "#1e1b4b",
    "borderRight": f"1px solid {SIDEBAR_BORDER}",
    "overflowY": "auto",
    "overflowX": "hidden",
    "zIndex": 1,
    "boxShadow": "0px 4px 10px rgba(0,0,0,0.25)",
}

CONTENT_STYLE = {
    "marginLeft": "290px",
    "marginTop": "130px",
    "padding": "20px 20px 40px 20px",
    "backgroundColor": "#ffffff",
    "minHeight": "100vh",
}

CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "borderRadius": "14px",
    "padding": "12px 14px",
    "border": "1px solid #e5e7eb",
    "boxShadow": "0 6px 18px rgba(15, 23, 42, 0.08)",
    "color": "#0f172a",
    "minHeight": "78px",
    "display": "flex",
    "flexDirection": "column",
    "justifyContent": "center",
    "transition": "transform 120ms ease, box-shadow 120ms ease",
}

KPI_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
    "gap": "12px",
    "marginBottom": "14px",
}

KPI_VALUE_STYLE = {
    "fontSize": "24px",
    "fontWeight": "800",
    "letterSpacing": "-0.3px",
    "lineHeight": "1.05",
}
KPI_LABEL_STYLE = {
    "fontSize": "12px",
    "fontWeight": "600",
    "color": "#475569",
    "textTransform": "uppercase",
    "letterSpacing": "0.6px",
    "marginBottom": "6px",
}
STANDARD_HEIGHT = 750

# --------- DATA META ---------
STATE_MAP = {
    1: "Schleswig-Holstein",
    2: "Hamburg",
    3: "Niedersachsen",
    4: "Bremen",
    5: "Nordrhein-Westfalen",
    6: "Hessen",
    7: "Rheinland-Pfalz",
    8: "Baden-Württemberg",
    9: "Bayern",
    10: "Saarland",
    11: "Berlin",
    12: "Brandenburg",
    13: "Mecklenburg-Vorpommern",
    14: "Sachsen",
    15: "Sachsen-Anhalt",
    16: "Thüringen",
}

AGE_COLS = {
    "Kinder <14": "Opfer Kinder bis 14 Jahre- insgesamt",
    "Jugendliche 14–<18": "Opfer Jugendliche 14 bis unter 18 Jahre - insgesamt",
    "Heranwachsende 18–<21": "Opfer - Heranwachsende 18 bis unter 21 Jahre - insgesamt",
    "Erwachsene 21–<60": "Opfer Erwachsene 21 bis unter 60 Jahre - insgesamt",
    "Senior:innen 60+": "Opfer - Erwachsene 60 Jahre und aelter - insgesamt",
}

# --------- POPULATION SHARES BY AGE GROUP (Germany, approx.) ---------
AGE_POPULATION_SHARE = {
    "Kinder <14": 0.13,              # 13 %
    "Jugendliche 14–<18": 0.05,       # 5 %
    "Heranwachsende 18–<21": 0.04,    # 4 %
    "Erwachsene 21–<60": 0.52,        # 52 %
    "Senior:innen 60+": 0.26,         # 26 %
}


CRIME_SYNONYMS = {
    # ===== HOMICIDE =====
    "Mord Totschlag und Tötung auf Verlangen": "Mord & Totschlag",
    "Mord": "Mord",
    "Totschlag": "Totschlag",

    # ===== SEXUAL CRIME =====
    "Vergewaltigung sexuelle Nötigung und sexueller Übergriff": "Sexualstraftaten",
    "Vergewaltigung sexuelle Nötigung und sexueller Übergriff im besonders schweren Fall": "Sexualstraftaten",
    "Sexueller Missbrauch von Kindern": "Missbrauch Kinder",

    # ===== ROBBERY =====
    "Raub räuberische Erpressung und räuberischer Angriff auf Kraftfahrer": "Raub & Erpressung",
    "Raub räuberische Erpressung auf/gegen Geldinstitute": "Raub Banken/Post",
    "Raub räuberische Erpressung auf/gegen sonstige Kassenräume und Geschäfte": "Raub Geschäfte",
    "Raub räuberische Erpressung auf/gegen sonstige Zahlstellen und Geschäfte": "Raub Geschäfte",
    "Handtaschenraub": "Handtaschenraub",
    "Sonstige Raubüberfälle auf Straßen": "Raub auf Straßen",
    "Raubüberfälle in Wohnungen": "Raub in Wohnungen",

    # ===== ASSAULT =====
    "Gefährliche und schwere Körperverletzung": "Schwere KV",
    "Vorsätzliche einfache Körperverletzung": "Einfache KV",

    # ===== POLICE OFFENCES =====
    "Widerstand gegen und tätlicher Angriff auf Vollstreckungsbeamte": "Widerstand/Angriff Beamte",
    "Widerstand gegen Vollstreckungsbeamte": "Widerstand gegen Beamte",
    "Tätlicher Angriff auf Vollstreckungsbeamte": "Angriff auf Beamte",

    # ===== OTHER =====
    "Gewaltkriminalität": "Gewaltkriminalität",
    "Diebstahl insgesamt": "Diebstahl",
    "Betrug insgesamt": "Betrug",
    "Computerbetrug": "Cyberbetrug",
    "Rauschgiftdelikte": "Drogendelikte",
    "Sachbeschädigung": "Sachbeschädigung",
}


# --------- DATA CLEANING FUNCTION ---------
def clean_df(df_all: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and standardize the crime victim dataframe.
    """
    report = {
        "rows_in": int(len(df_all)),
        "rows_out": None,
        "dropped_missing_keys": 0,
        "missing_cols": [],
        "sanity": {},
    }

    if df_all is None or df_all.empty:
        report["rows_out"] = 0
        return df_all, report

    d = df_all.copy()

    # 1) Standardize column names + trim strings
    d.columns = [str(c).strip() for c in d.columns]
    for c in ["Straftat", "Fallstatus", "Stadt/Landkreis", "Bundesland", "Region"]:
        if c in d.columns:
            d[c] = d[c].astype(str).str.strip()

    # 2) Geographic keys
    if "Gemeindeschluessel" in d.columns:
        d["Gemeindeschluessel"] = pd.to_numeric(d["Gemeindeschluessel"], errors="coerce")
        d["Bundesland_Code"] = (d["Gemeindeschluessel"] // 1000).astype("Int64")
        d["Bundesland"] = d["Bundesland_Code"].map(STATE_MAP)

    if "Region" not in d.columns:
        if "Stadt/Landkreis" in d.columns:
            d["Region"] = d["Stadt/Landkreis"]
        else:
            d["Region"] = "Unbekannt"

    # 3) Fallstatus handling (prefer totals 'insg.'; otherwise sum available statuses)
    if "Fallstatus" in d.columns:
        d["Fallstatus"] = d["Fallstatus"].astype(str).str.strip().str.lower()
    else:
        report["missing_cols"].append("Fallstatus")

    # If we have multiple fallstatus rows (e.g., voll./vers./insg.), avoid double counting:
    # - If 'insg.' exists for a group, keep only 'insg.'
    # - Otherwise (no insg.), sum all statuses (so categories like 2022 sexual 'vers.' are not lost)
    if "Fallstatus" in d.columns:
        # Define group keys for deduplication / collapsing
        group_keys = [
            c
            for c in [
                "Jahr",
                "Bundesland",
                "Region",
                "Stadt/Landkreis",
                "Straftat",
                "Straftat_kurz",
                "Gemeindeschluessel",
                "Bundesland_Code",
            ]
            if c in d.columns
        ]

        def _collapse_status(g: pd.DataFrame) -> pd.DataFrame:
            if (g["Fallstatus"] == "insg.").any():
                g = g[g["Fallstatus"] == "insg."]
            return g

        # Keep only rows that will contribute, then aggregate numeric victim columns
        d = (
            d.groupby(group_keys, dropna=False, as_index=False)
            .apply(lambda g: _collapse_status(g))
            .reset_index(drop=True)
        )
        # After selecting the contributing rows, collapse duplicates by summing numeric columns
        numeric_cols = [c for c in d.columns if c == "Oper insgesamt" or str(c).startswith("Opfer ")]
        if numeric_cols:
            d = (
                d.groupby(group_keys, dropna=False)[numeric_cols]
                .sum()
                .reset_index()
            )

    # 4) Crime short names
    def short(s: str) -> str:
        s = str(s).strip()
        for long_name, short_name in CRIME_SYNONYMS.items():
            if long_name in s:
                return short_name
        return s.replace("  ", " ").strip()

    if "Straftat" in d.columns:
        d["Straftat_kurz"] = d["Straftat"].apply(short)
    else:
        report["missing_cols"].append("Straftat")
        d["Straftat_kurz"] = "Unbekannt"

    # 5) Numeric conversions
    if "Jahr" in d.columns:
        d["Jahr"] = pd.to_numeric(d["Jahr"], errors="coerce").astype("Int64")
    else:
        report["missing_cols"].append("Jahr")

    victim_like_cols = []
    for col in d.columns:
        if col == "Oper insgesamt" or str(col).startswith("Opfer "):
            victim_like_cols.append(col)

    for col in victim_like_cols:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)

    # 6) Drop rows missing essential keys
    essential = [c for c in ["Bundesland", "Region", "Straftat_kurz", "Jahr"] if c in d.columns]
    before = len(d)
    if essential:
        d = d.dropna(subset=essential).copy()
    report["dropped_missing_keys"] = int(before - len(d))

    # 7) Sanity checks
    sanity = {}
    if "Oper insgesamt" in d.columns and "Opfer maennlich" in d.columns and "Opfer weiblich" in d.columns:
        mf = d["Opfer maennlich"] + d["Opfer weiblich"]
        sanity["rows_total_lt_male_female"] = int((d["Oper insgesamt"] < mf).sum())
    else:
        sanity["rows_total_lt_male_female"] = None

    age_cols_present = [c for c in AGE_COLS.values() if c in d.columns]
    if "Oper insgesamt" in d.columns and age_cols_present:
        age_sum = d[age_cols_present].sum(axis=1)
        sanity["rows_total_lt_age_sum"] = int((d["Oper insgesamt"] < age_sum).sum())
    else:
        sanity["rows_total_lt_age_sum"] = None

    neg_count = 0
    for col in victim_like_cols:
        neg_count += int((d[col] < 0).sum())
    sanity["negative_values_count"] = int(neg_count)

    report["sanity"] = sanity
    report["rows_out"] = int(len(d))

    return d, report

# --------- LOAD DATA ---------
def load_data():
    dfs = []
    for year in range(2019, 2025):
        df = pd.read_csv(f"{year} Opfer.csv", sep=";", encoding="latin1")
        df.columns = [c.strip() for c in df.columns]  # removes trailing spaces
        df["Jahr"] = year
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    df_clean, rep = clean_df(df_all)
    print("✅ Clean report:", rep)
    return df_clean


df = load_data()
YEARS = sorted(df["Jahr"].unique())
CRIME_SHORT = sorted(df["Straftat_kurz"].unique())
STATES = sorted(df["Bundesland"].dropna().unique())


# Jobless Data


# --------- LOAD GEO DATA ---------
print("Lade Geodaten...")
try:
    # Load state boundaries
    gdf_states = gpd.read_file("data/gadm41_DEU_1.shp")
    gdf_states = gdf_states.explode(index_parts=True).reset_index(drop=True)
    gdf_states = gdf_states.to_crs("EPSG:4326")
    gdf_states["Bundesland"] = gdf_states["NAME_1"]

    # Load city boundaries (level 2) - still needed for city view
    gdf_cities = gpd.read_file("data/gadm41_DEU_2.shp")
    gdf_cities = gdf_cities.explode(index_parts=True).reset_index(drop=True)
    gdf_cities = gdf_cities.to_crs("EPSG:4326")
    gdf_cities["Bundesland"] = gdf_cities["NAME_1"]
    gdf_cities["City"] = gdf_cities["NAME_2"]

    print(
        f"Geladen: {len(gdf_states)} Bundesländer, {len(gdf_cities)} Städte/Landkreise")
except Exception as e:
    print(f"Fehler beim Laden der Geodaten: {e}")
    gdf_states = None
    gdf_cities = None

# --------- HELPERS ---------


def filter_data(years, crimes, states):
    d = df
    if years:
        d = d[d["Jahr"].isin(years)]
    if crimes:
        d = d[d["Straftat_kurz"].isin(crimes)]
    if states:
        d = d[d["Bundesland"].isin(states)]
    return d


def empty_fig(msg="Keine Daten verfügbar"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5,
                       showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def format_int(x):
    try:
        return f"{int(x):,}".replace(",", ".")
    except Exception:
        return "0"



# --------- KPI CALC ---------
def build_kpis(d):
    """
    KPIs:
    - Gesamtzahl der Opfer
    - Opfer pro Jahr (Ø)
    - männlich vs. weiblich (%)
    - Unter 18 vs. Erwachsene (%)
    - Anzahl Deliktsgruppen
    """
    if d.empty:
        return ("0", "0", "0 % / 0 %", "0 % / 0 %", "0")

    # 1) Gesamtzahl der Opfer
    total_victims = d["Oper insgesamt"].sum()

    # 2) Ø Opfer pro Jahr
    n_years = d["Jahr"].nunique()
    victims_per_year = int(round(total_victims / n_years)
                           ) if n_years > 0 else 0

    # 3) männlich vs. weiblich (%)
    male = d["Opfer maennlich"].sum() if "Opfer maennlich" in d.columns else 0
    female = d["Opfer weiblich"].sum() if "Opfer weiblich" in d.columns else 0
    sex_total = male + female

    def pct(part, whole):
        if whole <= 0:
            return "0,0 %"
        return f"{100 * part / whole:.1f} %".replace(".", ",")

    male_female_str = f"{pct(male, sex_total)} / {pct(female, sex_total)}"

    # 4) Unter 18 vs Erwachsene (%)
    col_children = "Opfer Kinder bis 14 Jahre- insgesamt"
    col_youth_14_18 = "Opfer Jugendliche 14 bis unter 18 Jahre - insgesamt"

    under18 = 0
    if col_children in d.columns:
        under18 += d[col_children].sum()
    if col_youth_14_18 in d.columns:
        under18 += d[col_youth_14_18].sum()

    adults = max(total_victims - under18, 0)
    under18_adults_str = f"{pct(under18, total_victims)} / {pct(adults, total_victims)}"

    # 5) Stadt mit höchster Kriminalitätsrate (Opfer pro 100k)
    top_city_rate = kpi_top_city_rate(d)

    return (
        format_int(total_victims),
        format_int(victims_per_year),
        male_female_str,
        under18_adults_str,
        top_city_rate,
    )


# --------- OVERVIEW FIGURES ---------
# Bubble chart: population vs crime rate Correlation
def fig_population_correlation(d):
    """
    Correlation between population size and crime rate (Bundesland level).
    """
    if d.empty:
        return empty_fig()

    # remove total category
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"].copy()
    if d2.empty:
        return empty_fig()

    # aggregate victims per Bundesland + year
    g = (
        d2.groupby(["Bundesland", "Jahr"])["Oper insgesamt"]
        .sum()
        .reset_index()
    )

    # attach population
    g["Population"] = g.apply(
        lambda r: get_population(r["Bundesland"], int(r["Jahr"])),
        axis=1,
    )

    g = g.dropna(subset=["Population"])
    if g.empty:
        return empty_fig("Keine Bevölkerungsdaten verfügbar")

    # aggregate across selected years
    g2 = (
        g.groupby("Bundesland")
        .agg(
            Victims=("Oper insgesamt", "sum"),
            Population=("Population", "mean"),
        )
        .reset_index()
    )

    g2["Rate_100k"] = 100000 * g2["Victims"] / g2["Population"]

    # Pearson correlation
    r = g2[["Population", "Rate_100k"]].corr().iloc[0, 1]

    fig = px.scatter(
        g2,
        x="Population",
        y="Rate_100k",
        size="Victims",
        hover_name="Bundesland",
        trendline="ols",
        color_discrete_sequence=["#1a80bb"],  # same blue as male victims
        labels={
            "Population": "Einwohnerzahl",
            "Rate_100k": "Opfer pro 100.000 Einwohner",
        },
        title=f"Bevölkerung vs. Kriminalitätsrate (r = {r:.2f})",
    )

    fig.update_traces(
        marker=dict(
            opacity=0.8,
            line=dict(width=1, color="#0f172a"),
            color="#1a80bb",   # enforce same blue everywhere
        )
    )

    fig.update_layout(
        height=520,
        plot_bgcolor="white",
        margin=dict(l=60, r=30, t=70, b=60),
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

    return fig



def fig_trend(d, metric_mode="abs"):
    """
    Single-line trend chart.
    metric_mode:
        - "abs"  -> absolute victims
        - "rate" -> victims per 100k inhabitants (Germany)
    """
    if d.empty:
        return empty_fig()

    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"].copy()
    if d2.empty:
        return empty_fig("Keine Deliktsdaten verfügbar")

    g = d2.groupby("Jahr")["Oper insgesamt"].sum().reset_index()
    g = g.sort_values("Jahr")

    # --- abs vs rate/100k (Germany) ---
    if metric_mode == "rate":
        # Requires Deutschland rows in population_states.csv
        g["Pop_DE"] = g["Jahr"].apply(
            lambda y: get_population("Deutschland", int(y)))
        g["Value"] = g.apply(lambda r: per_100k(
            r["Oper insgesamt"], r["Pop_DE"]), axis=1)
        y_title = "Opfer pro 100.000 Einwohner"
        hover_line = "<b>%{x}</b><br>%{y:.2f} pro 100.000<extra></extra>"
    else:
        g["Value"] = g["Oper insgesamt"]
        y_title = "erfasste Fälle"
        hover_line = "<b>%{x}</b><br>%{y:,} Opfer<extra></extra>"

    # --- year-to-year change based on plotted metric ---
    g["Delta"] = g["Value"].diff()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=g["Jahr"],
            y=g["Value"],
            mode="lines+markers",
            name="Trend",
            line=dict(color="#0f172a", width=4),   # ⬅ black line
            marker=dict(
                size=12,
                symbol="square",
                color="#0f172a",                  # ⬅ black markers
                line=dict(color="#0f172a", width=1),
            ),
            hovertemplate=hover_line,
        )
    )

    # --- Arrows for yearly increase / decrease ---
    for _, row in g.iloc[1:].iterrows():
        delta = row["Delta"]
        if pd.isna(delta):
            continue

        arrow_symbol = "▲" if delta >= 0 else "▼"
        arrow_color = "#dc2626" if delta >= 0 else "#16a34a"

        if metric_mode == "rate":
            arrow_text = f"{arrow_symbol} {abs(delta):.2f}"
        else:
            arrow_text = f"{arrow_symbol} {abs(delta):,}".replace(",", ".")

        fig.add_annotation(
            x=row["Jahr"],
            y=row["Value"],
            text=arrow_text,
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.8,
            arrowcolor=arrow_color,
            font=dict(size=12, color=arrow_color),
            ax=0,
            ay=-30 if delta >= 0 else 30,
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor=arrow_color,
            borderwidth=1,
        )

    fig.update_layout(
        title="Zeitliche Entwicklung der Opferzahlen",
        xaxis_title="",
        yaxis_title=y_title,
        height=550,  # you wanted smaller charts
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=70),
        showlegend=False,
    )

    # horizontal years
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#d1d5db",
        tickmode="linear",
        dtick=1,
        tickangle=0,   # <- horizontal
    )
    fig.update_yaxes(showgrid=True, gridcolor="#d1d5db")

    return fig


def fig_top5(d):
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()
    g = (
        d2.groupby("Straftat_kurz")["Oper insgesamt"]
        .sum()
        .nlargest(5)
        .reset_index()
        .sort_values("Oper insgesamt")
    )
    fig = px.bar(
        g,
        x="Oper insgesamt",
        y="Straftat_kurz",
        orientation="h",
        title="Top 5 Deliktsgruppen nach Opferzahl",
        labels={"Oper insgesamt": "Opferzahl",
                "Straftat_kurz": "Deliktsgruppe"},
    )
    # use dashboard primary blue for bars
    fig.update_traces(marker_color="#1a80bb",
                      textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    return fig


# --- New: Top-5 trend for a selected state (line chart) ---
def fig_state_top5_trend(d_state, metric_mode="abs"):
    if d_state.empty:
        return empty_fig()

    d2 = d_state[d_state["Straftat_kurz"] != "Straftaten insgesamt"].copy()
    if d2.empty:
        return empty_fig("Keine Deliktsdaten verfügbar")

    # top-5 by absolute totals
    top5 = d2.groupby("Straftat_kurz")[
        "Oper insgesamt"].sum().nlargest(5).index.tolist()
    g = (
        d2[d2["Straftat_kurz"].isin(top5)]
        .groupby(["Jahr", "Straftat_kurz"])["Oper insgesamt"]
        .sum()
        .reset_index()
    )

    state_name = d_state["Bundesland"].iloc[0] if (
        "Bundesland" in d_state.columns and not d_state["Bundesland"].empty) else ""

    if metric_mode == "rate":
        # compute per-100k using year-specific population
        g["Value"] = g.apply(lambda r: per_100k(r["Oper insgesamt"], get_population(
            state_name, int(r["Jahr"]))) if state_name else 0.0, axis=1)
        y_label = "Opfer pro 100.000 Einwohner"
    else:
        g["Value"] = g["Oper insgesamt"]
        y_label = "Opferzahl"

    # use blue shades palette consistent with dashboard theme
    palette = ["#1e40af", "#3b82f6", "#60a5fa", "#93c5fd", "#0ea5e9"]
    fig = px.line(
        g,
        x="Jahr",
        y="Value",
        color="Straftat_kurz",
        markers=True,
        title=f"Top-5 Deliktsgruppen in {state_name} (Zeitverlauf)",
        labels={"Value": y_label, "Straftat_kurz": "Deliktsgruppe"},
        color_discrete_sequence=palette,
    )

    if metric_mode == "rate":
        fig.update_traces(
            hovertemplate="%{x}<br>%{y:.1f} / 100k<extra></extra>")
    else:
        fig.update_traces(
            hovertemplate="%{x}<br>%{y:,.0f} Fälle<extra></extra>")

    fig.update_layout(height=420, plot_bgcolor="white",
                      margin=dict(l=60, r=30, t=70, b=60))
    return fig


# --- New: Age composition for a selected state ---


def fig_donut(d):
    """
    Statt Donut: Treemap zur Darstellung der Deliktsstruktur.
    Besser lesbar bei vielen Kategorien.
    """
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()

    g = d2.groupby("Straftat_kurz")["Oper insgesamt"].sum().reset_index()

    fig = px.treemap(
        g,
        path=["Straftat_kurz"],
        values="Oper insgesamt",
        color="Oper insgesamt",
        color_continuous_scale="Turbo",
        title="Struktur der Deliktsgruppen (Treemap)",
    )

    fig.update_layout(margin=dict(t=50, l=0, r=0, b=0))
    return fig


# Bubble chart for overview
def fig_overview_bubble(d):
    """Animated bubble chart: crime types over years (one bubble per crime type)."""
    if d.empty:
        return empty_fig()

    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"].copy()
    if d2.empty:
        return empty_fig("Keine Deliktsdaten verfügbar")

    g = (
        d2.groupby(["Jahr", "Straftat_kurz"])["Oper insgesamt"]
        .sum()
        .reset_index()
        .rename(columns={"Oper insgesamt": "Opfer"})
    )

    # Keep it readable: top 25 crime types overall
    top_types = (
        g.groupby("Straftat_kurz")["Opfer"]
        .sum()
        .nlargest(25)
        .index
        .tolist()
    )
    g = g[g["Straftat_kurz"].isin(top_types)].copy()

    # --- Boost bubble sizes for smaller categories (< 100k) so they remain visible ---
    # Keep the real value in `Opfer` (used for x + hover), but use a boosted column for marker sizing.
    threshold = 100_000
    boost_factor = 3.0
    g["SizePlot"] = np.where(g["Opfer"] < threshold,
                             g["Opfer"] * boost_factor, g["Opfer"])

    # Rank inside each year so bubbles sit on a stable vertical axis
    g["Rank"] = g.groupby("Jahr")["Opfer"].rank(
        method="first", ascending=False)

    # --- Fixed color mapping (ensure Einfache KV is dark red) ---
    color_map = {
        "Einfache KV": "#FF0000",   # dark red
    }
    fig = px.scatter(
        g,
        x="Opfer",
        y="Rank",
        size="SizePlot",
        color="Straftat_kurz",
        animation_frame="Jahr",
        hover_name="Straftat_kurz",
        size_max=85,
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        color_discrete_map=color_map,
        labels={"Opfer": "Opferzahl", "Rank": ""},
        title="Entwicklung der Deliktsgruppen im Zeitverlauf ",
    )

    fig.update_yaxes(autorange="reversed",
                     showticklabels=False, showgrid=False)

    fig.update_traces(
        marker=dict(opacity=0.75, line=dict(
            width=0.5, color="rgba(17,24,39,0.35)")),
        hovertemplate="<b>%{hovertext}</b><br>Opfer: %{x:,}<extra></extra>",
    )

    fig.update_layout(
        height=620,
        plot_bgcolor="white",
        margin=dict(l=80, r=30, t=70, b=40),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=-0.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            title_text="Deliktsgruppe",
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Opferzahl",
            range=[-50_000, 550_000],
            showgrid=True,
            gridcolor="#e5e7eb",
        ),
    )

    # Smooth animation (Play button exists automatically)
    if fig.layout.updatemenus and len(fig.layout.updatemenus) > 0:
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1400
        fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 800

    return fig


# --------- PIE CHART FOR OVERVIEW ---------
PKS_PIE_COLORS = [
    "#1e40af",  # blue
    "#f59e42",  # orange
    "#84cc16",  # lime green
    "#f43f5e",  # rose
    "#a21caf",  # purple
    "#0ea5e9",  # sky blue
    "#facc15",  # yellow
    "#64748b",  # slate
]


def fig_crime_pie(d):
    """
    Two-level pie chart like the reference figure:
    - Main chart (left): Top 10 crime categories + one slice "Andere".
    - Sub chart (right): Breakdown of the remaining categories (those inside "Andere").

    If there are <= 10 categories, we show a single pie.
    """
    if d.empty:
        return empty_fig()

    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()

    g = (
        d2.groupby("Straftat_kurz")["Oper insgesamt"]
        .sum()
        .reset_index()
        .sort_values("Oper insgesamt", ascending=False)
    )

    # Keep top 10 for the main chart
    top_n = 10
    top = g.head(top_n).copy()
    rest = g.iloc[top_n:].copy()

    # If there is no remainder, show a single donut pie
    if rest.empty:
        fig = px.pie(
            top,
            names="Straftat_kurz",
            values="Oper insgesamt",
            hole=0.4,
            title="Anteile der Deliktsgruppen (Überblick)",
            color_discrete_sequence=PKS_PIE_COLORS,
        )
        fig.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Opfer: %{value:,}<br>%{percent}<extra></extra>",
        )
        fig.update_layout(height=STANDARD_HEIGHT,
                          legend_title_text="Deliktsgruppe")
        return fig

    # Add "Andere" slice to the main chart
    other_sum = rest["Oper insgesamt"].sum()
    # --- Insert: Compute sub pie percentages relative to overall total ---
    total_sum = g["Oper insgesamt"].sum()
    # Percentages in the sub pie should be relative to the overall total (same basis as "Andere")
    rest = rest.copy()
    rest["pct_total"] = np.where(
        total_sum > 0, 100 * rest["Oper insgesamt"] / total_sum, 0.0)
    rest["pct_label"] = rest["pct_total"].map(
        lambda p: f"{p:.2f}%".replace(".", ","))
    main_df = pd.concat(
        [
            top,
            pd.DataFrame(
                {"Straftat_kurz": ["Andere"], "Oper insgesamt": [other_sum]}),
        ],
        ignore_index=True,
    )

    # Build a two-pie layout (main + sub)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        column_widths=[0.60, 0.40],
        horizontal_spacing=0.02,
        subplot_titles=(
            "Top 10 + Andere",
            "Aufschlüsselung von \"Andere\"",
        ),
    )

    # --- Colors ---
    # Main pie: use the PKS colors for the top 10; a neutral grey for "Andere"
    main_colors = (PKS_PIE_COLORS +
                   px.colors.qualitative.Dark24)[: len(main_df)]
    if len(main_colors) >= 1:
        main_colors[-1] = "#cbd5e1"  # grey for "Andere"

    # Sub pie: use a bigger qualitative palette
    sub_colors = (px.colors.qualitative.Set3 +
                  px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet)
    sub_colors = sub_colors[: len(rest)]

    # --- Main pie (left) ---
    fig.add_trace(
        go.Pie(
            labels=main_df["Straftat_kurz"],
            values=main_df["Oper insgesamt"],
            hole=0.4,
            textinfo="percent+label",
            marker=dict(colors=main_colors),
            sort=False,
            hovertemplate="<b>%{label}</b><br>Opfer: %{value:,}<br>%{percent}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --- Sub pie (right): breakdown of remainder ---
    fig.add_trace(
        go.Pie(
            labels=rest["Straftat_kurz"],
            values=rest["Oper insgesamt"],
            hole=0.0,
            text=rest["pct_label"],
            textinfo="label+text",
            marker=dict(colors=sub_colors),
            sort=False,
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Opfer: %{value:,}<br>"
                "Anteil gesamt: %{customdata:.2f}%<extra></extra>"
            ),
            customdata=rest["pct_total"],
        ),
        row=1,
        col=2,
    )
    # Demo chekcpoint
    # ---- Visual connector (wedge) between the two pies (paper coordinates) ----
    # Left pie domain will be roughly x in [0.0, ~0.60], right pie domain in [~0.62, 1.0]
    # We draw a light-grey wedge from the right edge of the left pie to the left edge of the right pie.
    x_left_edge = 0.60
    x_right_edge = 0.62
    y_top = 0.64
    y_bottom = 0.36
    x_apex = 0.52
    y_apex = 0.50

    # Filled wedge
    fig.add_shape(
        type="path",
        xref="paper",
        yref="paper",
        path=(
            f"M {x_apex},{y_apex} "
            f"L {x_left_edge},{y_top} "
            f"L {x_right_edge},{y_top} "
            f"L {x_right_edge},{y_bottom} "
            f"L {x_left_edge},{y_bottom} "
            f"Z"
        ),
        fillcolor="#cbd5e1",
        opacity=0.55,
        line=dict(color="#94a3b8", width=2),
        layer="above",
    )

    # Connector lines (to emulate the reference figure)
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=x_left_edge,
        y0=y_top,
        x1=x_right_edge,
        y1=y_top,
        line=dict(color="#94a3b8", width=2),
        layer="above",
    )
    fig.add_shape(
        type="line",
        xref="paper",
        yref="paper",
        x0=x_left_edge,
        y0=y_bottom,
        x1=x_right_edge,
        y1=y_bottom,
        line=dict(color="#94a3b8", width=2),
        layer="above",
    )

    # Optional: label in the wedge area (subtle)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.61,
        y=0.50,
        text="Andere",
        showarrow=False,
        font=dict(size=12, color="#475569"),
        bgcolor="rgba(255,255,255,0.0)",
    )

    fig.update_layout(
        title_text="Anteile der Deliktsgruppen",
        height=550,
        legend_title_text="Deliktsgruppe",
        margin=dict(t=80, l=10, r=10, b=40),
        legend=dict(
            orientation="v",
            xanchor="right",
            x=-0.02,
            yanchor="top",
            y=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            font=dict(size=11),
        ),
    )

    return fig


# --------- GEOGRAPHIC FIGURES ---------
def prepare_state_geo_data(d, value_col="Oper insgesamt", age_group_col=None):
    """Prepare state-level geographic data for the given metric column."""
    if d.empty or gdf_states is None:
        return None, None

    victims_df = d[d["Straftat_kurz"] != "Straftaten insgesamt"]

    if value_col not in victims_df.columns:
        value_col = "Oper insgesamt"

    # Calculate total victims for each state
    victims = (
        victims_df.groupby("Bundesland")[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "Opfer_insgesamt"})
    )

    # Calculate age group victims if specified
    age_group_victims = None
    if age_group_col and age_group_col in victims_df.columns:
        age_group_victims = (
            victims_df.groupby("Bundesland")[age_group_col]
            .sum()
            .reset_index()
            .rename(columns={age_group_col: "Opfer_altersgruppe"})
        )

    # Merge with geo data
    gdf_merged = gdf_states.merge(victims, on="Bundesland", how="left")

    # Add age group data if available
    if age_group_victims is not None:
        gdf_merged = gdf_merged.merge(
            age_group_victims, on="Bundesland", how="left")
    else:
        gdf_merged["Opfer_altersgruppe"] = 0

    # --- Consolidate duplicate columns (handles _x/_y suffixes after merges) ---
    gdf_merged = _consolidate_duplicate_columns(gdf_merged)
    if gdf_merged.columns.duplicated().any():
        gdf_merged = gdf_merged.loc[:, ~gdf_merged.columns.duplicated()]

    # Fill NaN values
    gdf_merged["Opfer_insgesamt"] = gdf_merged["Opfer_insgesamt"].fillna(0)
    gdf_merged["Opfer_altersgruppe"] = gdf_merged["Opfer_altersgruppe"].fillna(
        0)

    # --- Compute mean population for the selected years and per-100k (state-level only) ---
    years = sorted(d["Jahr"].unique()) if "Jahr" in d.columns else []

    def mean_pop_for_state(bl):
        if POP.empty or not years:
            return None
        sub = POP[(POP["Bundesland"] == bl) & (POP["Jahr"].isin(years))]
        if sub.empty:
            return None
        return float(sub["Population"].mean())

    gdf_merged["Population_mean"] = gdf_merged["Bundesland"].apply(
        mean_pop_for_state)

    def safe_per100k(val, pop):
        try:
            return per_100k(val, pop)
        except Exception:
            return 0.0

    gdf_merged["Opfer_insgesamt_per100k"] = gdf_merged.apply(
        lambda r: safe_per100k(r["Opfer_insgesamt"], r["Population_mean"]), axis=1
    )
    gdf_merged["Opfer_altersgruppe_per100k"] = gdf_merged.apply(
        lambda r: safe_per100k(r["Opfer_altersgruppe"], r["Population_mean"]), axis=1
    )

    # Compute map center robustly
    try:
        gdf_projected = gdf_merged.to_crs("EPSG:32632")
        centroid = gdf_projected.geometry.centroid
        centroid_wgs84 = centroid.to_crs("EPSG:4326")
        center_lat = centroid_wgs84.y.mean()
        center_lon = centroid_wgs84.x.mean()
    except Exception as e:
        logger.warning(
            "Could not compute projected centroid for states: %s", e)
        center_lat = gdf_merged.geometry.centroid.y.mean()
        center_lon = gdf_merged.geometry.centroid.x.mean()

    geojson_data = json.loads(gdf_merged.to_json())
    return gdf_merged, geojson_data, (center_lat, center_lon)


def _norm_admin_name(x: str) -> str:
    """Normalize German admin/city strings so Region names match shapefile city names better."""
    x = str(x).lower().strip()

    # Replace umlauts/eszett (common mismatch cause)
    x = (
        x.replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )

    # Remove common administrative words that appear in CSV but not in shapes
    x = re.sub(
        r"\b(landkreis|kreisfreie\s+stadt|kreis|stadt|region|lk|sk|reg\.|bezirk)\b",
        " ",
        x,
    )

    # Remove punctuation and collapse whitespace
    x = re.sub(r"[^a-z0-9\s-]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def _consolidate_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidate duplicate column groups that differ only by merge suffixes
    (e.g., 'Opfer_insgesamt', 'Opfer_insgesamt_x', 'Opfer_insgesamt_y').

    Strategy:
    - For each base name (remove trailing _x/_y), if multiple candidates exist:
      - choose the candidate with the most non-null values as the 'best'
      - fill NA in 'best' from others, drop the other candidates
      - finally rename 'best' back to the base name
    """
    cols = list(df.columns)
    groups = {}
    for c in cols:
        base = re.sub(r'(_x|_y)$', '', c)
        groups.setdefault(base, []).append(c)

    for base, candidates in groups.items():
        if len(candidates) <= 1:
            continue
        # choose the column with most non-null values
        best = max(candidates, key=lambda col: df[col].notna().sum())
        for c in candidates:
            if c == best:
                continue
            df[best] = df[best].fillna(df[c])
            df.drop(columns=[c], inplace=True)
        if best != base:
            df.rename(columns={best: base}, inplace=True)

    # Final fallback: drop literal duplicate labels if any remain
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df


def prepare_city_geo_data(d, selected_state=None, value_col="Oper insgesamt", age_group_col=None):
    """
    Prepare city-level geographic data.
    - selected_state = None  -> all cities in Germany
    - selected_state = Name  -> only cities of this state

    FIX:
    - match Region->City first
    - keep only matched cities
    - THEN apply Top-N later in fig_geo_map

    This prevents Top10 showing 9, Top20 showing 19, etc.
    """
    if d.empty or gdf_cities is None:
        return None, None, None

    if selected_state:
        state_data = d[d["Bundesland"] == selected_state].copy()
        gdf_subset = gdf_cities[gdf_cities["Bundesland"]
                                == selected_state].copy()
    else:
        state_data = d.copy()
        gdf_subset = gdf_cities.copy()

    if state_data.empty or gdf_subset.empty:
        return None, None, None

    if value_col not in state_data.columns:
        value_col = "Oper insgesamt"

    # --- Aggregate by Region + Bundesland (prevents ambiguity like "Neustadt" in multiple states) ---
    city_victims = (
        state_data.groupby(["Region", "Bundesland"])[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "Opfer_insgesamt"})
    )

    # --- Age group victims (same grouping) ---
    if age_group_col and age_group_col in state_data.columns:
        age_group_city_victims = (
            state_data.groupby(["Region", "Bundesland"])[age_group_col]
            .sum()
            .reset_index()
            .rename(columns={age_group_col: "Opfer_altersgruppe"})
        )
        city_victims = city_victims.merge(
            age_group_city_victims, on=["Region", "Bundesland"], how="left"
        )
    else:
        city_victims["Opfer_altersgruppe"] = 0

    city_victims["Opfer_altersgruppe"] = city_victims["Opfer_altersgruppe"].fillna(
        0)

    # --- Build per-state normalized lookup for shapefile city names ---
    gdf_subset = gdf_subset.copy()
    gdf_subset["City_norm"] = gdf_subset["City"].apply(_norm_admin_name)

    # Dict: {Bundesland -> {City_norm -> City}}
    lookup_by_state = {}
    for bl, sub in gdf_subset.groupby("Bundesland"):
        lookup_by_state[bl] = dict(zip(sub["City_norm"], sub["City"]))

    def match_city(region_name: str, bundesland_name: str):
        r = _norm_admin_name(region_name)
        if not r:
            return None

        city_norm_to_city = lookup_by_state.get(bundesland_name, {})
        if not city_norm_to_city:
            return None

        # 1) exact normalized match
        if r in city_norm_to_city:
            return city_norm_to_city[r]

        # 2) substring fallback inside the same Bundesland
        for cn, real_city in city_norm_to_city.items():
            if r in cn or cn in r:
                return real_city

        return None

    # Match Region -> City within the same Bundesland (much higher accuracy)
    city_victims["City_match"] = city_victims.apply(
        lambda row: match_city(row["Region"], row["Bundesland"]), axis=1
    )

    # Debug: show unmatched Regions (enable if needed)
    # unmatched = city_victims[city_victims["City_match"].isna()][["Region"]].drop_duplicates()
    # print("Unmatched Regions (sample):", unmatched.head(30).to_string(index=False))

    # Keep only matched entries (these are drawable on the map)
    matched = city_victims.dropna(subset=["City_match"]).copy()
    if matched.empty:
        return None, None, None

    matched_city = (
        matched.groupby(["Bundesland", "City_match"])[
            ["Opfer_insgesamt", "Opfer_altersgruppe"]]
        .sum()
        .reset_index()
        .rename(columns={"City_match": "City"})
    )

    # Merge into GeoDataFrame using BOTH keys (Bundesland + City)
    gdf_merged = gdf_subset.merge(
        matched_city, on=["Bundesland", "City"], how="left")
    # --- Consolidate duplicate columns (handles _x/_y suffixes after merges) ---
    gdf_merged = _consolidate_duplicate_columns(gdf_merged)
    if gdf_merged.columns.duplicated().any():
        gdf_merged = gdf_merged.loc[:, ~gdf_merged.columns.duplicated()]
    gdf_merged["Opfer_insgesamt"] = gdf_merged["Opfer_insgesamt"].fillna(0)
    gdf_merged["Opfer_altersgruppe"] = gdf_merged["Opfer_altersgruppe"].fillna(
        0)

    # Calculate map center
    try:
        gdf_projected = gdf_merged.to_crs("EPSG:32632")
        centroid = gdf_projected.geometry.centroid
        centroid_wgs84 = centroid.to_crs("EPSG:4326")
        center_lat = centroid_wgs84.y.mean()
        center_lon = centroid_wgs84.x.mean()
    except Exception as e:
        print(
            f"Warning: Could not calculate proper centroid for {selected_state or 'Deutschland'}, using simple mean: {e}"
        )
        center_lat = gdf_merged.geometry.centroid.y.mean()
        center_lon = gdf_merged.geometry.centroid.x.mean()

    geojson_data = json.loads(gdf_merged.to_json())
    return gdf_merged, geojson_data, (center_lat, center_lon)


# ----- COLOR SCALES FOR SAFETY MODE -----
COLOR_SCALE_UNSAFE = "Reds"
# Safe mode: green-only scale, REVERSED so that LOWER crime = DARKER green
COLOR_SCALE_SAFE = "Greens_r"
COLOR_SCALE_ALL = "OrRd"


# ----- MAP FOCUS (Germany only) -----
# Bounding box for Germany in WGS84 (lon/lat). Used to keep the map from drifting to other countries.
GERMANY_BOUNDS = {"west": 5.5, "east": 15.7, "south": 47.0, "north": 55.6}


def _bounds_from_gdf(gdf: gpd.GeoDataFrame, pad: float = 0.4):
    """Compute a padded WGS84 bounding box from a GeoDataFrame."""
    try:
        g = gdf.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = g.total_bounds
        return {
            "west": float(minx - pad),
            "south": float(miny - pad),
            "east": float(maxx + pad),
            "north": float(maxy + pad),
        }
    except Exception:
        return GERMANY_BOUNDS


def _apply_map_focus(fig: go.Figure, bounds: dict, center: dict, zoom: float):
    """Force Mapbox view to stay within bounds (prevents showing surrounding countries too much)."""
    fig.update_layout(
        mapbox=dict(
            center=center,
            zoom=zoom,
            bounds=bounds,
        )
    )
    return fig


def fig_geo_map(d, selected_state=None, city_mode="bundesland", age_group="all", safety_mode="all", metric_mode="abs"):
    """
    Handles BOTH Bundesländer & City view with safety-mode coloring.
    safety_mode:
        - "safe"   → green scale (low = good)
        - "unsafe" → red scale (high = dangerous)
        - "all"    → neutral scale
    metric_mode:
        - "abs" -> absolute counts
        - "rate" -> victims per 100k (states only)
    """

    if d.empty or gdf_states is None:
        return empty_fig("Keine Geodaten verfügbar")

    # ----- Select metric column (age-aware) -----
    value_col = "Oper insgesamt"
    age_group_col = None
    age_label_for_title = "alle Altersgruppen"
    if age_group != "all" and age_group in AGE_COLS:
        candidate = AGE_COLS[age_group]
        if candidate in d.columns:
            age_group_col = candidate
            age_label_for_title = age_group

    # Metric used for coloring + Top-N ranking
    metric_col = "Opfer_altersgruppe" if age_group_col is not None else "Opfer_insgesamt"
    metric_label = f"Opfer {age_label_for_title}" if age_group_col is not None else "Opfer gesamt"
    # Flag to indicate per-100k rate mode
    is_rate = (metric_mode == "rate")

    # ----- Choose color scale -----
    if safety_mode == "safe":
        color_scale = COLOR_SCALE_SAFE   # greens
        ascending = True                # safest first
    elif safety_mode == "unsafe":
        color_scale = COLOR_SCALE_UNSAFE  # reds
        ascending = False
    else:
        color_scale = COLOR_SCALE_ALL   # orange neutral
        ascending = False

    # ----------------------------------------------------
    # ✅ BUNDESLÄNDER VIEW ----------------------------------------------------
    # ----------------------------------------------------
    if city_mode == "bundesland" and selected_state is None:
        gdf_states_data, geojson_data, center = prepare_state_geo_data(
            d, value_col, age_group_col)
        if gdf_states_data is None:
            return empty_fig("Keine Bundeslanddaten verfügbar")

        # If rate mode requested, use per-100k columns (state-level only)
        if metric_mode == "rate":
            metric_col = "Opfer_altersgruppe_per100k" if age_group_col is not None else "Opfer_insgesamt_per100k"
            metric_label = "Opfer pro 100.000 Einwohner"

        # Sort by safe/unsafe using metric_col
        gdf_states_data = gdf_states_data.sort_values(
            metric_col, ascending=ascending)

        # Debug: record columns to logger (debug level, not printed by default)
        logger.debug('gdf_states_data cols: %s', list(gdf_states_data.columns))
        logger.debug('gdf_states_data duplicated: %s',
                     gdf_states_data.columns[gdf_states_data.columns.duplicated()].tolist())

        # Build custom_data avoiding duplicates (include absolute counts when showing rates)
        if metric_mode == "rate":
            if age_group != "all":
                custom_data_states = ["Bundesland", metric_col,
                                      "Opfer_insgesamt", "Opfer_altersgruppe"]
            else:
                custom_data_states = ["Bundesland",
                                      metric_col, "Opfer_insgesamt"]
        else:
            if age_group != "all":
                custom_data_states = ["Bundesland", metric_col,
                                      "Opfer_insgesamt", "Opfer_altersgruppe"]
            else:
                custom_data_states = ["Bundesland", metric_col]
        custom_data_states = list(dict.fromkeys(custom_data_states))

        try:
            fig = px.choropleth_map(
                gdf_states_data,
                geojson=geojson_data,
                locations=gdf_states_data.index,
                color=metric_col,
                hover_name="Bundesland",
                hover_data={
                    metric_col: True,
                    "Opfer_insgesamt": True,
                    "Bundesland": False
                },
                custom_data=custom_data_states,
                opacity=0.7,
                map_style="carto-positron",
                color_continuous_scale=color_scale,
                zoom=4.5,
                center={"lat": center[0], "lon": center[1]},
                title=f"Opfer nach Bundesland – {age_label_for_title}",
            )
        except Exception as e:
            logger.error("Error creating choropleth for states: %s", e)
            try:
                logger.debug('gdf_states_data cols: %s',
                             list(gdf_states_data.columns))
                logger.debug('gdf_states_data duplicated: %s',
                             gdf_states_data.columns[gdf_states_data.columns.duplicated()].tolist())
            except Exception:
                pass
            raise

        # Hovertemplate: always show metric_label, total, and age group if selected
        if metric_mode == "rate":
            # show rate (1 decimal) plus the absolute counts
            if age_group != "all":
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        + f"{metric_label}: %{{customdata[1]:.1f}} / 100k<br>"
                        + "Opfer gesamt: %{customdata[2]:,.0f}<br>"
                        + f"Opfer {age_label_for_title}: %{{customdata[3]:,.0f}}<br>"
                        + "<extra></extra>"
                    )
                )
            else:
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        + f"{metric_label}: %{{customdata[1]:.1f}} / 100k<br>"
                        + "Opfer gesamt: %{customdata[2]:,.0f}<br>"
                        + "<extra></extra>"
                    )
                )
            # colorbar title: per 100k
            fig.update_coloraxes(colorbar=dict(title="Opfer / 100k"))
        else:
            if age_group != "all":
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        + f"{metric_label}: %{{customdata[1]:,.0f}}<br>"
                        + "Opfer gesamt: %{customdata[2]:,.0f}<br>"
                        + f"Opfer {age_label_for_title}: %{{customdata[3]:,.0f}}<br>"
                        + "<extra></extra>"
                    )
                )
            else:
                fig.update_traces(
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        + f"{metric_label}: %{{customdata[1]:,.0f}}<br>"
                        + "<extra></extra>"
                    )
                )

        fig.update_layout(
            margin={"l": 0, "r": 50, "t": 0, "b": 0},
            height=420,
            clickmode="event+select",
        )

        # Small colorbar (full height, compact thickness)
        fig.update_coloraxes(
            showscale=True,
            colorbar=dict(
                title="",
                len=1.0,           # full height
                lenmode="fraction",
                thickness=10,      # thinner
                x=1.02,            # just outside the map
                y=0.5,
                yanchor="middle",
                outlinewidth=0,
            ),
        )

        # Keep the map focused on Germany
        bounds = _bounds_from_gdf(gdf_states_data, pad=0.35)
        _apply_map_focus(
            fig,
            bounds=bounds,
            center={"lat": 51.0, "lon": 10.2},
            zoom=5.6,
        )

        return fig

    # ----------------------------------------------------
    # ✅ CITY VIEW (all Germany OR inside Bundesland)
    # ----------------------------------------------------
    gdf_cities_data, geojson_data, center = prepare_city_geo_data(
        d, selected_state, value_col, age_group_col)
    if gdf_cities_data is None:
        return empty_fig("Keine Städtedaten verfügbar")

    center_lat, center_lon = center

    gdf_plot = gdf_cities_data.copy()

    # For Top-N views, rank ONLY cities with data (avoid irrelevant 0-value polygons)
    if city_mode != "all" and isinstance(city_mode, int):
        gdf_rank = gdf_plot[gdf_plot[metric_col] > 0].copy()
        if gdf_rank.empty:
            return empty_fig("Keine Städtedaten verfügbar (nach Filter).")
        gdf_plot = gdf_rank.sort_values(
            metric_col, ascending=ascending).head(city_mode)

    try:
        fig = px.choropleth_map(
            gdf_plot,
            geojson=geojson_data,
            locations=gdf_plot.index,
            color=metric_col,
            hover_name="City",
            hover_data={
                metric_col: True,
                "Opfer_insgesamt": True,
                "Opfer_altersgruppe": True,
                "Bundesland": True,
                "City": False
            },
            # Build custom_data avoiding duplicates
            # When age_group == 'all' metric_col == 'Opfer_insgesamt', so avoid adding it twice
            custom_data=(lambda: (list(dict.fromkeys([
                "City", "Bundesland", metric_col,
                "Opfer_insgesamt", "Opfer_altersgruppe"
            ]))))(),
            opacity=0.8,
            map_style="carto-positron",
            color_continuous_scale=color_scale,
            zoom=6 if selected_state else 5,
            center={"lat": center_lat, "lon": center_lon},
            title=f"Opfer – Städteansicht – {age_label_for_title}",
        )
    except Exception as e:
        logger.error("Error creating choropleth for cities: %s", e)
        try:
            logger.debug('gdf_plot cols: %s', list(gdf_plot.columns))
            logger.debug('gdf_plot duplicated: %s',
                         gdf_plot.columns[gdf_plot.columns.duplicated()].tolist())
        except Exception:
            pass
        raise

    # Hovertemplate: always show metric_label, total, and age group if selected
    if age_group != "all":
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "Bundesland: %{customdata[1]}<br>"
                + f"{metric_label}: %{{customdata[2]:,.0f}}<br>"
                + "Opfer gesamt: %{customdata[3]:,.0f}<br>"
                + f"Opfer {age_label_for_title}: %{{customdata[4]:,.0f}}<br>"
                + "<extra></extra>"
            )
        )
    else:
        fig.update_traces(
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + "Bundesland: %{customdata[1]}<br>"
                + f"{metric_label}: %{{customdata[2]:,.0f}}<br>"
                + "<extra></extra>"
            )
        )

    fig.update_layout(
        margin={"l": 0, "r": 50, "t": 0, "b": 0},
        height=420,
        clickmode="none",
    )

    # Small colorbar (full height, compact thickness)
    fig.update_coloraxes(
        showscale=True,
        colorbar=dict(
            title="",
            len=1.0,
            lenmode="fraction",
            thickness=10,
            x=1.02,
            y=0.5,
            yanchor="middle",
            outlinewidth=0,
        ),
    )

    # Keep map focused (Germany or selected Bundesland)
    if selected_state:
        bounds = _bounds_from_gdf(gdf_plot, pad=0.15)
        zoom_val = 6.6
        center_val = {"lat": center_lat, "lon": center_lon}
    else:
        bounds = GERMANY_BOUNDS
        zoom_val = 5.6
        center_val = {"lat": 51.0, "lon": 10.2}

    _apply_map_focus(fig, bounds=bounds, center=center_val, zoom=zoom_val)

    return fig


def fig_geo_state_bar(d):
    if d.empty:
        return empty_fig()

    # Remove total-crime category
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]

    # Aggregate only real crime categories
    g = (
        d2.groupby("Bundesland")["Oper insgesamt"]
        .sum()
        .reset_index()
        .sort_values("Oper insgesamt", ascending=True)
    )

    # Add formatted label column for value labels
    g["Label"] = g["Oper insgesamt"].apply(format_int)

    # Normalize values for color intensity
    min_val = g["Oper insgesamt"].min()
    max_val = g["Oper insgesamt"].max()
    norm = (g["Oper insgesamt"] - min_val) / (max_val - min_val + 1e-9)

    # Create figure and add lollipop stems and markers
    fig = go.Figure()

    # Stems
    fig.add_trace(
        go.Scatter(
            x=g["Oper insgesamt"],
            y=g["Bundesland"],
            mode="lines",
            line=dict(color="#94a3b8", width=2),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Dots with value labels
    fig.add_trace(
        go.Scatter(
            x=g["Oper insgesamt"],
            y=g["Bundesland"],
            mode="markers+text",
            text=g["Label"],
            textposition="middle right",
            textfont=dict(size=12, color="#0f172a"),
            marker=dict(
                size=14,
                color=norm,
                colorscale="Blues",
                showscale=False,
                line=dict(color="#1e40af", width=0.6),
            ),
            hovertemplate="<b>%{y}</b><br>Opfer: %{x}<extra></extra>",
            showlegend=False,
        )
    )

    # Layout styling (tighter margins to remove extra whitespace)
    fig.update_layout(
        title="Opfer nach Bundesland",
        xaxis_title="Opferzahl",
        yaxis_title="Bundesland",
        height=420,
        margin=dict(l=60, r=20, t=40, b=40),
        plot_bgcolor="white",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=False)

    # Increase x-axis range but keep it tight around actual values to remove left/right gaps
    min_x = float(g["Oper insgesamt"].min()) if not g.empty else 0.0
    max_x = float(g["Oper insgesamt"].max()) if not g.empty else 0.0
    # Use slightly inward lower bound (near smallest value) and slight padding on upper bound
    lower = min_x * 0.92 if min_x > 0 else 0
    upper = max_x * 1.02 if max_x > 0 else 1
    fig.update_xaxes(range=[lower, upper])

    return fig


def fig_geo_top(d):
    if d.empty:
        return empty_fig()

    # Aggregate by city/region only
    g = (
        d.groupby("Region")["Oper insgesamt"]
        .sum()
        .reset_index()
        .nlargest(10, "Oper insgesamt")
        .sort_values("Oper insgesamt")
    )

    # Add formatted label column for value labels
    g["Label"] = g["Oper insgesamt"].apply(format_int)

    fig = px.bar(
        g,
        x="Oper insgesamt",
        y="Region",               # <-- Only Landkreis names
        orientation="h",
        text="Label",
        title="Top 10 Landkreise nach Opferzahl",
        labels={"Oper insgesamt": "Opferzahl", "Region": "Landkreis"},
    )

    # Set static blue color for all bars
    fig.update_traces(
        # same blue as Zeitliche Einblicke (Change 2019–2024)
        marker_color="#1a80bb"
    )
    # Show value labels at the end of bars
    fig.update_traces(textposition="outside", cliponaxis=False)

    fig.update_layout(
        coloraxis_showscale=False,
        xaxis_title="Opferzahl",
        yaxis_title="Landkreis",
        height=420,
        margin=dict(l=80, r=80, t=50, b=40),
    )

    return fig


# --------- CRIME TYPE FIGURES ---------
def fig_heatmap(d):
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()

    g = (
        d2.groupby(["Straftat_kurz", "Jahr"])["Oper insgesamt"]
        .sum()
        .reset_index()
    )

    pivot = (
        g.pivot_table(index="Straftat_kurz", columns="Jahr",
                      values="Oper insgesamt", aggfunc="sum")
        .fillna(0)
    )

    # order (largest near top), then reverse like your other charts
    row_order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[row_order]

    x_years = [int(y) for y in pivot.columns.tolist()]
    y_crimes = pivot.index.tolist()
    z = pivot.values

    # text inside cells
    text = np.vectorize(lambda v: f"{int(v):,}".replace(",", "."))(z)

    fig = go.Figure(
        data=go.Heatmap(
            x=x_years,
            y=y_crimes,
            z=z,
            colorscale="Blues",   # blue scale (consistent with gender comparison chart)
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=10, color="#111827"),
            hovertemplate="<b>%{y}</b><br>Jahr: %{x}<br>Opfer: %{z:,}<extra></extra>",
            xgap=1,  # grid gaps (like the image)
            ygap=1,
            colorbar=dict(title="Opferzahl"),
        )
    )

    n_rows = max(1, len(y_crimes))
    fig.update_layout(
        title="Heatmap – Opferzahlen nach Deliktsgruppe und Jahr",
        height=max(750, int(n_rows * 26)),  # auto bigger when many crime types
        margin=dict(l=220, r=30, t=70, b=70),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    fig.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)

    # square-ish cells
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig


def fig_stacked(d):
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"]
    if d2.empty:
        return empty_fig()

    # Top 6 crime types by total victims
    top = (
        d2.groupby("Straftat_kurz")["Oper insgesamt"]
        .sum()
        .nlargest(6)
        .index
    )
    d_top = d2[d2["Straftat_kurz"].isin(top)].copy()

    g = (
        d_top.groupby(["Jahr", "Straftat_kurz"])["Oper insgesamt"]
        .sum()
        .reset_index()
    )

    # --- New design: stacked area chart (cleaner than stacked bars) ---
    fig = go.Figure()
    # Muted sequential palette (crime/intensity feel) instead of random categorical colors
    # Muted sequential palette (crime/intensity feel) instead of random categorical colors
    colors = [
        "#7f1d1d",  # dark red
        "#b91c1c",  # red
        "#ef4444",  # light red
        "#f97316",  # orange
        "#f59e0b",  # amber
        "#fde68a",  # light yellow
    ]

    for i, crime in enumerate(sorted(top)):
        df_c = g[g["Straftat_kurz"] == crime].sort_values("Jahr")
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=df_c["Jahr"],
                y=df_c["Oper insgesamt"],
                mode="lines",
                name=crime,
                stackgroup="one",

                # 🔑 LINE
                line=dict(
                    width=2,
                    color=color,
                ),

                # 🔑 AREA (BODY)
                fill="tonexty",
                fillcolor=color,
                opacity=0.6,

                hovertemplate=(
                    f"<b>{crime}</b><br>"
                    "Jahr: %{x}<br>"
                    "Opfer: %{y:,}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Top-Deliktsgruppen im Zeitverlauf",
        xaxis_title="",
        yaxis_title="erfasste Fälle",
        height=700,
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=90),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#e5e7eb",
            borderwidth=1,
            title_text="Deliktsgruppe",
            font=dict(size=11),
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#d1d5db",
        tickmode="linear",
        dtick=1,
        tickangle=-90,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#d1d5db")

    return fig


AGE_POPULATION_SHARE = {
    "Kinder <14": 0.129,
    "Jugendliche 14–<18": 0.045,
    "Heranwachsende 18–<21": 0.037,
    "Erwachsene 21–<60": 0.569,
    "Senior:innen 60+": 0.22,
}


def fig_age(d, crime):
    if d.empty:
        return empty_fig()

    d_sel = d[d["Straftat_kurz"] == crime]
    if d_sel.empty:
        return empty_fig("Keine Daten für diese Deliktsgruppe")

    # --- Aggregate victims by age group ---
    values = {
        label: d_sel[col].sum()
        for label, col in AGE_COLS.items()
        if col in d_sel.columns
    }

    if not values:
        return empty_fig("Keine Altersdaten verfügbar")

    df_age = pd.DataFrame(
        {"Altersgruppe": list(values.keys()), "Opfer": list(values.values())}
    )

    total_victims = df_age["Opfer"].sum()
    if total_victims <= 0:
        return empty_fig("Keine gültigen Opferzahlen")

    # --- Victim share ---
    df_age["Victim_share"] = df_age["Opfer"] / total_victims

    # --- Population share ---
    df_age["Population_share"] = df_age["Altersgruppe"].map(
        AGE_POPULATION_SHARE
    )

    df_age = df_age.dropna(subset=["Population_share"])

    # --- Risk index (100 = expected) ---
    df_age["Risk_index"] = 100 * (
        df_age["Victim_share"] / df_age["Population_share"]
    )

    # --- Sort for clarity ---
    df_age = df_age.sort_values("Risk_index")

    # --- Colors ---
    AGE_COLORS = {
        "Kinder <14": "#38bdf8",
        "Jugendliche 14–<18": "#22c55e",
        "Heranwachsende 18–<21": "#facc15",
        "Erwachsene 21–<60": "#ef4444",
        "Senior:innen 60+": "#8b5cf6",
    }

    fig = go.Figure()

    # Reference line (expected = 100)
    fig.add_vline(
        x=100,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="Durchschnitt (100)",
        annotation_position="top",
    )

    # Lollipop stems
    fig.add_trace(
        go.Scatter(
            x=df_age["Risk_index"],
            y=df_age["Altersgruppe"],
            mode="lines",
            line=dict(color="#e5e7eb", width=4),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Lollipop dots
    fig.add_trace(
        go.Scatter(
            x=df_age["Risk_index"],
            y=df_age["Altersgruppe"],
            mode="markers+text",
            marker=dict(
                size=18,
                color=[AGE_COLORS[a] for a in df_age["Altersgruppe"]],
                line=dict(color="white", width=1.5),
            ),
            text=[f"{v:.0f}" for v in df_age["Risk_index"]],
            textposition="middle right",
            hovertemplate="<b>%{y}</b><br>Risikofaktor: %{x:.1f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title={
            "text": f"Altersstruktur der Opfer – {crime} (bevölkerungsbereinigt)",
            "x": 0.02,
            "xanchor": "left",
        },
        xaxis=dict(
            title="Risikofaktor (100 = erwarteter Anteil)",
            showgrid=True,
            gridcolor="#f1f5f9",
            zeroline=False,
            range=[
                min(80, df_age["Risk_index"].min() * 0.9),
                df_age["Risk_index"].max() * 1.25,
            ],
        ),
        yaxis=dict(showgrid=False),
        height=STANDARD_HEIGHT,
        margin=dict(l=160, r=60, t=70, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        transition_duration=600,
        transition_easing="cubic-in-out",
    )

    return fig


# --------- TEMPORAL FIGURES ---------
def fig_state_trend(d):
    if d.empty:
        return empty_fig()

    # Top 6 Bundesländer by total victims (same logic as before)
    top_states = (
        d.groupby("Bundesland")["Oper insgesamt"]
        .sum()
        .nlargest(6)
        .index
    )

    d_top = d[d["Bundesland"].isin(top_states)]

    g = (
        d_top.groupby(["Jahr", "Bundesland"])["Oper insgesamt"]
        .sum()
        .reset_index()
    )

    fig = go.Figure()

    # Use a restrained qualitative palette
    colors = px.colors.qualitative.Dark24

    for i, state in enumerate(sorted(top_states)):
        df_s = g[g["Bundesland"] == state]

        fig.add_trace(
            go.Scatter(
                x=df_s["Jahr"],
                y=df_s["Oper insgesamt"],
                mode="lines+markers",
                name=state,
                line=dict(
                    width=3,
                    color=colors[i % len(colors)],
                ),
                marker=dict(
                    size=9,
                    symbol="square",   # same marker style as reference
                    color=colors[i % len(colors)],
                    line=dict(color="#111827", width=0.6),
                ),
                hovertemplate="<b>%{x}</b><br>%{y:,} Opfer<extra></extra>",
            )
        )

    fig.update_layout(
        title="Ländervergleich im Zeitverlauf",
        xaxis_title="",
        yaxis_title="erfasste Fälle",
        height=STANDARD_HEIGHT,
        plot_bgcolor="white",
        margin=dict(l=70, r=30, t=70, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#d1d5db",
        tickmode="linear",
        dtick=1,
        tickangle=-90,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#d1d5db",
    )

    return fig


def fig_diverg(d):
    if d.empty:
        return empty_fig()

    years = sorted(d["Jahr"].unique())
    if len(years) < 2:
        return empty_fig("Mindestens zwei Jahre notwendig.")

    first, last = years[0], years[-1]

    g = d.groupby(["Bundesland", "Jahr"])["Oper insgesamt"].sum().reset_index()
    start = g[g["Jahr"] == first].set_index("Bundesland")["Oper insgesamt"]
    end = g[g["Jahr"] == last].set_index("Bundesland")["Oper insgesamt"]

    diff = (end - start).dropna().reset_index()
    diff.columns = ["Bundesland", "Delta"]
    diff = diff.sort_values("Delta")

    # colors (keep yours)
    colors = ["#9ca3af" if x < 0 else "#1a80bb" for x in diff["Delta"]]

    # labels (German thousands + plus sign)
    def fmt_de(x):
        s = f"{int(round(x)):,}".replace(",", ".")
        return f"+{s}" if x > 0 else s

    diff["Label"] = diff["Delta"].apply(fmt_de)

    fig = go.Figure(
        go.Bar(
            x=diff["Delta"],
            y=diff["Bundesland"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="white", width=0.8),  # subtle separation
            ),
            text=diff["Label"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Δ Opfer: %{x:,}<extra></extra>",
        )
    )

    # zero reference line (clean diverging look)
    fig.add_vline(x=0, line_width=2, line_color="#0f172a", opacity=0.9)

    # symmetric-ish range so positives/negatives feel balanced
    max_abs = float(np.max(np.abs(diff["Delta"]))) if len(diff) else 1.0
    pad = max_abs * 0.18

    fig.update_layout(
        title=f"Veränderung der Opferzahlen {first} → {last}",
        height=550,
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=120, r=70, t=70, b=40),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif",
                  size=12, color="#0f172a"),
    )

    fig.update_xaxes(
        title="Differenz der Opferzahlen",
        range=[-max_abs - pad, max_abs + pad],
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        tickformat=",",  # keeps nice ticks
    )
    fig.update_yaxes(
        title="",
        showgrid=False,
    )

    return fig


def fig_gender(d):
    """
    Diverging dot plot (mirrored lollipop) in the same visual style as the reference image:
    - Left side: male victims (negative)
    - Right side: female victims (positive)
    - Color-coded stems (same color as dots)
    - Category labels placed on the center spine (x=0)
    """
    if d.empty:
        return empty_fig("Keine Daten verfügbar")

    # Exclude totalsmarker_color
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"].copy()
    if d2.empty:
        return empty_fig("Keine Deliktsdaten verfügbar")

    # Guard against missing columns
    required_cols = ["Opfer maennlich", "Opfer weiblich"]
    for c in required_cols:
        if c not in d2.columns:
            return empty_fig(f"Spalte fehlt: {c}")

    # Aggregate by crime type
    g = (
        d2.groupby("Straftat_kurz")[["Opfer maennlich", "Opfer weiblich"]]
        .sum()
        .reset_index()
    )

    # Keep Top 12 crime types by total victims (readability)
    g["Total"] = g["Opfer maennlich"] + g["Opfer weiblich"]
    g = g.sort_values("Total", ascending=False).head(12).copy()

    # Diverging values
    g["Male_neg"] = -g["Opfer maennlich"]

    # Reverse order so that smallest total at top, largest at bottom (Einfache KV at bottom)
    g = g.sort_values("Total", ascending=False)

    fig = go.Figure()

    # Male bars (left / negative)
    fig.add_trace(
        go.Bar(
            x=-g["Opfer maennlich"],
            y=g["Straftat_kurz"],
            orientation="h",
            name="Männlich",
            marker_color="#1a80bb",  # blue  (men)
            hovertemplate="<b>%{y}</b><br>Männlich: %{customdata:,}<extra></extra>",
            customdata=g["Opfer maennlich"],
        )
    )

    # Female bars (right / positive)
    fig.add_trace(
        go.Bar(
            x=g["Opfer weiblich"],
            y=g["Straftat_kurz"],
            orientation="h",
            name="Weiblich",
            marker_color="#b8b8b8",  # grey (women)
            hovertemplate="<b>%{y}</b><br>Weiblich: %{x:,}<extra></extra>",
        )
    )

    # Center reference line
    fig.add_vline(
        x=0,
        line_width=3,
        line_color="#0f172a",
    )

    # Symmetric x-range
    max_x = float(max(g["Opfer weiblich"].max(), g["Opfer maennlich"].max()))
    pad = max_x * 0.15

    fig.update_layout(
        title="Geschlechtervergleich nach Deliktsgruppe",
        barmode="overlay",
        xaxis=dict(
            title="Opferzahl nach Geschlecht",
            range=[-(max_x + pad), (max_x + pad)],
            showgrid=True,
            gridcolor="#e5e7eb",
            zeroline=False,
            tickvals=[-max_x, -max_x/2, 0, max_x/2, max_x],
            ticktext=[
                f"{int(max_x):,}",
                f"{int(max_x/2):,}",
                "0",
                f"{int(max_x/2):,}",
                f"{int(max_x):,}",
            ],
        ),
        yaxis=dict(title="", showgrid=False),
        height=550,
        plot_bgcolor="white",
        margin=dict(l=140, r=80, t=70, b=50),
        legend=dict(
            orientation="v",
            x=1.02,
            y=1,
            yanchor="top",
        ),
    )

    return fig


# --------- DASH APP ---------
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.LUX],
    suppress_callback_exceptions=True,
)
app.title = "Crime Analysis Dashboard"

# --------- SIDEBAR ---------


def sidebar_layout(path):
    def nav_link(label, href):
        return dbc.NavLink(
            label,
            href=href,
            active=(path == href or (
                href == "/overview" and path in ("/", None))),
            className="w-100 text-start mb-1",
        )

    return html.Div(
        style=SIDEBAR_STYLE,
        children=[
            # --- Navigation FIRST ---
            dbc.Card(
                body=True,
                style={"marginBottom": "12px"},
                children=[
                    html.H5("Navigation", className="card-title"),
                    dbc.Nav(
                        [
                            nav_link("Übersicht", "/overview"),
                            nav_link("Geografisch", "/geo"),
                            nav_link("Deliktskategorien", "/crime"),
                            nav_link("Zeitliche Einblicke", "/temporal"),
                        ],
                        vertical=True,
                        pills=True,
                    ),
                ],
            ),
            # --- Analysefilter BELOW ---
            dbc.Card(
                body=True,
                children=[
                    html.H5("Analysefilter", className="card-title"),
                    html.Label("Metrik", className="mt-2"),
                    dcc.Dropdown(
                        id="filter-metric",
                        options=[
                            {"label": "Absolute Opfer", "value": "abs"},
                            {"label": "Opfer pro 100.000 Einwohner", "value": "rate"},
                        ],
                        value="abs",
                        clearable=False,
                    ),

                    html.Label("Jahr(e)", className="mt-2"),
                    dcc.Dropdown(
                        id="filter-year",
                        options=[{"label": str(y), "value": y} for y in YEARS],
                        value=YEARS,
                        multi=True,
                    ),
                    html.Label("Deliktsgruppen", className="mt-3"),
                    dcc.Dropdown(
                        id="filter-crime",
                        options=[{"label": c, "value": c}
                                 for c in CRIME_SHORT],
                        multi=True,
                        value=[],
                    ),
                    html.Label("Bundesland", className="mt-3"),
                    dcc.Dropdown(
                        id="filter-state",
                        options=[{"label": s, "value": s} for s in STATES],
                        multi=True,
                        value=[],
                    ),
                ],
            ),
        ],
    )


# --------- PAGE LAYOUTS ---------
def layout_overview():
    return html.Div(
        children=[
            html.H2("Übersicht", className="mb-3"),
            html.P(
                "Überblick über zentrale Kennzahlen, Trends und die Verteilung nach Deliktsgruppen.",
                className="text-muted",
            ),
            html.Div(
                style=KPI_GRID_STYLE,
                children=[
                    html.Div(
                        style={**CARD_STYLE, "borderLeft": "6px solid #1e1b4b"},
                        children=[
                            html.Div("Gesamtzahl der Opfer",
                                     style=KPI_LABEL_STYLE),
                            html.Div(id="kpi-total-victims",
                                     style=KPI_VALUE_STYLE),
                        ],
                    ),
                    html.Div(
                        style={**CARD_STYLE, "borderLeft": "6px solid #1e1b4b"},
                        children=[
                            html.Div("Opfer pro Jahr (Ø)",
                                     style=KPI_LABEL_STYLE),
                            html.Div(
                                id="kpi-victims-per-year", style=KPI_VALUE_STYLE
                            ),
                        ],
                    ),
                    html.Div(
                        style={**CARD_STYLE, "borderLeft": "6px solid #1e1b4b"},
                        children=[
                            html.Div(
                                "Verhältnis männlich / weiblich", style=KPI_LABEL_STYLE
                            ),
                            html.Div(id="kpi-male-female",
                                     style=KPI_VALUE_STYLE),
                        ],
                    ),
                    html.Div(
                        style={**CARD_STYLE, "borderLeft": "6px solid #1e1b4b"},
                        children=[
                            html.Div("Unter 18 / Erwachsene",
                                     style=KPI_LABEL_STYLE),
                            html.Div(id="kpi-under18-adults",
                                     style=KPI_VALUE_STYLE),
                        ],
                    ),
                    html.Div(
                        style={**CARD_STYLE, "borderLeft": "6px solid #1e1b4b"},
                        children=[
                            html.Div("Stadt mit höchster Kriminalitätsrate",
                            style=KPI_LABEL_STYLE),
                             html.Div(id="kpi-crime-types",
                            style=KPI_VALUE_STYLE),
                        ],
                    ),
                ],
            ),
            dcc.Graph(id="trend"),

           
            dcc.Graph(id="population-correlation"),

            html.Br(),
            dcc.Graph(id="gender"),
            html.Br(),
            dcc.Graph(id="crime-pie"),
        ]
    )

# KPI : City with highest victim rate
def kpi_top_city_rate(d: pd.DataFrame) -> str:
    """City/Region with the highest victim rate (per 100,000), within current filters."""
    if d is None or d.empty:
        return "–"

    d2 = d.copy()

    # remove totals row if present
    if "Straftat_kurz" in d2.columns:
        d2 = d2[d2["Straftat_kurz"] != "Straftaten insgesamt"].copy()

    if d2.empty:
        return "–"

    victim_col = "Oper insgesamt" if "Oper insgesamt" in d2.columns else None
    if victim_col is None:
        return "–"

    # Prefer Region + Bundesland (avoids duplicates like Neustadt)
    group_keys = []
    if "Region" in d2.columns:
        group_keys.append("Region")
    if "Bundesland" in d2.columns:
        group_keys.append("Bundesland")
    if not group_keys:
        return "–"

    victims = (
        d2.groupby(group_keys, dropna=False)[victim_col]
        .sum()
        .reset_index(name="victims")
    )

    # Try to compute per-100k using a population column if available
    pop_candidates = [
        "Einwohner", "Einwohnerzahl", "Bevoelkerung", "Bevölkerung", "Population", "pop"
    ]
    pop_col = next((c for c in pop_candidates if c in d2.columns), None)

    if pop_col is not None:
        pop = (
            d2.groupby(group_keys, dropna=False)[pop_col]
            .max()
            .reset_index(name="population")
        )
        merged = victims.merge(pop, on=group_keys, how="left")
        merged["rate_per_100k"] = np.where(
            merged["population"] > 0,
            merged["victims"] / merged["population"] * 100000.0,
            np.nan,
        )
    else:
        # If we don't have city-population columns, try to approximate using STATE population
        # (Bundesland population per selected year(s)) from population_states.csv.
        if "Bundesland" in d2.columns and "Jahr" in d2.columns:
            years_by_state = (
                d2[["Bundesland", "Jahr"]]
                .drop_duplicates()
                .copy()
            )
            years_by_state["Population_state"] = years_by_state.apply(
                lambda r: get_population(str(r["Bundesland"]), int(r["Jahr"])),
                axis=1,
            )
            pop_state = (
                years_by_state.dropna(subset=["Population_state"])
                .groupby("Bundesland")["Population_state"]
                .sum()
                .reset_index(name="population")
            )
            merged = victims.merge(pop_state, on="Bundesland", how="left")
            merged["rate_per_100k"] = np.where(
                merged["population"] > 0,
                merged["victims"] / merged["population"] * 100000.0,
                np.nan,
            )
        else:
            merged = victims.copy()
            merged["rate_per_100k"] = np.nan

        # If we still couldn't compute a rate, look for an existing rate column.
        rate_candidates = [
            "Opfer pro 100.000 Einwohner",
            "Opfer pro 100.000",
            "Opfer je 100.000",
            "Rate",
            "rate",
            "Victim_rate",
            "Opfer_rate",
        ]
        rate_col = next((c for c in rate_candidates if c in d2.columns), None)

        if rate_col is None:
            # Last resort: cannot compute rate → show city with highest absolute victims
            top = victims.sort_values("victims", ascending=False).head(1)
            if top.empty:
                return "–"
            city = str(top.iloc[0].get("Region", ""))
            bl = str(top.iloc[0].get("Bundesland", ""))
            return f"{city} ({bl})" if bl and bl != "nan" else city

        rate = (
            d2.groupby(group_keys, dropna=False)[rate_col]
            .mean()
            .reset_index(name="rate_per_100k")
        )
        merged = victims.merge(rate, on=group_keys, how="left")

    merged = merged.dropna(subset=["rate_per_100k"]).copy()
    if merged.empty:
        return "–"

    top = merged.sort_values("rate_per_100k", ascending=False).head(1)
    city = str(top.iloc[0].get("Region", ""))
    bl = str(top.iloc[0].get("Bundesland", ""))
    rate_val = float(top.iloc[0]["rate_per_100k"])

    # German formatting 1.234,5
    rate_str = f"{rate_val:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")

    if bl and bl != "nan":
        return f"{city} ({bl}) • {rate_str} /100k"
    return f"{city} • {rate_str} /100k"


def layout_geo():
    return html.Div(
        children=[
            html.H2("Geografische Analyse", className="mb-3"),

            # Store bleibt, weil wir weiterhin per Klick Bundesland auswählen
            dcc.Store(id="selected-state-store", data=None),

            # ===== FILTERLEISTE ÜBER DER KARTE =====
            html.Div(
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "20px",
                    "padding": "12px",
                    "backgroundColor": "#ffffff",  # same as main page
                    "borderRadius": "8px",
                    "border": "none",              # remove differentiation
                    "width": "100%",
                },
                children=[
                    # --- Ansicht / Anzahl Städte ---
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Ansicht"),
                            html.Div(
                                style={"display": "flex",
                                       "alignItems": "center", "gap": "12px"},
                                children=[
                                    # Back button (appears only on Landkreis level)
                                    html.Div(
                                        id="state-back-button",
                                        style={"display": "none"},
                                        children=[
                                            html.Button(
                                                "← Zurück",
                                                id="back-to-germany",
                                                n_clicks=0,
                                                style={
                                                    "backgroundColor": HEADER_BG,
                                                    "color": "white",
                                                    "border": "none",
                                                    "padding": "6px 10px",
                                                    "borderRadius": "6px",
                                                    "cursor": "pointer",
                                                },
                                            )
                                        ],
                                    ),
                                    # Ansicht radios
                                    dcc.RadioItems(
                                        id="geo-city-mode",
                                        options=[
                                            {"label": " Bundesländer",
                                                "value": "bundesland"},
                                            {"label": " Städte", "value": "all"},
                                        ],
                                        value="bundesland",
                                        labelStyle={
                                            "display": "inline-block",
                                            "marginRight": "18px",
                                            "fontWeight": "600",
                                        },
                                        inputStyle={
                                            "marginRight": "6px",
                                            "accentColor": HEADER_BG,
                                        },
                                    ),
                                ],
                            ),
                        ],
                    ),

                    # --- Altersgruppe ---
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            html.Label("Altersgruppe"),
                            dcc.Dropdown(
                                id="geo-age-group",
                                options=(
                                    [{"label": "Alle", "value": "all"}]
                                    + [
                                        {
                                            "label": label.split()[1] if "<" in label or "+" in label else label,
                                            "value": label,
                                        }
                                        for label in AGE_COLS.keys()
                                    ]
                                ),
                                value="all",
                                clearable=False,
                            ),
                        ],
                    ),

                    # --- Modus + Aktuelle Ansicht (NO back button here) ---
                    html.Div(
                        style={"flex": "1", "display": "flex",
                               "flexDirection": "column"},
                        children=[
                            html.Label("Modus"),
                            html.Div(
                                style={
                                    "display": "flex",
                                    "alignItems": "center",
                                    "gap": "14px",
                                },
                                children=[
                                    dcc.RadioItems(
                                        id="geo-safety-mode",
                                        options=[
                                            {"label": " Gefährlich",
                                                "value": "unsafe"},
                                            {"label": " Sicher", "value": "safe"},
                                        ],
                                        value="unsafe",
                                        labelStyle={
                                            "display": "inline-block",
                                            "marginRight": "16px",
                                            "fontWeight": "600",
                                        },
                                        inputStyle={
                                            "marginRight": "6px",
                                            "accentColor": HEADER_BG,
                                        },
                                    ),
                                    # Current view display (kept)
                                    html.Div(
                                        id="current-state-display",
                                        children="",
                                        style={"display": "none"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # ===== ENDE FILTERLEISTE =====

            # TOP ROW: Map (left) + placeholder for new chart (right)
            dbc.Row(
                [
                    # LEFT: Map preview (~55%)
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="map",
                                style={"height": "420px"},
                                config={"displayModeBar": False},
                            ),
                        ],
                        md=7,
                        xs=12,
                    ),

                    # RIGHT: Placeholder for future chart (~45%)
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="geo-new-chart",
                                style={"height": "420px"},
                                config={"displayModeBar": False},
                            ),

                        ],
                        md=5,
                        xs=12,
                    ),
                ],
                className="g-3",
            ),

            # BOTTOM SECTION: Wide stacked charts (bar on top, lollipop below)
            html.Div(style={"height": "36px"}),

            # Full-width bar chart (top)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="geo-crime-profile",
                                style={"height": "550px"},
                                config={"displayModeBar": False},
                            ),
                        ],
                        md=12,
                        xs=12,
                    ),
                ],
                className="g-3",
            ),


            # Full-width lollipop chart (below the bar chart)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dcc.Graph(
                                id="geo-trend",
                                style={
                                    "height": "650px", "marginBottom": "0px", "paddingBottom": "0px"},
                                config={"displayModeBar": False},
                            ),
                        ],
                        md=12,
                        xs=12,
                        style={"marginBottom": "0px", "paddingBottom": "0px"},
                    ),
                ],
                className="g-0",
                style={"marginBottom": "0px",
                       "paddingBottom": "0px", "--bs-gutter-y": "0px"},
            ),


        ],
        style={"marginBottom": "0px", "paddingBottom": "0px"},
    )


def layout_crime():
    return html.Div(
        children=[
            html.H2("Crime Types (Deliktsstruktur)", className="mb-3"),
            html.P(
                "Analyse der Opferzahlen nach Deliktsgruppen sowie der Altersstruktur der Opfer.",
                className="text-muted",
            ),

            dcc.Graph(
                id="top5-crime", style={"width": "100%", "height": f"{STANDARD_HEIGHT}px"}),
            html.Br(),
            dcc.Graph(id="heat", style={"width": "100%"}, config={
                      "responsive": True}),
            html.Br(),
            html.Div(
                style={"maxWidth": "500px"},
                children=[
                    html.Label("Deliktsgruppe für Altersanalyse"),
                    dcc.Dropdown(
                        id="age-crime",
                        options=[{"label": c, "value": c}
                                 for c in CRIME_SHORT],
                        value="Straftaten insgesamt",
                        clearable=False,
                    ),
                ],
            ),
            html.Br(),

            dcc.Graph(id="agechart", style={
                      "width": "100%", "height": f"{STANDARD_HEIGHT}px"}),
        ]
    )


def layout_temporal():
    return html.Div(
        children=[
            html.H2("Zeitliche Einblicke", className="mb-3"),
            html.P(
                "Dynamik der Opferzahlen im Ländervergleich sowie geschlechtsspezifische Muster.",
                className="text-muted",
            ),
            dcc.Graph(id="overview-bubble"),
            html.Br(),
            dcc.Graph(id="diverg"),
            html.Br(),
            dcc.Graph(id="trendstates"),
        ]
    )

# Def Layout_Trend is Removed


# Which city is safer or dangerous for children function (now as risk scatter)


# --------- Helper: Bar chart for children 0–14 Top-N ---------


# viollence agains Women over time


def fig_violence_women(d):
    """
    Multi-line trend chart:
    - One line per crime type (Straftat_kurz)
    - Distinct line styles & markers
    - Same clean, official-style look
    """
    if d.empty:
        return empty_fig("Keine Daten verfügbar")

    # Exclude total category
    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"].copy()
    if d2.empty:
        return empty_fig("Keine Deliktsdaten verfügbar")

    # Aggregate female victims per year & crime type
    g = (
        d2.groupby(["Jahr", "Straftat_kurz"])["Opfer weiblich"]
        .sum()
        .reset_index()
    )

    crime_types = g["Straftat_kurz"].unique().tolist()

    # Predefined styles (cycled if more crime types exist)
    colors = px.colors.qualitative.Dark24
    markers = [
        "circle", "square", "diamond", "triangle-up",
        "triangle-down", "cross", "x", "star",
        "hexagon", "pentagon",
    ]
    line_dashes = ["solid", "dash", "dot", "dashdot"]

    fig = go.Figure()

    for i, crime in enumerate(crime_types):
        df_c = g[g["Straftat_kurz"] == crime]

        fig.add_trace(
            go.Scatter(
                x=df_c["Jahr"],
                y=df_c["Opfer weiblich"],
                mode="lines+markers",
                name=crime,
                line=dict(
                    color=colors[i % len(colors)],
                    width=3,
                    dash=line_dashes[i % len(line_dashes)],
                ),
                marker=dict(
                    symbol=markers[i % len(markers)],
                    size=9,
                    color=colors[i % len(colors)],
                ),
                hovertemplate="<b>%{x}</b><br>%{y:,} Fälle<extra></extra>",
            )
        )

    fig.update_layout(
        title="Steigt die Gewalt gegen Frauen an? – Entwicklung nach Deliktsart",
        xaxis_title="",
        yaxis_title="erfasste Fälle (weibliche Opfer)",
        height=STANDARD_HEIGHT,
        plot_bgcolor="white",
        margin=dict(l=60, r=30, t=70, b=120),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
        ),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#d1d5db",
        tickmode="linear",
        dtick=1,
        tickangle=-90,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#d1d5db")

    return fig

# which crime types are growing fastest


def fig_fastest_growing_crimes(d):
    if d.empty:
        return empty_fig("Keine Daten verfügbar")

    d2 = d[d["Straftat_kurz"] != "Straftaten insgesamt"].copy()

    years = sorted(d2["Jahr"].unique())
    if len(years) < 2:
        return empty_fig("Mindestens zwei Jahre notwendig (2019 & 2024).")

    start, end = years[0], years[-1]

    # Sum victims by crime type for first and last year
    g = (
        d2.groupby(["Straftat_kurz", "Jahr"])["Oper insgesamt"]
        .sum()
        .reset_index()
    )

    start_vals = g[g["Jahr"] == start].set_index("Straftat_kurz")[
        "Oper insgesamt"]
    end_vals = g[g["Jahr"] == end].set_index("Straftat_kurz")["Oper insgesamt"]

    df_growth = (end_vals - start_vals).dropna().reset_index()
    df_growth.columns = ["Straftat_kurz", "Wachstum"]

    df_growth = df_growth.sort_values("Wachstum", ascending=False)

    fig = px.bar(
        df_growth,
        x="Wachstum",
        y="Straftat_kurz",
        orientation="h",
        color="Wachstum",
        color_continuous_scale="YlOrRd",
        labels={"Wachstum": f"Zunahme der Opfer {start}–{end}"},
        title=f"Am schnellsten wachsende Deliktsgruppen ({start}–{end})",
    )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(coloraxis_showscale=False)

    return fig


# --------- ROOT LAYOUT (HEADER + SIDEBAR + CONTENT) ---------
app.layout = html.Div(
    children=[
        html.Div(
            style={
                "backgroundColor": HEADER_BG,
                "padding": "22px 30px",
                "paddingBottom": "30px",
                "borderBottom": f"1px solid {HEADER_BORDER}",
                "boxShadow": "0px 4px 10px rgba(0,0,0,0.25)",
                "position": "fixed",
                "top": 0,
                "left": 0,
                "right": 0,
                "zIndex": 1000,
                "textAlign": "center",
                "fontFamily": "Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
            },
            children=[
                html.H1(
                    "Crime Analysis Dashboard",
                    style={
                        "fontSize": "28px",
                        "fontWeight": "700",
                        "color": HEADER_TEXT_MAIN,
                        "marginBottom": "4px",
                        "textAlign": "center",
                    },
                ),
                html.H4(
                    "Polizeiliche Kriminalstatistik Deutschland (2019–2024)",
                    style={
                        "fontSize": "18px",
                        "fontWeight": "450",
                        "color": HEADER_TEXT_SUB,
                        "marginTop": "0px",
                        "textAlign": "center",
                    },
                ),
                # Toggle button for sidebar (☰)
                html.Button(
                    "☰",
                    id="toggle-sidebar",
                    n_clicks=0,
                    title="Sidebar ein-/ausblenden",
                    style={
                        "position": "absolute",
                        "top": "30px",
                        "left": "30px",
                        "fontSize": "22px",
                        "background": "transparent",
                        "border": "none",
                        "color": "white",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
        dcc.Location(id="url"),
        dcc.Store(id="sidebar-visible", data=True),
        html.Div(id="sidebar"),
        html.Div(id="page-content", style=CONTENT_STYLE),
    ]
)


# --------- NAVIGATION CALLBACKS ---------
@app.callback(Output("sidebar", "children"), Input("url", "pathname"))
def update_sidebar(path):
    return sidebar_layout(path)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(path):
    if path in ("/", "/overview", None):
        return layout_overview()
    if path == "/geo":
        return layout_geo()
    if path == "/crime":
        return layout_crime()
    if path == "/correlations":
        return layout_correlations()
    if path == "/temporal":
        return layout_temporal()
    return html.Div([html.H2("404 – Seite nicht gefunden")])

# --------- SIDEBAR TOGGLE CALLBACKS ---------
# Toggle sidebar visibility store


@app.callback(
    Output("sidebar-visible", "data"),
    Input("toggle-sidebar", "n_clicks"),
    State("sidebar-visible", "data"),
)
def toggle_sidebar(n_clicks, visible):
    # Keep sidebar visible on initial load
    if not callback_context.triggered:
        return visible

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger == "toggle-sidebar" and n_clicks and n_clicks > 0:
        return not visible

    return visible

# Dynamically update sidebar and content styles


@app.callback(
    Output("sidebar", "style"),
    Output("page-content", "style"),
    Input("sidebar-visible", "data"),
)
def update_sidebar_visibility(visible):
    if visible:
        return SIDEBAR_STYLE, CONTENT_STYLE
    else:
        hidden_sidebar = SIDEBAR_STYLE.copy()
        hidden_sidebar["display"] = "none"
        expanded_content = CONTENT_STYLE.copy()
        expanded_content["marginLeft"] = "0px"
        return hidden_sidebar, expanded_content


# --------- OVERVIEW CALLBACK ---------
# --------- CALLBACKS: ÜBERSICHT (KPIs + Trend + Bubble + Pie) ---------
@app.callback(
    [
        Output("kpi-total-victims", "children"),
        Output("kpi-victims-per-year", "children"),
        Output("kpi-male-female", "children"),
        Output("kpi-under18-adults", "children"),
        Output("kpi-crime-types", "children"),
        Output("trend", "figure"),
        Output("gender", "figure"),
        Output("crime-pie", "figure"),
        Output("population-correlation", "figure"),
    ],
    [
        Input("filter-year", "value"),
        Input("filter-crime", "value"),
        Input("filter-state", "value"),
        Input("filter-metric", "value"),  # ✅ NEW
    ],
)
def update_overview(years, crimes, states, metric_mode):
    """
    Updates the Übersicht page (KPIs + charts) based on the global sidebar filters.
    IMPORTANT: This must be the ONLY callback that outputs crime-pie.figure.
    """

    years = YEARS if not years else years
    crimes = [] if crimes is None else crimes
    states = [] if states is None else states
    metric_mode = metric_mode or "abs"

    d_filtered = filter_data(years, crimes, states)
    fig_corr = fig_population_correlation(d_filtered)

    # KPIs (still absolute — KPIs usually should not be rate-based)
    kpi1, kpi2, kpi3, kpi4, kpi5 = build_kpis(d_filtered)

    # Figures
    trend_fig = fig_trend(d_filtered, metric_mode=metric_mode)  # ✅ NEW
    gender_fig = fig_gender(d_filtered)
    pie_fig = fig_crime_pie(d_filtered)

    return (
        kpi1,
        kpi2,
        kpi3,
        kpi4,
        kpi5,
        trend_fig,
        gender_fig,
        pie_fig,
        fig_corr,
    )


# --------- GEOGRAPHIC CALLBACKS ---------


@app.callback(
    Output("selected-state-store", "data"),
    Input("map", "clickData"),
    Input("back-to-germany", "n_clicks"),
    Input("filter-state", "value"),
    State("selected-state-store", "data"),
)
def update_selected_state(click_data, back_clicks, filter_states, current_state):
    """Handle state selection logic"""
    ctx = callback_context

    if not ctx.triggered:
        return current_state

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Reset if back button clicked or filter changed
    if trigger_id == 'back-to-germany' or trigger_id == 'filter-state':
        return None

    # Only process map clicks if we're at state level (not city level)
    if trigger_id == 'map' and click_data and current_state is None:
        # We're at state level, so process the click
        try:
            if click_data and 'points' in click_data and click_data['points']:
                point = click_data['points'][0]
                # Try to get state name from different possible locations
                if 'hovertext' in point:
                    return point['hovertext']
                elif 'location' in point:
                    return point['location']
                elif 'customdata' in point and point['customdata']:
                    return point['customdata'][0]
        except Exception as e:
            print(f"Error processing click: {e}")

    # If we're already in a state view (current_state is not None),
    # clicking on the map should do nothing
    return current_state


@app.callback(
    Output("map", "figure"),
    Output("geo-new-chart", "figure"),
    Output("geo-trend", "figure"),
    Output("geo-crime-profile", "figure"),
    Output("current-state-display", "children"),
    Output("state-back-button", "style"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
    Input("selected-state-store", "data"),
    Input("geo-city-mode", "value"),
    Input("geo-age-group", "value"),
    Input("geo-safety-mode", "value"),
    Input("filter-metric", "value"),
)
def update_geo_components(
    years, crimes, states, selected_state, city_mode, age_group, safety_mode, metric_mode
):
    d = filter_data(years or YEARS, crimes or [], states or [])

    map_fig = fig_geo_map(
        d,
        selected_state=selected_state,
        city_mode=city_mode,
        age_group=age_group,
        safety_mode=safety_mode,
        metric_mode=metric_mode,
    )

    state_bar_fig = fig_geo_state_bar(d)
    top_regions_fig = fig_geo_top(d)

    # New chart: Top-5 crime categories (trend) for selected state
    if selected_state:
        df_state = d[d["Bundesland"] == selected_state]
        if not df_state.empty:
            geo_new_fig = fig_state_top5_trend(
                df_state, metric_mode=metric_mode)
        else:
            geo_new_fig = empty_fig(f"Keine Daten für {selected_state}")
    else:
        geo_new_fig = empty_fig(
            "Klicke auf ein Bundesland, um die Top-5 Deliktsgruppen anzuzeigen")

    # Update info text
    if selected_state:
        text = f"Aktuelle Ansicht: {selected_state} – Städteansicht"
        back_style = {"display": "block"}
    else:
        text = (
            "Aktuelle Ansicht: Deutschland – Bundesländer"
            if city_mode == "bundesland"
            else "Aktuelle Ansicht: Deutschland – Städte"
        )
        back_style = {"display": "none"}

    return map_fig, geo_new_fig, state_bar_fig, top_regions_fig, text, back_style


# --------- CRIME TYPES CALLBACK ---------
@app.callback(
    Output("heat", "figure"),
    Output("agechart", "figure"),
    Output("top5-crime", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
    Input("age-crime", "value"),
)
def update_crime(years, crimes, states, age_crime_sel):
    d = filter_data(years or YEARS, crimes or [], states or [])

    heat_fig = fig_heatmap(d)
    stacked_fig = fig_stacked(d)
    age_fig = fig_age(d, age_crime_sel)
    top5_fig = fig_top5(d)

    return heat_fig, age_fig, top5_fig


# --------- TEMPORAL CALLBACK ---------


@app.callback(
    Output("trendstates", "figure"),
    Output("diverg", "figure"),
    Output("overview-bubble", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
)
def update_temporal(years, crimes, states):
    d = filter_data(years or YEARS, crimes or [], states or [])
    return fig_state_trend(d), fig_diverg(d), fig_overview_bubble(d)


if __name__ == "__main__":
    app.run(debug=True)

# --------- GEO FULL ANALYTICAL CALLBACK ---------

# This callback reads selected-state-store and updates map + the two analytical charts (geo-crime-profile, geo-trend)


@app.callback(
    Output("map", "figure"),
    Output("geo-new-chart", "figure"),
    Output("geo-crime-profile", "figure"),
    Output("geo-trend", "figure"),
    Input("filter-year", "value"),
    Input("filter-crime", "value"),
    Input("filter-state", "value"),
    Input("geo-city-mode", "value"),
    Input("geo-age-group", "value"),
    Input("geo-safety-mode", "value"),
    Input("selected-state-store", "data"),
    Input("filter-metric", "value"),
)
def update_geo_full_analysis(
    years, crimes, states,
    geo_city_mode, geo_age_group, geo_safety_mode,
    selected_state, metric_mode
):
    # Apply global filters
    d_filtered = filter_data(years, crimes, states)

    # Decide drill level
    if geo_city_mode == "bundesland" or selected_state is None:
        map_fig = fig_geo_map(
            d_filtered,
            selected_state=None,
            city_mode="bundesland",
            age_group=geo_age_group,
            safety_mode=geo_safety_mode,
            metric_mode=metric_mode,
        )
        side_df = d_filtered
    else:
        map_fig = fig_geo_map(
            d_filtered,
            selected_state=selected_state,
            city_mode="all",
            age_group=geo_age_group,
            safety_mode=geo_safety_mode,
            metric_mode=metric_mode,
        )
        side_df = d_filtered[d_filtered["Bundesland"] == selected_state]

    # New chart: Top-5 crime categories (trend) for selected state (or hint)
    if selected_state:
        df_state = d_filtered[d_filtered["Bundesland"] == selected_state]
        if not df_state.empty:
            geo_new_fig = fig_state_top5_trend(
                df_state, metric_mode=metric_mode)
        else:
            geo_new_fig = empty_fig(f"Keine Daten für {selected_state}")
    else:
        geo_new_fig = empty_fig(
            "Klicke auf ein Bundesland, um die Top-5 Deliktsgruppen anzuzeigen")

    # Analytical charts (structure + trend)
    crime_profile_fig = fig_top5(side_df)
    trend_fig = fig_trend(side_df, metric_mode=metric_mode)

    return map_fig, geo_new_fig, crime_profile_fig, trend_fig

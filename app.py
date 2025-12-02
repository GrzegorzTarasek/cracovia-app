import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO

# ===================== STAŁE =====================
POS_OPTIONS = [
    "ŚO", "LŚO", "PŚO", "LO",
    "ŚPD", "8", "ŚP", "ŚPO", "10", "PW", "LW", "WAHADŁO", "NAPASTNIK",
]
DEFAULT_TEAMS = ["C1", "C2", "U-19", "U-17"]

# ===================== DANE Z GITHUBA (CSV) =====================
# Wszystko czytamy z katalogu: https://github.com/GrzegorzTarasek/cracovia-app/tree/main/dane
BASE_CSV_URL = "https://raw.githubusercontent.com/GrzegorzTarasek/cracovia-app/main/dane/"

# Prosty cache w pamięci (żeby nie pobierać CSV za każdym razem)
TABLES: dict[str, pd.DataFrame] = {}


def _load_csv(fname: str) -> pd.DataFrame:
    url = BASE_CSV_URL + fname
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.DataFrame()
    return df


@st.cache_data(show_spinner=False)
def load_motoryka_table() -> pd.DataFrame:
    """
    motoryka_stats.csv – motoryka
    """
    if "motoryka_stats" in TABLES:
        return TABLES["motoryka_stats"]
    df = _load_csv("motoryka_stats.csv")
    TABLES["motoryka_stats"] = df
    return df


@st.cache_data(show_spinner=False)
def load_fantasy_table() -> pd.DataFrame:
    """
    fantasypasy.csv – odpowiednik fantasypasy_stats
    """
    if "fantasypasy_stats" in TABLES:
        return TABLES["fantasypasy_stats"]
    df = _load_csv("fantasypasy.csv")
    TABLES["fantasypasy_stats"] = df
    return df


@st.cache_data(show_spinner=False)
def load_players_table() -> pd.DataFrame:
    """
    players_table.csv – lista zawodników
    """
    if "players" in TABLES:
        return TABLES["players"]
    df = _load_csv("players_table.csv")
    TABLES["players"] = df
    return df


@st.cache_data(show_spinner=False)
def load_teams_table() -> pd.DataFrame:
    """
    teams.csv – lista zespołów
    """
    if "teams" in TABLES:
        return TABLES["teams"]
    df = _load_csv("teams.csv")
    TABLES["teams"] = df
    return df


@st.cache_data(show_spinner=False)
def load_periods_table() -> pd.DataFrame:
    """
    measurement_periods.csv – okresy pomiarowe
    """
    if "measurement_periods" in TABLES:
        return TABLES["measurement_periods"]
    df = _load_csv("measurement_periods.csv")
    TABLES["measurement_periods"] = df
    return df


# ===================== EMULACJA fetch_df(...) NA CSV =====================
def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    Zastępuje wcześniejsze zapytania SQL na bazie – na podstawie tekstu SELECT
    zwraca odpowiedni DataFrame z CSV.
    Rozpoznajemy tylko kwerendy, które faktycznie występują w kodzie.
    """
    if sql is None:
        return pd.DataFrame()

    params = params or {}
    s = " ".join(sql.split()).lower()

    # ---- LISTA ZESPOŁÓW ----
    # SELECT Team FROM teams ORDER BY Team;
    if "from teams" in s and "select" in s and "team" in s:
        df = load_teams_table().copy()
        if "Team" not in df.columns:
            return pd.DataFrame(columns=["Team"])
        return (
            df[["Team"]]
            .dropna()
            .drop_duplicates()
            .sort_values("Team")
        )

    # ---- LISTA ZAWODNIKÓW (tylko Name) ----
    # SELECT Name FROM players ORDER BY Name;
    if "from players" in s and "select" in s and "name" in s and "team" not in s:
        df = load_players_table().copy()
        if "Name" not in df.columns:
            return pd.DataFrame(columns=["Name"])
        return (
            df[["Name"]]
            .dropna()
            .drop_duplicates()
            .sort_values("Name")
        )

    # ---- LISTA ZAWODNIKÓW Z TEAM & POSITION ----
    # SELECT Name, Team, Position FROM players ORDER BY Name;
    if "from players" in s and "select" in s and "team" in s:
        df = load_players_table().copy()
        cols = [c for c in ["Name", "Team", "Position"] if c in df.columns]
        if not cols:
            return pd.DataFrame()
        return (
            df[cols]
            .dropna(subset=["Name"])
            .sort_values("Name")
        )

    # ---- OKRESY POMIAROWE ----
    # SELECT PeriodID, Label, DateStart, DateEnd FROM measurement_periods ...
    if "from measurement_periods" in s:
        df = load_periods_table().copy()
        cols = [c for c in ["PeriodID", "Label", "DateStart", "DateEnd"] if c in df.columns]
        if not cols:
            return pd.DataFrame()
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        df = df[cols].dropna(subset=["DateStart", "DateEnd"]).drop_duplicates()
        return df.sort_values(["DateStart", "DateEnd"], ascending=[False, False])

    # ---- PARY DAT Z MOTORYKA / FANTASYPASY ----
    # SELECT DISTINCT DateStart, DateEnd FROM motoryka_stats ...
    if "distinct datestart" in s and "dateend" in s:
        if "from motoryka_stats" in s:
            df = load_motoryka_table().copy()
        elif "from fantasypasy_stats" in s:
            df = load_fantasy_table().copy()
        else:
            return pd.DataFrame()

        if "DateStart" not in df.columns or "DateEnd" not in df.columns:
            return pd.DataFrame()

        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")

        df = (
            df[["DateStart", "DateEnd"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["DateStart", "DateEnd"], ascending=[False, False])
        )
        return df

    # ---- PODSTAWOWE ZAPYTANIA DO MOTORYKA_STATS ----
    if "from motoryka_stats" in s:
        df = load_motoryka_table().copy()

        # WHERE Name = :name
        if "where name" in s and "name" in params and "Name" in df.columns:
            df = df[df["Name"] == params["name"]]

        # filtrowanie po zakresie dat – ds/de
        if ":ds" in s and "ds" in params and "DateStart" in df.columns:
            df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
            df = df[df["DateStart"] >= pd.to_datetime(params["ds"])]
        if ":de" in s and "de" in params and "DateEnd" in df.columns:
            df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
            df = df[df["DateEnd"] <= pd.to_datetime(params["de"])]

        # filtr po Team IN :teams
        if "team in :teams" in s and "teams" in params and "Team" in df.columns:
            df = df[df["Team"].isin(list(params["teams"]))]

        return df

    # ---- PODSTAWOWE ZAPYTANIA DO FANTASYPASY_STATS ----
    if "from fantasypasy_stats" in s:
        df = load_fantasy_table().copy()

        # WHERE Name = :name
        if "where name" in s and "name" in params and "Name" in df.columns:
            df = df[df["Name"] == params["name"]]

        # filtrowanie po zakresie dat – ds/de
        if ":ds" in s and "ds" in params and "DateStart" in df.columns:
            df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
            df = df[df["DateStart"] >= pd.to_datetime(params["ds"])]
        if ":de" in s and "de" in params and "DateEnd" in df.columns:
            df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
            df = df[df["DateEnd"] <= pd.to_datetime(params["de"])]

        # filtr po Team IN :teams
        if "team in :teams" in s and "teams" in params and "Team" in df.columns:
            df = df[df["Team"].isin(list(params["teams"]))]

        return df

    # fallback – jakby coś nowego się pojawiło
    return pd.DataFrame()

    # ========= DIAGNOSTYKA DLA DANYCH FANTASY =========
    st.subheader("DIAGNOSTYKA FANTASYPASY (tylko test – usuń po sprawdzeniu)")

    fant_debug = load_fantasy_table().copy()

    st.write("Kolumny w fantasypasy_stats.csv:")
    st.write(fant_debug.columns.tolist())

    # Konwersja dat
    fant_debug["DateStart"] = pd.to_datetime(fant_debug["DateStart"], errors="coerce")
    fant_debug["DateEnd"] = pd.to_datetime(fant_debug["DateEnd"], errors="coerce")

    st.write("Pierwsze 20 wierszy:")
    st.write(fant_debug.head(20))

    st.write("Unikalne pary dat w FANTASY:")
    st.write(fant_debug[["DateStart", "DateEnd"]].drop_duplicates())

    st.write("Typy danych:")
    st.write(fant_debug.dtypes)

# ===================== FUNKCJE LISTUJĄCE (korzystają już z CSV) =====================
def get_team_list():
    try:
        df = load_teams_table()
        return df["Team"].dropna().unique().tolist() if not df.empty else DEFAULT_TEAMS
    except Exception:
        return DEFAULT_TEAMS


def get_player_list():
    try:
        df = load_players_table()
        return df["Name"].dropna().unique().tolist() if not df.empty else []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def get_periods_df(table: str, teams: tuple | None = None) -> pd.DataFrame:
    """
    Zwraca okresy z measurement_periods TYLKO takie, dla których są dane
    w wybranej tabeli (FANTASYPASY albo MOTORYKA).
    Jeśli podasz teams != None, filtruje też po zespołach.
    """
    periods = get_periods_df().copy()
    if periods.empty:
        return periods

    table = table.upper()

    if table == "FANTASYPASY":
        src = load_fantasy_table().copy()
    else:   # "MOTORYKA"
        src = load_motoryka_table().copy()

    if src.empty:
        # brak danych w tej tabeli
        return periods.iloc[0:0].copy()

    # ewentualny filtr po zespołach
    if teams:
        if "Team" in src.columns:
            src = src[src["Team"].isin(teams)]
        if src.empty:
            return periods.iloc[0:0].copy()

    # ujednolicamy typy dat na date (bez czasu)
    periods["DateStart"] = pd.to_datetime(periods["DateStart"], errors="coerce").dt.date
    periods["DateEnd"] = pd.to_datetime(periods["DateEnd"], errors="coerce").dt.date

    src["DateStart"] = pd.to_datetime(src["DateStart"], errors="coerce").dt.date
    src["DateEnd"] = pd.to_datetime(src["DateEnd"], errors="coerce").dt.date

    pairs = (
        src[["DateStart", "DateEnd"]]
        .dropna()
        .drop_duplicates()
    )

    if pairs.empty:
        return periods.iloc[0:0].copy()

    # zostawiamy tylko okresy, które występują w danych źródłowych
    out = periods.merge(pairs, on=["DateStart", "DateEnd"], how="inner")

    return out.sort_values(["DateStart", "DateEnd"], ascending=[False, False])



@st.cache_data(show_spinner=False)
def load_fantasy(date_start=None, date_end=None, teams=None):
    sql = """
        SELECT Name, Team, Position, DateStart, DateEnd,
                NumberOfGames, Minutes,
                Goal, Assist, ChanceAssist, KeyPass,
                KeyLoss, DuelLossInBox, MissBlockShot,
                Finalization, KeyIndividualAction, KeyRecover, DuelWinInBox, BlockShot,
                PktOff, PktDef
        FROM fantasypasy_stats
        WHERE 1=1
    """
    params = {}
    if date_start and date_end:
        sql += " AND NOT (DateEnd < :ds OR DateStart > :de)"
        params["ds"] = date_start
        params["de"] = date_end
    elif date_start:
        sql += " AND DateEnd >= :ds"
        params["ds"] = date_start
    elif date_end:
        sql += " AND DateStart <= :de"
        params["de"] = date_end

    if teams:
        sql += " AND Team IN :teams"
        params["teams"] = tuple(teams)

    sql += " ORDER BY DateStart DESC"
    return fetch_df(sql, params)


@st.cache_data(show_spinner=False)
def load_motoryka_all(date_start=None, date_end=None, teams=None):
    sql = """
        SELECT Name, Team, Position, DateStart, DateEnd,
                Minutes, TD_m, HSR_m, Sprint_m, ACC, DECEL, PlayerIntensityIndex
        FROM motoryka_stats
        WHERE 1=1
    """
    params = {}
    if date_start and date_end:
        sql += " AND NOT (DateEnd < :ds OR DateStart > :de)"
        params["ds"] = date_start
        params["de"] = date_end
    elif date_start:
        sql += " AND DateEnd >= :ds"
        params["ds"] = date_start
    elif date_end:
        sql += " AND DateStart <= :de"
        params["de"] = date_end

    if teams:
        sql += " AND Team IN :teams"
        params["teams"] = tuple(teams)

    sql += " ORDER BY DateStart DESC"
    return fetch_df(sql, params)


def load_motoryka_for_compare(date_start=None, date_end=None):
    sql = """
        SELECT Name, Team, Position, DateStart, DateEnd,
               Minutes, HSR_m, Sprint_m, ACC, DECEL, PlayerIntensityIndex
        FROM motoryka_stats
        WHERE 1=1
    """
    params = {}
    if date_start:
        sql += " AND DateStart >= :ds"
        params["ds"] = date_start
    if date_end:
        sql += " AND DateEnd <= :de"
        params["de"] = date_end
    sql += " ORDER BY DateStart DESC"
    return fetch_df(sql, params)


@st.cache_data(show_spinner=False)
def get_c1_mean_pii(date_start=None, date_end=None):
    """
    Średni PlayerIntensityIndex dla zespołu C1 w zadanym zakresie dat.
    """
    df = load_motoryka_all(date_start, date_end, ["C1"])
    if df is None or df.empty or "PlayerIntensityIndex" not in df.columns:
        return np.nan
    return pd.to_numeric(df["PlayerIntensityIndex"], errors="coerce").mean()


# ==========================================
# Funkcja Excel
# ==========================================

def build_player_excel_report(player_name: str, moto: pd.DataFrame, fant: pd.DataFrame) -> BytesIO:
    """
    Eksport profilu zawodnika do Excela:
    - arkusz 'Fantasypasy' – surowe dane FANTASYPASY
    - arkusz 'Porównanie' – PII vs C1 + metryki per minutę
    """
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book

        # ===== 1) FANTASYPASY – surowe dane =====
        if fant is not None and not fant.empty:
            fant = fant.copy()
            fant["DateStart"] = pd.to_datetime(fant["DateStart"], errors="coerce").dt.date
            fant["DateEnd"] = pd.to_datetime(fant["DateEnd"], errors="coerce").dt.date

            fant_full = fant[[
                "DateStart", "DateEnd", "Goal", "Assist", "ChanceAssist", "KeyPass",
                "KeyLoss", "DuelLossInBox", "MissBlockShot", "Finalization",
                "KeyIndividualAction", "KeyRecover", "DuelWinInBox", "BlockShot",
                "PktOff", "PktDef"
            ]]

            fant_full.to_excel(writer, sheet_name="Fantasypasy", index=False, startrow=1)
            ws_fant = writer.sheets["Fantasypasy"]

            header_fmt = workbook.add_format({"bold": True})
            ws_fant.write(0, 0, f"FANTASYPASY – surowe dane ({player_name})", header_fmt)

        # ===== 2) PORÓWNANIE – PII vs C1 + metryki per minutę =====
        if moto is not None and not moto.empty:
            moto = moto.copy()
            moto["DateStart"] = pd.to_datetime(moto["DateStart"], errors="coerce").dt.date
            moto["DateEnd"] = pd.to_datetime(moto["DateEnd"], errors="coerce").dt.date
            moto["PlayerIntensityIndex"] = pd.to_numeric(
                moto.get("PlayerIntensityIndex"), errors="coerce"
            )
            moto["Minutes"] = pd.to_numeric(
                moto.get("Minutes"), errors="coerce"
            ).replace(0, np.nan)

            # średnia PII C1
            c1_mean_idx = get_c1_mean_pii(None, None)
            if pd.isna(c1_mean_idx):
                c1_mean_idx = pd.to_numeric(
                    moto.loc[moto["Team"] == "C1", "PlayerIntensityIndex"],
                    errors="coerce"
                ).mean()

            if pd.notna(c1_mean_idx) and c1_mean_idx != 0:
                moto["PII_diff_vs_C1"] = moto["PlayerIntensityIndex"] - c1_mean_idx
                moto["PII_ratio_vs_C1"] = moto["PlayerIntensityIndex"] / c1_mean_idx
            else:
                moto["PII_diff_vs_C1"] = np.nan
                moto["PII_ratio_vs_C1"] = np.nan

            per_min_cols = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]
            for col in per_min_cols:
                if col in moto.columns:
                    moto[col + "_per_min"] = (
                        pd.to_numeric(moto[col], errors="coerce") / moto["Minutes"]
                    ).replace([np.inf, -np.inf], np.nan)
                else:
                    moto[col + "_per_min"] = np.nan

            comp_cols = [
                "DateStart", "DateEnd", "Team", "Position", "Minutes",
                "PlayerIntensityIndex", "PII_diff_vs_C1", "PII_ratio_vs_C1"
            ]
            comp_cols = [c for c in comp_cols if c in moto.columns]

            comparison_data = moto[comp_cols].sort_values(["DateStart", "DateEnd"])
            comparison_data.to_excel(
                writer, sheet_name="Porównanie", index=False, startrow=1
            )
            ws_cmp = writer.sheets["Porównanie"]

            header_fmt = workbook.add_format({"bold": True})
            ws_cmp.write(0, 0, f"PORÓWNANIE – indeks PII {player_name} vs C1", header_fmt)

            per_min_out_cols = ["DateStart", "DateEnd", "Team", "Position", "Minutes"] + [
                col + "_per_min"
                for col in per_min_cols
                if col + "_per_min" in moto.columns
            ]
            per_min_out_cols = [c for c in per_min_out_cols if c in moto.columns]

            if len(per_min_out_cols) > 0:
                per_min_df = moto[per_min_out_cols].sort_values(["DateStart", "DateEnd"])
                startrow = len(comparison_data) + 4

                ws_cmp.write(
                    startrow - 1,
                    0,
                    "Metryki per minuta (TD_m, HSR_m, Sprint_m, ACC, DECEL)",
                    header_fmt
                )
                per_min_df.to_excel(
                    writer, sheet_name="Porównanie", index=False, startrow=startrow
                )

    output.seek(0)
    return output


# ===================== NAGŁÓWEK + NAWIGACJA =====================
with st.sidebar:
    st.markdown("---")
    st.header("Nawigacja")

    sekcja = st.radio(
        "Sekcja",
        [
            "Analiza",
            "Profil zawodnika",
            "Over/Under",
        ],
        key="nav_section",
    )

    if sekcja == "Analiza":
        page = st.radio(
            "Strona",
            ["Porównania", "Analiza (pozycje & zespoły)", "Wykresy zmian",
             "Indeks – porównania", "Fantasy – przegląd graficzny"],
            key="nav_page_ana",
        )

    elif sekcja == "Over/Under":
        page = "Over/Under"

    else:  # "Profil zawodnika"
        page = "Profil zawodnika"


st.title(" Cracovia – rejestry i analizy")


# Helpers dla pozycji
def extract_positions(series: pd.Series) -> list:
    all_pos = set()
    for val in series.dropna():
        parts = str(val).replace("\\", "/").split("/")
        for p in parts:
            p = p.strip().upper()
            if p:
                all_pos.add(p)
    return sorted(all_pos)


def _explode_positions(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for p in str(row["Position"]).replace("\\", "/").split("/"):
            p = p.strip().upper()
            if not p:
                continue
            r = row.copy()
            r["Position"] = p
            rows.append(r)
    return pd.DataFrame(rows) if rows else df.iloc[0:0].copy()


# ===================== POMOCNICY DO ANALIZY =====================
def _q25(x: pd.Series):
    return x.quantile(0.25)


def _q75(x: pd.Series):
    return x.quantile(0.75)


def flat_agg(df: pd.DataFrame, group_cols: list, num_cols: list) -> pd.DataFrame:
    df2 = df.copy()
    for c in num_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    agg = df2.groupby(group_cols, dropna=False)[num_cols].agg(
        ['count', 'sum', 'mean', 'median', 'std', 'min', 'max', _q25, _q75]
    )
    try:
        agg.rename(columns={'_q25': 'q25', '_q75': 'q75'}, level=1, inplace=True)
    except Exception:
        pass
    agg.columns = [f"{c}__{s}" for c, s in agg.columns.to_flat_index()]
    return agg.reset_index()


def add_per90_from_sums(agg_df, minutes_col_prefix, cols):
    out = agg_df.copy()
    mins = out.get(f"{minutes_col_prefix}__sum", pd.Series(dtype=float)).replace(0, np.nan)
    for c in cols:
        if f"{c}__sum" in out.columns:
            out[f"{c}__per90"] = (out[f"{c}__sum"] * 90.0) / mins
    return out


def download_button_for_df(df, label, filename):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=filename, mime="text/csv")


# ===================== PORÓWNANIA (Młodzież vs C1) =====================
def load_motoryka_for_compare_wrapper(date_start=None, date_end=None):
    return load_motoryka_for_compare(date_start, date_end)


def ref_c1_global(df: pd.DataFrame):
    df_c1 = df[df["Team"] == "C1"]
    if df_c1.empty:
        return None
    agg = df_c1.agg({
        "HSR_m": "mean",
        "Sprint_m": "mean",
        "ACC": "mean",
        "DECEL": "mean",
        "PlayerIntensityIndex": "mean"
    }).to_frame(name="C1_mean").T.reset_index(drop=True)
    agg["__key_global__"] = 1
    return agg


def ref_c1_by_position(df: pd.DataFrame):
    df_c1e = _explode_positions(df[df["Team"] == "C1"].copy())
    if df_c1e.empty:
        return None
    grp = (df_c1e.groupby("Position")
           .agg({"HSR_m": "mean", "Sprint_m": "mean", "ACC": "mean",
                 "DECEL": "mean", "PlayerIntensityIndex": "mean"})
           .rename(columns={"HSR_m": "HSR_m_C1", "Sprint_m": "Sprint_m_C1",
                            "ACC": "ACC_C1", "DECEL": "DECEL_C1",
                            "PlayerIntensityIndex": "PII_C1"})
           ).reset_index()
    return grp


def add_diffs(df, ref, by_position=True):
    df = df.copy()

    # 1) Normalizacja ewentualnych sufiksów po merge
    for base in ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]:
        for suf in ["_x", "_y"]:
            col = base + suf
            if col in df.columns and base not in df.columns:
                df.rename(columns={col: base}, inplace=True)

    # 2) Metryki, które realnie mamy
    metrics_all = ["HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]
    metrics = [m for m in metrics_all if m in df.columns]

    if not metrics:
        return df

    # 3) Jeśli ref to DataFrame
    if isinstance(ref, pd.DataFrame):
        if by_position:
            df = df.merge(ref, on="Position", how="left")
            for m in metrics:
                ref_col = "PII_C1" if m == "PlayerIntensityIndex" and "PII_C1" in df.columns else f"{m}_C1"
                if ref_col in df.columns:
                    df[m + "_diff"] = pd.to_numeric(df[m], errors="coerce") - pd.to_numeric(df[ref_col], errors="coerce")
                    denom = pd.to_numeric(df[ref_col], errors="coerce").replace(0, np.nan)
                    df[m + "_pct"] = pd.to_numeric(df[m], errors="coerce") / denom
        else:
            df["__key_global__"] = 1
            df = df.merge(ref, on="__key_global__", how="left", suffixes=("", "_C1"))
            for m in metrics:
                ref_col = f"{m}_C1" if f"{m}_C1" in df.columns else m
                if ref_col in df.columns:
                    df[m + "_diff"] = pd.to_numeric(df[m], errors="coerce") - pd.to_numeric(df[ref_col], errors="coerce")
                    denom = pd.to_numeric(df[ref_col], errors="coerce").replace(0, np.nan)
                    df[m + "_pct"] = pd.to_numeric(df[m], errors="coerce") / denom
            df.drop(columns=["__key_global__"], errors="ignore", inplace=True)

        return df

    # 4) W przeciwnym razie (ref to np. "C1") – licz średnie z df
    ref_team = ref if isinstance(ref, str) else "C1"
    if by_position and "Position" in df.columns:
        for m in metrics:
            ref_means = (
                df.loc[df["Team"] == ref_team]
                .groupby("Position")[m]
                .mean()
                .rename("ref_mean_" + m)
            )
            df = df.merge(ref_means, left_on="Position", right_index=True, how="left")
            df[m + "_diff"] = pd.to_numeric(df[m], errors="coerce") - pd.to_numeric(df["ref_mean_" + m], errors="coerce")
            denom = pd.to_numeric(df["ref_mean_" + m], errors="coerce").replace(0, np.nan)
            df[m + "_pct"] = pd.to_numeric(df[m], errors="coerce") / denom
            df.drop(columns=["ref_mean_" + m], inplace=True)
    else:
        for m in metrics:
            ref_mean = pd.to_numeric(df.loc[df["Team"] == ref_team, m], errors="coerce").mean()
            df[m + "_diff"] = pd.to_numeric(df[m], errors="coerce") - ref_mean
            df[m + "_pct"] = pd.to_numeric(df[m], errors="coerce") / (ref_mean if ref_mean not in (0, None, np.nan) else np.nan)

    return df



# ================= STRONA: PORÓWNANIA ====================
if page == "Porównania":
    st.subheader("Porównanie młodzieży do pierwszego zespołu C1")

    mode_cmp = st.radio(
        "Wybór zakresu",
        ["Okres/Test z rejestru", "Z istniejących par dat", "Ręcznie"],
        horizontal=True,
        key="cmp_mode",
    )

    ds_f, de_f = None, None
    periods = get_periods_df()

    if mode_cmp == "Okres/Test z rejestru":
        if periods.empty:
            st.info("Brak zapisanych okresów – wybierz inny tryb.")
        else:
            labels = [
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]"
                for _, r in periods.iterrows()
            ]
            pick = st.selectbox("Okres/Test", labels, index=0, key="cmp_pick_period")
            sel = periods.iloc[labels.index(pick)]
            ds_f, de_f = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_f.date()} → {de_f.date()}")

    elif mode_cmp == "Z istniejących par dat":
        pairs = fetch_df(
            "SELECT DISTINCT DateStart, DateEnd "
            "FROM motoryka_stats "
            "ORDER BY DateStart DESC, DateEnd DESC"
        )
        if pairs.empty:
            st.info("Brak danych – wybierz „Ręcznie”.")
        else:
            pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
            pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")
            opts = [f"{r.DateStart.date()} → {r.DateEnd.date()}" for _, r in pairs.iterrows()]
            pick = st.selectbox("Para dat", opts, index=0, key="cmp_pick_pair")
            sel = pairs.iloc[opts.index(pick)]
            ds_f, de_f = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_f.date()} → {de_f.date()}")

    else:
        c1, c2 = st.columns(2)
        ds_f = c1.date_input("Od (DateStart)", key="cmp_ds")
        de_f = c2.date_input("Do (DateEnd)", key="cmp_de")

    scope = st.radio(
        "Zakres referencji C1",
        ["Globalna średnia C1", "Średnie C1 per pozycja"],
        horizontal=True,
        key="cmp_scope",
    )

    df_m = load_motoryka_for_compare_wrapper(ds_f, de_f)
    ref = None

    if df_m.empty:
        st.info("Brak danych w wybranym zakresie.")
    else:
        show_only_non_c1 = st.toggle(
            "Pokaż tylko zawodników spoza C1",
            value=True,
            key="cmp_non_c1_only",
        )
        df_view = df_m[df_m["Team"] != "C1"].copy() if show_only_non_c1 else df_m.copy()

        all_positions = sorted(
            set(_explode_positions(df_m)["Position"].unique().tolist())
        )
        pos_pick_multi = st.multiselect(
            "Filtr pozycji (hybrydy wliczają się automatycznie):",
            options=all_positions,
            default=[],
            key="cmp_pos_multi",
        )
        if pos_pick_multi:
            df_view = _explode_positions(df_view)
            df_view = df_view[df_view["Position"].isin(pos_pick_multi)]

        players = sorted(df_view["Name"].dropna().unique().tolist())
        pick_players = st.multiselect(
            "Zawodnicy do tabeli",
            players,
            default=players[:10],
            key="cmp_players_multi",
        )
        if pick_players:
            df_view = df_view[df_view["Name"].isin(pick_players)]

        ref = (
            ref_c1_global(df_m)
            if scope == "Globalna średnia C1"
            else ref_c1_by_position(df_m)
        )

    if ref is None:
        st.info("Brak referencji C1 dla wybranego zakresu.")
    else:
        df_comp = add_diffs(
            df_view,
            ref,
            by_position=(scope != "Globalna średnia C1"),
        )

        metryki = ["HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]
        kolumny = (
            ["Name", "Team", "Position", "DateStart", "DateEnd"]
            + metryki
            + [m + "_diff" for m in metryki]
            + [m + "_pct" for m in metryki]
        )

        tabela = (
            df_comp[kolumny]
            .sort_values(
                ["Position", "Name", "DateStart"],
                ascending=[True, True, False],
            )
            .rename(
                columns={
                    "Name": "Zawodnik",
                    "Team": "Zespół",
                    "Position": "Pozycja",
                    "DateStart": "Data od",
                    "DateEnd": "Data do",
                }
            )
        )

        st.dataframe(
            tabela,
            use_container_width=True,
        )

        st.markdown(
            """
<div style='margin-top: 1rem; font-size: 0.9rem; line-height: 1.5;'>
<b>Legenda kolumn:</b><br>
• <b>HSR_m</b>, <b>Sprint_m</b>, <b>ACC</b>, <b>DECEL</b>, <b>PlayerIntensityIndex</b> – wartości surowe (średnie lub sumy z motoryki).<br>
• <b>_diff</b> – różnica między zawodnikiem a średnią zespołu C1 (dla wybranego zakresu i trybu porównania).<br>
&nbsp;&nbsp;&nbsp;&nbsp;Pozytywna wartość = zawodnik powyżej średniej C1, ujemna = poniżej.<br>
• <b>_pct</b> – stosunek wartości zawodnika do średniej C1 (np. 1.10 = 110%).<br>
• <b>Pozycja</b> – pozycja z której pochodzi wpis (uwzględnia hybrydy).<br>
• <b>Data od / Data do</b> – zakres okresu/testu, z którego pochodzi pomiar.<br>
</div>
""",
            unsafe_allow_html=True,
        )


# ===================== ANALIZA (pozycje & zespoły) =====================
elif page == "Analiza (pozycje & zespoły)":
    st.subheader("Analiza statystyk – per pozycja i per zespół")

    # =========================
    # Wybór zespołów i źródła danych
    # =========================
    teams_pick = st.multiselect(
        "Zespoły (puste = wszystkie)",
        get_team_list(),
        default=[],
        key="an_team_multi"
    )

    src = st.radio(
        "Tabela",
        ["FANTASYPASY", "MOTORYKA"],
        horizontal=True,
        key="an_src"
    )

    mode_a = st.radio(
        "Wybór zakresu",
        ["Okres/Test z rejestru", "Z istniejących par dat", "Ręcznie"],
        horizontal=True,
        key="an_mode"
    )

    # =========================
    # Ustalenie okresów
    # =========================
    ds_a, de_a = None, None

    periods_all = get_periods_df()
    periods = get_periods_with_data(
        "FANTASYPASY" if src == "FANTASYPASY" else "MOTORYKA",
        tuple(teams_pick) if teams_pick else None,
    )

    # =========================
    # TRYB 1 — Okres/Test z rejestru
    # =========================
    if mode_a == "Okres/Test z rejestru":
        if periods.empty:
            st.info("Brak okresów z danymi w wybranej tabeli (i zespołach).")
        else:
            labels = [
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]"
                for _, r in periods.iterrows()
            ]
            pick = st.selectbox("Okres/Test", labels, index=0, key="an_pick_period")
            sel = periods.iloc[labels.index(pick)]
            ds_a, de_a = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_a.date()} → {de_a.date()}")
    # =========================
    # TRYB 2 — Z istniejących par dat (tylko tam, gdzie SĄ dane)
    # =========================
    elif mode_a == "Z istniejących par dat":
        if src == "FANTASYPASY":
            df_src = load_fantasy(None, None, teams_pick or None)
        else:
            df_src = load_motoryka_all(None, None, teams_pick or None)

        if df_src is None or df_src.empty:
            st.info("Brak danych w wybranej tabeli (i zespołach) – wybierz „Ręcznie”.")
        else:
            df_src = df_src.copy()
            df_src["DateStart"] = pd.to_datetime(df_src["DateStart"], errors="coerce")
            df_src["DateEnd"] = pd.to_datetime(df_src["DateEnd"], errors="coerce")

            pairs = (
                df_src[["DateStart", "DateEnd"]]
                .dropna()
                .drop_duplicates()
                .sort_values(["DateStart", "DateEnd"], ascending=[False, False])
            )

            if pairs.empty:
                st.info("Brak poprawnych par dat w danych – wybierz inny tryb.")
            else:
                opts = [
                    f"{r.DateStart.date()} → {r.DateEnd.date()}"
                    for _, r in pairs.iterrows()
                ]
                pick = st.selectbox("Para dat", opts, index=0, key="an_pick_pair")
                sel = pairs.iloc[opts.index(pick)]
                ds_a, de_a = sel["DateStart"], sel["DateEnd"]
                st.caption(f"Zakres: {ds_a.date()} → {de_a.date()}")

    # =========================
    # TRYB 3 — Ręcznie
    # =========================
    else:
        c1, c2 = st.columns(2)
        ds_a = c1.date_input("Od (DateStart)", value=None, key="an_ds")
        de_a = c2.date_input("Do (DateEnd)", value=None, key="an_de")

    # =====================================================
    #                 ANALIZA – FANTASYPASY
    # =====================================================
    if src == "FANTASYPASY":
        df = load_fantasy(ds_a, de_a, teams_pick or None)
        if df is None or df.empty:
            st.info("Brak danych FANTASYPASY w wybranym zakresie.")
        else:
            df_pos = _explode_positions(df)
            all_pos = sorted(df_pos["Position"].dropna().unique().tolist())

            pos_pick = st.multiselect(
                "Filtr pozycji (hybrydy uwzględnione)",
                all_pos,
                default=[],
                key="an_pos_fant"
            )
            if pos_pick:
                df_pos = df_pos[df_pos["Position"].isin(pos_pick)]

            num_cols = [
                "NumberOfGames", "Minutes",
                "Goal", "Assist", "ChanceAssist", "KeyPass",
                "KeyLoss", "DuelLossInBox", "MissBlockShot",
                "Finalization", "KeyIndividualAction", "KeyRecover",
                "DuelWinInBox", "BlockShot", "PktOff", "PktDef"
            ]

            # ---------------------------------
            #   AGREGACJA: POZYCJA
            # ---------------------------------
            st.markdown("### Pozycja")
            agg_pos = add_per90_from_sums(
                flat_agg(df_pos, ["Position"], num_cols),
                "Minutes",
                [
                    "Goal", "Assist", "ChanceAssist", "KeyPass",
                    "KeyLoss", "DuelLossInBox", "MissBlockShot",
                    "Finalization", "KeyIndividualAction", "KeyRecover",
                    "DuelWinInBox", "BlockShot", "PktOff", "PktDef"
                ]
            )
            agg_pos_disp = agg_pos.rename(columns={
                "Position": "Pozycja"
            })
            st.dataframe(agg_pos_disp, use_container_width=True)
            download_button_for_df(
                agg_pos,
                "📥 Pobierz CSV (pozycja)",
                "fantasypasy_per_position.csv"
            )

            # ---------------------------------
            #   AGREGACJA: ZESPÓŁ
            # ---------------------------------
            st.markdown("### Zespół")
            agg_team = add_per90_from_sums(
                flat_agg(df, ["Team"], num_cols),
                "Minutes",
                [
                    "Goal", "Assist", "ChanceAssist", "KeyPass",
                    "KeyLoss", "DuelLossInBox", "MissBlockShot",
                    "Finalization", "KeyIndividualAction", "KeyRecover",
                    "DuelWinInBox", "BlockShot", "PktOff", "PktDef"
                ]
            )
            agg_team_disp = agg_team.rename(columns={
                "Team": "Zespół"
            })
            st.dataframe(agg_team_disp, use_container_width=True)
            download_button_for_df(
                agg_team,
                "📥 Pobierz CSV (zespół)",
                "fantasypasy_per_team.csv"
            )

            # ---------------------------------
            #   AGREGACJA: ZESPÓŁ × POZYCJA
            # ---------------------------------
            st.markdown("### Zespół × Pozycja")
            agg_pos_team = add_per90_from_sums(
                flat_agg(df_pos, ["Team", "Position"], num_cols),
                "Minutes",
                [
                    "Goal", "Assist", "ChanceAssist", "KeyPass",
                    "KeyLoss", "DuelLossInBox", "MissBlockShot",
                    "Finalization", "KeyIndividualAction", "KeyRecover",
                    "DuelWinInBox", "BlockShot", "PktOff", "PktDef"
                ]
            )
            agg_pos_team_disp = agg_pos_team.rename(columns={
                "Team": "Zespół",
                "Position": "Pozycja"
            })
            st.dataframe(agg_pos_team_disp, use_container_width=True)
            download_button_for_df(
                agg_pos_team,
                "📥 Pobierz CSV (zespół×pozycja)",
                "fantasypasy_position_team.csv"
            )

            st.markdown(
                """
<div style='margin-top: 1rem; font-size: 0.9rem; line-height: 1.5;'>
<b>Legenda kolumn (Analiza – FANTASYPASY):</b><br>
• <b>__count</b> – liczba wpisów w grupie.<br>
• <b>__sum</b> – suma wartości w grupie.<br>
• <b>__mean</b> – średnia; <b>__median</b> – mediana; <b>__std</b> – odchylenie standardowe.<br>
• <b>__min</b> / <b>__max</b> – wartości minimalne / maksymalne.<br>
• <b>__q25</b> / <b>__q75</b> – kwartyle 25% / 75%.<br>
• <b>__per90</b> – przeliczenie na 90 minut (gdy dotyczy).<br>
</div>
""",
                unsafe_allow_html=True,
            )

    # =====================================================
    #                 ANALIZA – MOTORYKA
    # =====================================================
    else:
        df = load_motoryka_all(ds_a, de_a, teams_pick or None)
        if df is None or df.empty:
            st.info("Brak danych MOTORYKA w wybranym zakresie.")
        else:
            df_pos = _explode_positions(df)
            all_pos = sorted(df_pos["Position"].dropna().unique().tolist())

            pos_pick = st.multiselect(
                "Filtr pozycji (hybrydy uwzględnione)",
                all_pos,
                default=[],
                key="an_pos_moto"
            )
            if pos_pick:
                df_pos = df_pos[df_pos["Position"].isin(pos_pick)]

            num_cols = [
                "Minutes", "TD_m", "HSR_m", "Sprint_m",
                "ACC", "DECEL", "PlayerIntensityIndex"
            ]

            # ---------------------------------
            #   AGREGACJA: POZYCJA
            # ---------------------------------
            st.markdown("### Pozycja")
            agg_pos = flat_agg(df_pos, ["Position"], num_cols)
            agg_pos_disp = agg_pos.rename(columns={
                "Position": "Pozycja"
            })
            st.dataframe(agg_pos_disp, use_container_width=True)
            download_button_for_df(
                agg_pos,
                "📥 Pobierz CSV (pozycja)",
                "motoryka_per_position.csv"
            )

            # ---------------------------------
            #   AGREGACJA: ZESPÓŁ
            # ---------------------------------
            st.markdown("### Zespół")
            agg_team = flat_agg(df, ["Team"], num_cols)
            agg_team_disp = agg_team.rename(columns={
                "Team": "Zespół"
            })
            st.dataframe(agg_team_disp, use_container_width=True)
            download_button_for_df(
                agg_team,
                "📥 Pobierz CSV (zespół)",
                "motoryka_per_team.csv"
            )

            # ---------------------------------
            #   AGREGACJA: ZESPÓŁ × POZYCJA
            # ---------------------------------
            st.markdown("### Zespół × Pozycja")
            agg_pos_team = flat_agg(df_pos, ["Team", "Position"], num_cols)
            agg_pos_team_disp = agg_pos_team.rename(columns={
                "Team": "Zespół",
                "Position": "Pozycja"
            })
            st.dataframe(agg_pos_team_disp, use_container_width=True)
            download_button_for_df(
                agg_pos_team,
                "📥 Pobierz CSV (zespół×pozycja)",
                "motoryka_position_team.csv"
            )

            st.markdown(
                """
<div style='margin-top: 1rem; font-size: 0.9rem; line-height: 1.5;'>
<b>Legenda kolumn (Analiza – MOTORYKA):</b><br>
• <b>__count</b> – liczba wpisów w grupie.<br>
• <b>__sum</b> – suma wartości w grupie.<br>
• <b>__mean</b> – średnia; <b>__median</b> – mediana; <b>__std</b> – odchylenie standardowe.<br>
• <b>__min</b> / <b>__max</b> – wartości minimalne / maksymalne.<br>
• <b>__q25</b> / <b>__q75</b> – kwartyle 25% / 75%.<br>
• (MOTORYKA nie ma przeliczeń __per90 w tej tabeli).<br>
</div>
""",
                unsafe_allow_html=True,
            )


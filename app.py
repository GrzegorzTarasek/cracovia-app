import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

# ===================== LEKKIE „EMULOWANIE” fetch_df() =====================
def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    """
    Zastępuje wcześniejsze zapytania SQL na bazie – na podstawie tekstu SELECT
    zwraca odpowiedni DataFrame z CSV.
    """
    if params is None:
        params = {}

    s = sql.lower().strip()

    # ---- LISTA ZESPOŁÓW ----
    # SELECT Team FROM teams ORDER BY Team;
    if "from teams" in s and "select" in s and "team" in s:
        df = load_teams_table().copy()
        if "Team" not in df.columns:
            return pd.DataFrame(columns=["Team"])
        return df[["Team"]].dropna().drop_duplicates().sort_values("Team")

    # ---- LISTA ZAWODNIKÓW ----
    # SELECT Name FROM players ORDER BY Name;
    if "from players" in s and "select" in s and "name" in s and "team" not in s:
        df = load_players_table().copy()
        if "Name" not in df.columns:
            return pd.DataFrame(columns=["Name"])
        return df[["Name"]].dropna().drop_duplicates().sort_values("Name")

    # ---- LISTA ZAWODNIKÓW Z TEAM & POSITION ----
    # SELECT Name, Team, Position FROM players ORDER BY Name;
    if "from players" in s and "select" in s and "team" in s:
        df = load_players_table().copy()
        cols = [c for c in ["Name", "Team", "Position"] if c in df.columns]
        if not cols:
            return pd.DataFrame()
        return df[cols].dropna(subset=["Name"]).sort_values("Name")

    # ---- OKRESY POMIAROWE ----
    # SELECT PeriodID, Label, DateStart, DateEnd FROM measurement_periods ...
    if "from measurement_periods" in s:
        df = load_periods_table().copy()
        # zabezpieczenie – ograniczamy kolumny, które faktycznie istnieją
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
        df = df[["DateStart", "DateEnd"]].dropna().drop_duplicates()
        return df.sort_values(["DateStart", "DateEnd"], ascending=[False, False])

    # ---- PODSTAWOWE ZAPYTANIA DO MOTORYKI ----
    # SELECT ... FROM motoryka_stats WHERE ...
    if "from motoryka_stats" in s:
        df = load_motoryka_table().copy()

        # WHERE Name = :name
        if "where name" in s and "name" in params and "Name" in df.columns:
            df = df[df["Name"] == params["name"]]

        # filtrowanie po zakresie dat – dla uproszczenia bierzemy paramy ds/de jeśli są
        if ":ds" in s and "ds" in params and "DateStart" in df.columns:
            df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
            df = df[df["DateStart"] >= pd.to_datetime(params["ds"])]
        if ":de" in s and "de" in params and "DateEnd" in df.columns:
            df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
            df = df[df["DateEnd"] <= pd.to_datetime(params["de"])]

        return df

    # ---- PODSTAWOWE ZAPYTANIA DO FANTASYPASY ----
    if "from fantasypasy_stats" in s:
        df = load_fantasy_table().copy()

        # filtrowanie po zakresie dat – ds/de
        if ":ds" in s and "ds" in params and "DateStart" in df.columns:
            df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
            df = df[df["DateStart"] >= pd.to_datetime(params["ds"])]
        if ":de" in s and "de" in params and "DateEnd" in df.columns:
            df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
            df = df[df["DateEnd"] <= pd.to_datetime(params["de"])]

        return df

    # fallback – jakby coś nowego się pojawiło
    return pd.DataFrame()

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
def get_periods_df():
    try:
        df = load_periods_table().copy()
        if df.empty:
            return pd.DataFrame(columns=["PeriodID", "Label", "DateStart", "DateEnd"])
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        cols = [c for c in ["PeriodID", "Label", "DateStart", "DateEnd"] if c in df.columns]
        return df[cols].dropna(subset=["DateStart", "DateEnd"]).sort_values(
            ["DateStart", "DateEnd"], ascending=[False, False]
        )
    except Exception:
        return pd.DataFrame(columns=["PeriodID", "Label", "DateStart", "DateEnd"])




# ===================== EMULACJA fetch_df(...) NA CSV =====================

def fetch_df(sql, params=None):
    """
    Zastępstwo dla starego pd.read_sql(...):
    rozpoznajemy konkretne zapytania, które masz w kodzie,
    i robimy na CSV to samo, co wcześniej robiła baza.
    """
    if sql is None:
        return pd.DataFrame()

    params = params or {}
    s = " ".join(sql.split()).lower()

    # --- measurement_periods: SELECT PeriodID, Label, DateStart, DateEnd ...
    if "from measurement_periods" in s:
        df = load_periods_table().copy()
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        return df.sort_values(
            ["DateStart", "Label"],
            ascending=[False, True],
        )[["PeriodID", "Label", "DateStart", "DateEnd"]]

    # --- DISTINCT DateStart, DateEnd FROM motoryka_stats ...
    if "distinct datestart, dateend" in s and "from motoryka_stats" in s:
        df = load_motoryka_table().copy()
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        df = (
            df[["DateStart", "DateEnd"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["DateStart", "DateEnd"], ascending=[False, False])
        )
        return df

    # --- DISTINCT DateStart, DateEnd FROM fantasypasy_stats ...
    if "distinct datestart, dateend" in s and "from fantasypasy_stats" in s:
        df = load_fantasy_table().copy()
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        df = (
            df[["DateStart", "DateEnd"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["DateStart", "DateEnd"], ascending=[False, False])
        )
        return df

    # --- SELECT Name, Team, Position FROM players ORDER BY Name;
    if "from players" in s:
        df = load_players_table().copy()
        cols = [c for c in ["Name", "Team", "Position"] if c in df.columns]
        return df[cols].sort_values("Name")

    # --- SELECT * FROM motoryka_stats WHERE Name = :name
    if "from motoryka_stats" in s and "where name" in s:
        df = load_motoryka_table().copy()
        if "Name" in df.columns and "name" in params:
            df = df[df["Name"] == params["name"]]
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        return df.sort_values("DateStart")

    # --- SELECT * FROM fantasypasy_stats WHERE Name = :name
    if "from fantasypasy_stats" in s and "where name" in s:
        df = load_fantasy_table().copy()
        if "Name" in df.columns and "name" in params:
            df = df[df["Name"] == params["name"]]
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        return df.sort_values("DateStart")

    # fallback: nic nie rozpoznaliśmy → pusta ramka zamiast wywalenia się aplikacji
    return pd.DataFrame()



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
def _q25(x: pd.Series): return x.quantile(0.25)
def _q75(x: pd.Series): return x.quantile(0.75)


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
    import numpy as np
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

    ds_a, de_a = None, None
    periods = get_periods_df()

    if mode_a == "Okres/Test z rejestru":
        if periods.empty:
            st.info("Brak zapisanych okresów – wybierz inną opcję.")
        else:
            labels = [
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]"
                for _, r in periods.iterrows()
            ]
            pick = st.selectbox("Okres/Test", labels, index=0, key="an_pick_period")
            sel = periods.iloc[labels.index(pick)]
            ds_a, de_a = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_a.date()} → {de_a.date()}")

    elif mode_a == "Z istniejących par dat":
        table = "motoryka_stats" if src == "MOTORYKA" else "fantasypasy_stats"
        pairs = fetch_df(
            f"SELECT DISTINCT DateStart, DateEnd "
            f"FROM {table} "
            f"ORDER BY DateStart DESC, DateEnd DESC"
        )
        if pairs.empty:
            st.info(f"Brak danych w {table} – wybierz „Ręcznie”.")
        else:
            pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
            pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")
            opts = [f"{r.DateStart.date()} → {r.DateEnd.date()}" for _, r in pairs.iterrows()]
            pick = st.selectbox("Para dat", opts, index=0, key="an_pick_pair")
            sel = pairs.iloc[opts.index(pick)]
            ds_a, de_a = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_a.date()} → {de_a.date()}")

    else:
        c1, c2 = st.columns(2)
        ds_a = c1.date_input("Od (DateStart)", value=None, key="an_ds")
        de_a = c2.date_input("Do (DateEnd)", value=None, key="an_de")

    # ===================== FANTASYPASY =====================
    if src == "FANTASYPASY":
        df = load_fantasy(ds_a, de_a, teams_pick or None)
        if df.empty:
            st.info("Brak danych w wybranym zakresie.")
        else:
            df_pos = _explode_positions(df)
            all_pos = sorted(df_pos["Position"].unique().tolist())
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

            # ------- Pozycja -------
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
                " Pobierz CSV (pozycja)",
                "fantasypasy_per_position.csv"
            )

            # ------- Zespół -------
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
                " Pobierz CSV (zespół)",
                "fantasypasy_per_team.csv"
            )

            # ------- Zespół × Pozycja -------
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
                " Pobierz CSV (zespół×pozycja)",
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

    # ===================== MOTORYKA =====================
    else:
        df = load_motoryka_all(ds_a, de_a, teams_pick or None)
        if df.empty:
            st.info("Brak danych w wybranym zakresie.")
        else:
            df_pos = _explode_positions(df)
            all_pos = sorted(df_pos["Position"].unique().tolist())
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

            # ------- Pozycja -------
            st.markdown("### Pozycja")
            agg_pos = flat_agg(df_pos, ["Position"], num_cols)
            agg_pos_disp = agg_pos.rename(columns={
                "Position": "Pozycja"
            })
            st.dataframe(agg_pos_disp, use_container_width=True)
            download_button_for_df(
                agg_pos,
                " Pobierz CSV (pozycja)",
                "motoryka_per_position.csv"
            )

            # ------- Zespół -------
            st.markdown("### Zespół")
            agg_team = flat_agg(df, ["Team"], num_cols)
            agg_team_disp = agg_team.rename(columns={
                "Team": "Zespół"
            })
            st.dataframe(agg_team_disp, use_container_width=True)
            download_button_for_df(
                agg_team,
                " Pobierz CSV (zespół)",
                "motoryka_per_team.csv"
            )

            # ------- Zespół × Pozycja -------
            st.markdown("### Zespół × Pozycja")
            agg_pos_team = flat_agg(df_pos, ["Team", "Position"], num_cols)
            agg_pos_team_disp = agg_pos_team.rename(columns={
                "Team": "Zespół",
                "Position": "Pozycja"
            })
            st.dataframe(agg_pos_team_disp, use_container_width=True)
            download_button_for_df(
                agg_pos_team,
                " Pobierz CSV (zespół×pozycja)",
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

# ===================== WYKRESY ZMIAN =====================
elif page == "Wykresy zmian":
    st.subheader("Wykresy zmian po dacie")
    st.caption("MOTORYKA – metryki zliczeniowe w przeliczeniu na minutę; bez indeksów.")

    dfw = load_motoryka_all(None, None, None)
    if dfw.empty:
        st.info("Brak danych motorycznych.")
        st.stop()

    per_minute_base = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

    mode_players = st.radio(
        "Zakres zawodników",
        ["Cała drużyna", "Pozycja", "Wybrani gracze"],
        horizontal=True,
        key="plot_players_mode"
    )

    teams_list = sorted(dfw["Team"].dropna().unique().tolist())
    positions_all = sorted(extract_positions(dfw["Position"]))

    selected_names = []
    if mode_players == "Cała drużyna":
        t = st.selectbox(
            "Drużyna",
            teams_list if teams_list else [""],
            key="plot_pick_team"
        )
        dff = dfw[dfw["Team"] == t].copy() if t else dfw.copy()
        selected_names = sorted(dff["Name"].dropna().unique().tolist())

    elif mode_players == "Pozycja":
        pos_pick = st.multiselect(
            "Pozycja",
            positions_all,
            default=[],
            key="plot_pick_positions"
        )
        if pos_pick:
            rows = []
            for _, r in dfw.iterrows():
                parts = str(r["Position"] or "").replace("\\", "/").split("/")
                if any(p.strip().upper() in pos_pick for p in parts):
                    rows.append(r)
            dff = pd.DataFrame(rows) if rows else dfw.iloc[0:0].copy()
        else:
            dff = dfw.iloc[0:0].copy()
        selected_names = sorted(dff["Name"].dropna().unique().tolist())

    else:  # Wybrani gracze
        all_names = sorted(dfw["Name"].dropna().unique().tolist())
        selected_names = st.multiselect(
            "Zawodnicy",
            all_names,
            default=all_names[:3],
            key="plot_pick_names"
        )
        dff = dfw[dfw["Name"].isin(selected_names)].copy() if selected_names else dfw.iloc[0:0].copy()

    if dff.empty or not selected_names:
        st.info("Brak danych po zastosowaniu wyboru zawodników.")
        st.stop()

    dff = dff.copy()
    dff["Minutes"] = pd.to_numeric(dff["Minutes"], errors="coerce").replace(0, np.nan)
    for m in per_minute_base:
        dff[m + "_per_min"] = pd.to_numeric(dff[m], errors="coerce") / dff["Minutes"]

    mid = pd.to_datetime(dff["DateStart"]) + (
        pd.to_datetime(dff["DateEnd"]) - pd.to_datetime(dff["DateStart"])
    ) / 2
    dff["DateMid"] = mid.dt.date

    periods = get_periods_df()

    dff["RangeFallback"] = dff["DateStart"].astype(str) + " → " + dff["DateEnd"].astype(str)

    if not periods.empty:
        dff = dff.merge(
            periods[["Label", "DateStart", "DateEnd"]],
            on=["DateStart", "DateEnd"],
            how="left"
        )
        dff["RangeLabel"] = dff["Label"].where(dff["Label"].notna(), dff["RangeFallback"])
    else:
        dff["RangeLabel"] = dff["RangeFallback"]

    metric = st.selectbox(
        "Metryka (na minutę)",
        [m + "_per_min" for m in per_minute_base],
        key="plot_metric_per_min"
    )

    base = (
        dff[["Name", "Team", "RangeLabel", "DateMid", metric]]
        .rename(columns={metric: "Value"})
        .dropna(subset=["Value"])
    )

    if base.empty:
        st.info("Brak wartości do wykresu dla wybranej metryki.")
        st.stop()

    table_view = base[["Name", "Team", "RangeLabel", "Value"]].sort_values(
        ["Team", "Name", "RangeLabel"]
    )

    plot_df = base.copy()

    if mode_players in ["Pozycja", "Wybrani gracze"]:
        plot_df["LegendLabel"] = plot_df["Team"].astype(str) + " | " + plot_df["Name"].astype(str)
        color_field = "LegendLabel"
        legend_title = "Zespół | Zawodnik"
    else:
        color_field = "Name"
        legend_title = "Zawodnik"

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("DateMid:T", title="Data (środek zakresu)"),
            y=alt.Y("Value:Q", title=metric),
            color=alt.Color(f"{color_field}:N", title=legend_title, legend=alt.Legend(columns=2)),
            tooltip=[
                alt.Tooltip("DateMid:T", title="Data (środek)"),
                alt.Tooltip("RangeLabel:N", title="Zakres"),
                alt.Tooltip("Team:N", title="Zespół"),
                alt.Tooltip("Name:N", title="Zawodnik"),
                alt.Tooltip("Value:Q", title=metric, format=".3f"),
            ],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("### Dane (z nazwą przedziału)")
    st.dataframe(table_view, use_container_width=True)


# ===================== FANTASY – PRZEGLĄD GRAFICZNY =====================
elif page == "Fantasy – przegląd graficzny":
    st.subheader("FANTASYPASY – przegląd graficzny (po drużynie / meczu / obu)")

    mode_fx = st.radio(
        "Wybór zakresu",
        ["Z istniejących par dat (FANTASYPASY)", "Okres/Test z rejestru", "Ręcznie"],
        horizontal=True,
        key="fx_mode"
    )

    selected_pairs = None
    ds_fv, de_fv = None, None
    periods = get_periods_df()

    if mode_fx == "Z istniejących par dat (FANTASYPASY)":
        pairs = fetch_df("""
            SELECT DISTINCT DateStart, DateEnd
            FROM fantasypasy_stats
            ORDER BY DateStart DESC, DateEnd DESC
        """)
        if pairs.empty:
            st.info("Brak zapisanych par dat w FANTASYPASY – wybierz inną opcję.")
        else:
            pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
            pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")
            opts = [f"{r.DateStart.date()} → {r.DateEnd.date()}" for _, r in pairs.iterrows()]
            label_to_tuple = {f"{r.DateStart.date()} → {r.DateEnd.date()}": (r.DateStart, r.DateEnd) for _, r in pairs.iterrows()}

            if "fx_pick_pairs" not in st.session_state:
                st.session_state["fx_pick_pairs"] = []

            c1, c2 = st.columns([1, 3])
            if c1.button("Wybierz wszystkie", key="fx_btn_all_pairs"):
                st.session_state["fx_pick_pairs"] = opts

            picked = c2.multiselect("Pary dat", opts, key="fx_pick_pairs")
            selected_pairs = {label_to_tuple[x] for x in picked} if picked else None

    elif mode_fx == "Okres/Test z rejestru":
        if periods.empty:
            st.info("Brak okresów w rejestrze – wybierz inną opcję.")
        else:
            labels = [f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]" for _, r in periods.iterrows()]
            label_to_tuple = {f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]": (r.DateStart, r.DateEnd) for _, r in periods.iterrows()}

            if "fx_pick_periods" not in st.session_state:
                st.session_state["fx_pick_periods"] = []

            c1, c2 = st.columns([1, 3])
            if c1.button("Wybierz wszystkie", key="fx_btn_all_periods"):
                st.session_state["fx_pick_periods"] = labels

            picked = c2.multiselect("Okresy/Testy", labels, key="fx_pick_periods")
            selected_pairs = {label_to_tuple[x] for x in picked} if picked else None

    else:
        c1, c2 = st.columns(2)
        ds_fv = c1.date_input("Od (DateStart)", value=None, key="fx_ds")
        de_fv = c2.date_input("Do (DateEnd)", value=None, key="fx_de")

    if selected_pairs is not None:
        df = load_fantasy(None, None, teams=None)
        if df.empty:
            st.info("Brak danych FANTASYPASY.")
            st.stop()
        pick_df = pd.DataFrame(list(selected_pairs), columns=["DateStart", "DateEnd"])
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        df = df.merge(pick_df, on=["DateStart", "DateEnd"], how="inner")
        st.caption(f"Wybrano {len(selected_pairs)} zakres(y) dat.")
    else:
        df = load_fantasy(ds_fv, de_fv, teams=None)

    if df.empty:
        st.info("Brak danych FANTASYPASY w wybranym zakresie.")
    else:
        mid = pd.to_datetime(df["DateStart"]) + (pd.to_datetime(df["DateEnd"]) - pd.to_datetime(df["DateStart"])) / 2
        df = df.copy()
        df["DateMid"] = mid.dt.date
        df["DateMidStr"] = df["DateMid"].astype(str)
        df["FacetBoth"] = df["Team"].astype(str) + " | " + df["DateMidStr"]

        metric = st.selectbox(
            "Metryka do analizy",
            ["PktOff", "PktDef", "Goal", "Assist", "ChanceAssist", "KeyPass",
             "KeyLoss", "DuelLossInBox", "MissBlockShot", "Finalization",
             "KeyIndividualAction", "KeyRecover", "DuelWinInBox", "BlockShot"],
            key="fx_metric"
        )

        seg = st.radio("Segregacja", ["po drużynie", "po meczu", "po obu (siatka)"],
                       horizontal=True, key="fx_seg2")

        teams = sorted(df["Team"].dropna().unique().tolist())
        team_pick = st.multiselect("Filtr drużyn (opcjonalnie)", teams, default=[], key="fx_teams2")
        if team_pick:
            df = df[df["Team"].isin(team_pick)]

        if df.empty:
            st.info("Brak danych po filtrach.")
        else:
            def facet_wrap(chart, field_shorthand: str, columns: int = 3):
                return chart.facet(facet=field_shorthand, columns=columns)

            st.write("PktOff vs PktDef (scatter)")
            scatter = alt.Chart(df).mark_circle(size=80, opacity=0.75).encode(
                x=alt.X("PktOff:Q", title="PktOff"),
                y=alt.Y("PktDef:Q", title="PktDef"),
                color=alt.Color("Team:N", title="Drużyna"),
                tooltip=["Name", "Team", "DateMid", "PktOff", "PktDef"]
            ).properties(height=330)

            if seg == "po drużynie":
                chart_scatter = facet_wrap(scatter, "Team:N", columns=3)
            elif seg == "po meczu":
                chart_scatter = facet_wrap(scatter, "DateMidStr:N", columns=3)
            else:
                chart_scatter = facet_wrap(scatter, "FacetBoth:N", columns=3)

            st.altair_chart(
                chart_scatter.resolve_scale(x="independent", y="independent"),
                use_container_width=True
            )

            st.write(f"Rozkład metryki: {metric}")
            box = alt.Chart(df).mark_boxplot(size=40).encode(
                y=alt.Y(f"{metric}:Q", title=metric),
                color=alt.Color("Team:N", title="Drużyna"),
                tooltip=["Team:N", "Name:N", "DateMid:T", f"{metric}:Q"]
            ).properties(height=300)

            if seg == "po drużynie":
                chart_box = facet_wrap(box, "Team:N", columns=3)
            elif seg == "po meczu":
                chart_box = facet_wrap(box, "DateMidStr:N", columns=3)
            else:
                chart_box = facet_wrap(box, "FacetBoth:N", columns=3)

            st.altair_chart(
                chart_box.resolve_scale(y="independent"),
                use_container_width=True
            )

            st.write(f"Ranking zawodników wg {metric}")
            rank_df = (
                df.groupby(["Name", "Team"], as_index=False)[metric]
                .mean()
                .sort_values(metric, ascending=False)
            )
            st.dataframe(rank_df.head(20), use_container_width=True)


# ===================== INDEKS – PORÓWNANIA =====================
elif page == "Indeks – porównania":
    st.subheader("Indeks – porównania i rankingi")

    mode_idx = st.radio(
        "Wybór zakresu",
        ["Okres/Test z rejestru", "Z istniejących par dat (MOTORYKA)"],
        horizontal=True,
        key="idx_range_mode"
    )

    ds_i, de_i = None, None
    periods = get_periods_df()

    if mode_idx == "Okres/Test z rejestru":
        if periods.empty:
            st.info("Brak zapisanych okresów – dodaj coś albo wybierz drugi tryb.")
            st.stop()
        else:
            labels = [
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]"
                for _, r in periods.iterrows()
            ]
            pick = st.selectbox("Okres/Test", labels, index=0, key="idx_pick_period")
            sel = periods.iloc[labels.index(pick)]
            ds_i, de_i = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_i.date()} → {de_i.date()}")

    else:
        pairs = fetch_df(
            """
            SELECT DISTINCT DateStart, DateEnd
            FROM motoryka_stats
            ORDER BY DateStart DESC, DateEnd DESC
            """
        )
        if pairs.empty:
            st.info("Brak par dat w motoryka_stats – najpierw zapisz dane motoryczne.")
            st.stop()
        else:
            pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
            pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")
            opts = [f"{r.DateStart.date()} → {r.DateEnd.date()}" for _, r in pairs.iterrows()]
            pick = st.selectbox("Para dat (MOTORYKA)", opts, index=0, key="idx_pick_pair")
            sel = pairs.iloc[opts.index(pick)]
            ds_i, de_i = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_i.date()} → {de_i.date()}")

    df = load_motoryka_all(ds_i, de_i, None)
    if df.empty:
        st.info("Brak danych w wybranym zakresie.")
        st.stop()

    df = df.copy()
    df["PlayerIntensityIndex"] = pd.to_numeric(df["PlayerIntensityIndex"], errors="coerce")
    df = df.dropna(subset=["PlayerIntensityIndex"])

    mid_i = pd.to_datetime(df["DateStart"]) + (
        pd.to_datetime(df["DateEnd"]) - pd.to_datetime(df["DateStart"])
    ) / 2
    df["DateMid"] = mid_i.dt.date

    if (
        "PlayerIntensityIndexComparingToTeamAverage" in df.columns
        and df["PlayerIntensityIndexComparingToTeamAverage"].notna().any()
    ):
        df["PII_vs_team_avg"] = pd.to_numeric(
            df["PlayerIntensityIndexComparingToTeamAverage"], errors="coerce"
        )
    else:
        c1_mean_idx = get_c1_mean_pii(ds_i, de_i)
        df["PII_vs_team_avg"] = df["PlayerIntensityIndex"] - c1_mean_idx

    mode = st.radio(
        "Tryb",
        ["Ranking ogólny", "Ranking per data", "Ranking per zespół", "Porównanie graczy"],
        horizontal=True,
        key="idx_mode_fix"
    )

    if mode == "Ranking ogólny":
        top_n = st.slider("Top N", 5, 50, 20, key="idx_top_all_fix")
        view = (
            df[["Name", "Team", "DateMid", "PlayerIntensityIndex", "PII_vs_team_avg"]]
            .sort_values("PlayerIntensityIndex", ascending=False)
            .head(top_n)
        )
        view_disp = view.rename(columns={
            "Name": "Zawodnik",
            "Team": "Zespół",
            "DateMid": "Data",
            "PlayerIntensityIndex": "Indeks intensywności zawodnika (PII)",
            "PII_vs_team_avg": "Różnica PII vs średnia C1",
        })
        st.dataframe(view_disp, use_container_width=True)

    elif mode == "Ranking per data":
        dates = sorted(df["DateMid"].unique().tolist(), reverse=True)
        pick = st.multiselect("Daty", dates, default=dates[:3], key="idx_dates_fix")
        top_n = st.slider("Top N na datę", 3, 20, 10, key="idx_top_date_fix")
        if not pick:
            st.info("Wybierz co najmniej jedną datę.")
        else:
            out = []
            for d in pick:
                tmp = (
                    df[df["DateMid"] == d]
                    .sort_values("PlayerIntensityIndex", ascending=False)
                    .head(top_n)
                )
                tmp = tmp.assign(_Date=d)
                out.append(tmp)
            view = pd.concat(out, ignore_index=True)[
                ["DateMid", "Name", "Team", "PlayerIntensityIndex", "PII_vs_team_avg"]
            ]
            view_disp = view.rename(columns={
                "DateMid": "Data",
                "Name": "Zawodnik",
                "Team": "Zespół",
                "PlayerIntensityIndex": "Indeks PII",
                "PII_vs_team_avg": "Różnica PII vs C1",
            })
            st.dataframe(view_disp, use_container_width=True)

            chart = (
                alt.Chart(view.rename(columns={"PlayerIntensityIndex": "Value"}))
                .mark_bar()
                .encode(
                    y=alt.Y("Name:N", sort="-x", title="Zawodnik"),
                    x=alt.X("Value:Q", title="Indeks"),
                    color=alt.Color("Team:N", title="Zespół"),
                    column=alt.Column("DateMid:T", header=alt.Header(labelAngle=0)),
                )
                .properties(height=24 * min(top_n, len(view["Name"].unique())))
            )
            st.altair_chart(chart, use_container_width=True)

    elif mode == "Ranking per zespół":
        teams = sorted(df["Team"].dropna().unique().tolist())
        pick = st.multiselect("Zespoły", teams, default=teams[:3], key="idx_teams_fix")
        top_n = st.slider("Top N na zespół", 3, 20, 10, key="idx_top_team_fix")
        if not pick:
            st.info("Wybierz co najmniej jeden zespół.")
        else:
            out = []
            for t in pick:
                tmp = (
                    df[df["Team"] == t]
                    .sort_values("PlayerIntensityIndex", ascending=False)
                    .head(top_n)
                )
                out.append(tmp)
            view = pd.concat(out, ignore_index=True)[
                ["Team", "Name", "DateMid", "PlayerIntensityIndex", "PII_vs_team_avg"]
            ]
            view_disp = view.rename(columns={
                "Team": "Zespół",
                "Name": "Zawodnik",
                "DateMid": "Data",
                "PlayerIntensityIndex": "Indeks PII",
                "PII_vs_team_avg": "Różnica PII vs C1",
            })
            st.dataframe(view_disp, use_container_width=True)

            chart = (
                alt.Chart(view.rename(columns={"PlayerIntensityIndex": "Value"}))
                .mark_bar()
                .encode(
                    y=alt.Y("Name:N", sort="-x", title="Zawodnik"),
                    x=alt.X("Value:Q", title="Indeks"),
                    color=alt.Color("Team:N", title="Zespół"),
                    column=alt.Column("Team:N", header=alt.Header(labelAngle=0)),
                )
                .properties(height=24 * min(top_n, len(view["Name"].unique())))
            )
            st.altair_chart(chart, use_container_width=True)

    else:
        teams = sorted(df["Team"].dropna().unique().tolist())
        t = st.selectbox("Zespół (opcjonalnie)", ["(wszystkie)"] + teams, key="idx_cmp_team_fix")
        base = df if t == "(wszystkie)" else df[df["Team"] == t]

        all_names = sorted(base["Name"].dropna().unique().tolist())
        names = st.multiselect("Zawodnicy", all_names, default=all_names[:3], key="idx_cmp_names_fix")
        if not names:
            st.info("Wybierz zawodników.")
            st.stop()

        subset = base[base["Name"].isin(names)].copy()

        metric = st.radio(
            "Metryka",
            ["PlayerIntensityIndex", "PII_vs_team_avg"],
            horizontal=True,
            key="idx_cmp_metric_fix",
        )

        src_df = subset[["Name", "Team", "DateMid", metric]].copy()
        plot = src_df.rename(columns={metric: "Value"}).dropna(subset=["Value"])

        line = (
            alt.Chart(plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("DateMid:T", title="Data"),
                y=alt.Y("Value:Q", title=metric),
                color=alt.Color("Name:N", title="Zawodnik"),
                tooltip=["DateMid:T", "Name:N", "Team:N", "Value:Q"],
            )
            .properties(height=420)
        )
        st.altair_chart(line, use_container_width=True)

        table = (
            subset.groupby("Name", as_index=False)[["PlayerIntensityIndex", "PII_vs_team_avg"]]
            .mean()
            .sort_values(metric, ascending=False)
        )
        table_disp = table.rename(columns={
            "Name": "Zawodnik",
            "PlayerIntensityIndex": "Indeks PII",
            "PII_vs_team_avg": "Różnica PII vs C1",
        })
        st.dataframe(table_disp, use_container_width=True)


# ===================== PROFIL ZAWODNIKA =====================
elif page == "Profil zawodnika":
    st.subheader("Profil zawodnika – pełny przegląd")

    players_df = fetch_df("SELECT Name, Team, Position FROM players ORDER BY Name;")
    players_list = players_df["Name"].tolist() if not players_df.empty else []
    p = st.selectbox("Zawodnik", players_list if players_list else [], key="prof_all_player")
    if not p:
        st.info("Brak zawodników w rejestrze.")
        st.stop()

    prow = players_df[players_df["Name"] == p].iloc[0] if not players_df.empty else None
    team_label = str(prow["Team"]) if prow is not None else "—"
    pos_str = str(prow["Position"] or "—")
    st.caption(f"**Zespół:** {team_label}   •   **Domyślne pozycje:** {pos_str}")

    q_moto = """
        SELECT *
        FROM motoryka_stats
        WHERE Name = :name
        ORDER BY DateStart;
    """
    moto = fetch_df(q_moto, {"name": p})

    q_fant = """
        SELECT *
        FROM fantasypasy_stats
        WHERE Name = :name
        ORDER BY DateStart;
    """
    fant = fetch_df(q_fant, {"name": p})

    if moto is None or moto.empty:
        moto = pd.DataFrame()
    if fant is None or fant.empty:
        fant = pd.DataFrame()

    per_min_cols = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

    if not moto.empty:
        moto = moto.copy()
        moto["Minutes"] = pd.to_numeric(moto.get("Minutes"), errors="coerce").replace(0, np.nan)

        for m in per_min_cols:
            if m in moto.columns:
                moto[m + "_per_min"] = pd.to_numeric(moto[m], errors="coerce") / moto["Minutes"]
            else:
                moto[m + "_per_min"] = np.nan

        mid_m = pd.to_datetime(moto["DateStart"]) + (
            pd.to_datetime(moto["DateEnd"]) - pd.to_datetime(moto["DateStart"])
        ) / 2
        moto["DateMid"] = mid_m.dt.date

        periods = get_periods_df()

        moto["RangeFallback"] = (
            moto["DateStart"].astype(str) + " → " + moto["DateEnd"].astype(str)
        )

        if not periods.empty:
            moto = moto.merge(
                periods,
                on=["DateStart", "DateEnd"],
                how="left"
            )
            moto["RangeLabel"] = moto["Label"].where(
                moto["Label"].notna(), moto["RangeFallback"]
            )
        else:
            moto["RangeLabel"] = moto["RangeFallback"]

        moto["PlayerIntensityIndex"] = pd.to_numeric(
            moto.get("PlayerIntensityIndex"),
            errors="coerce"
        )

        try:
            all_m = load_motoryka_all(None, None, None)
        except Exception:
            all_m = pd.DataFrame()

        if all_m is not None and not all_m.empty and "PlayerIntensityIndex" in all_m.columns:
            all_m = all_m.copy()
            all_m["PlayerIntensityIndex"] = pd.to_numeric(
                all_m["PlayerIntensityIndex"], errors="coerce"
            )

            team_mean = (
                all_m.groupby(["Team", "DateStart", "DateEnd"], as_index=False)["PlayerIntensityIndex"]
                .mean()
                .rename(columns={"PlayerIntensityIndex": "Team_PII_mean"})
            )

            moto = moto.merge(
                team_mean,
                on=["Team", "DateStart", "DateEnd"],
                how="left"
            )

            moto["PII_vs_team_avg"] = moto["PlayerIntensityIndex"] - moto["Team_PII_mean"]
        else:
            moto["PII_vs_team_avg"] = np.nan

    if not fant.empty:
        fant = fant.copy()
        fant["Minutes"] = pd.to_numeric(fant.get("Minutes"), errors="coerce")
        mid_f = pd.to_datetime(fant["DateStart"]) + (
            pd.to_datetime(fant["DateEnd"]) - pd.to_datetime(fant["DateStart"])
        ) / 2
        fant["DateMid"] = mid_f.dt.date

    if not moto.empty:
        mecze_m = pd.to_numeric(moto.get("NumberOfGames"), errors="coerce").sum()
        min_m = pd.to_numeric(moto.get("Minutes"), errors="coerce").sum()
        mecze_m = int(mecze_m) if pd.notna(mecze_m) else 0
        min_m = int(min_m) if pd.notna(min_m) else 0
        st.caption(f"MOTORYKA: mecze = {mecze_m}, minuty = {min_m}")

    if not fant.empty:
        mecze_f = pd.to_numeric(fant.get("NumberOfGames"), errors="coerce").sum()
        min_f = pd.to_numeric(fant.get("Minutes"), errors="coerce").sum()
        mecze_f = int(mecze_f) if pd.notna(mecze_f) else 0
        min_f = int(min_f) if pd.notna(min_f) else 0
        st.caption(f"FANTASYPASY: mecze = {mecze_f}, minuty = {min_f}")

    tabs_prof = st.tabs(["Motoryka", "Indeks", "FANTASYPASY", "Tabele i eksport"])

    # 1) MOTORYKA
    with tabs_prof[0]:
        if moto.empty:
            st.info("Brak danych motorycznych.")
        else:
            st.markdown("### Metryki motoryczne na minutę – przebieg w czasie")

            pick_m = st.multiselect(
                "Metryki (na minutę)",
                [m + "_per_min" for m in per_min_cols],
                default=["TD_m_per_min"],
                key="prof_all_moto_metrics"
            )

            if pick_m:
                plot_m = (
                    moto[["DateMid"] + pick_m]
                    .melt(id_vars="DateMid", var_name="Metric", value_name="Value")
                    .dropna(subset=["Value"])
                )

                chart_m = (
                    alt.Chart(plot_m)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("DateMid:T", title="Data"),
                        y=alt.Y("Value:Q", title="Wartość"),
                        color=alt.Color("Metric:N", title="Metryka"),
                        tooltip=["DateMid:T", "Metric:N", "Value:Q"],
                    )
                    .properties(height=420)
                )

                st.altair_chart(chart_m, use_container_width=True)

            st.markdown("### Podsumowanie metryk na zakres pomiaru")
            if "RangeLabel" in moto.columns:
                agg_cols = [c for c in moto.columns if c.endswith("_per_min")]
                if agg_cols:
                    summary = (
                        moto.groupby("RangeLabel")[agg_cols]
                        .agg(["mean", "median"])
                        .round(2)
                    )
                    st.dataframe(summary, use_container_width=True)

            st.markdown("---")
            st.subheader("Zawodnik vs C1 – per minuta (globalnie)")

            try:
                df_all_for_ref = load_motoryka_all(None, None, None).copy()
            except Exception:
                df_all_for_ref = pd.DataFrame()

            if df_all_for_ref.empty or df_all_for_ref[df_all_for_ref["Team"] == "C1"].empty:
                st.info("Brak danych referencyjnych C1.")
            else:
                df_all_for_ref["Minutes"] = pd.to_numeric(
                    df_all_for_ref["Minutes"], errors="coerce"
                ).replace(0, np.nan)

                for m in per_min_cols:
                    if m in df_all_for_ref.columns:
                        df_all_for_ref[m + "_per_min"] = (
                            pd.to_numeric(df_all_for_ref[m], errors="coerce")
                            / df_all_for_ref["Minutes"]
                        )

                ref_c1 = df_all_for_ref[df_all_for_ref["Team"] == "C1"]
                player_m = moto.copy()

                cols_per_min = [c + "_per_min" for c in per_min_cols if c + "_per_min" in player_m.columns]

                if cols_per_min:
                    player_stats = player_m[cols_per_min].mean().to_frame("Player_mean")
                    c1_stats = ref_c1[cols_per_min].mean().to_frame("C1_mean")

                    comp = player_stats.join(c1_stats, how="inner")
                    comp["diff (Player - C1)"] = comp["Player_mean"] - comp["C1_mean"]
                    st.dataframe(comp.round(3), use_container_width=True)

    # 2) INDEKS
    with tabs_prof[1]:
        if moto.empty or "PII_vs_team_avg" not in moto.columns:
            st.info("Brak danych indeksu intensywności lub PII_vs_team_avg.")
        else:
            st.markdown("### Player Intensity Index – przebieg i odniesienie do zespołu")

            chart_pii = (
                alt.Chart(
                    moto[["DateMid", "PlayerIntensityIndex", "PII_vs_team_avg"]]
                    .dropna(subset=["DateMid"])
                )
                .transform_fold(
                    ["PlayerIntensityIndex", "PII_vs_team_avg"],
                    as_=["Metric", "Value"]
                )
                .mark_line(point=True)
                .encode(
                    x=alt.X("DateMid:T", title="Data"),
                    y=alt.Y("Value:Q", title="Wartość"),
                    color=alt.Color("Metric:N", title="Metryka"),
                    tooltip=["DateMid:T", "Metric:N", "Value:Q"],
                )
                .properties(height=420)
            )
            st.altair_chart(chart_pii, use_container_width=True)

            st.markdown("### PII_vs_team_avg – najlepsze i najsłabsze okresy")
            if "RangeLabel" in moto.columns:
                agg_pii = (
                    moto.groupby("RangeLabel")["PII_vs_team_avg"]
                    .mean()
                    .dropna()
                    .sort_values(ascending=False)
                )
                c1, c2 = st.columns(2)
                if not agg_pii.empty:
                    c1.write("Top 5 okresów")
                    c1.dataframe(agg_pii.head(5).to_frame("PII_vs_team_avg").round(3))
                    c2.write("Bottom 5 okresów")
                    c2.dataframe(agg_pii.tail(5).to_frame("PII_vs_team_avg").round(3))

    # 3) FANTASYPASY
    with tabs_prof[2]:
        fant_metrics = [
            "PktOff", "PktDef", "Goal", "Assist", "ChanceAssist", "KeyPass",
            "KeyLoss", "DuelLossInBox", "MissBlockShot", "Finalization",
            "KeyIndividualAction", "KeyRecover", "DuelWinInBox", "BlockShot"
        ]
        if fant.empty:
            st.info("Brak danych FANTASYPASY.")
        else:
            st.markdown("### FANTASYPASY – przebieg metryk w czasie")
            pick_f = st.multiselect(
                "Metryki",
                fant_metrics,
                default=["PktOff", "PktDef"],
                key="prof_all_fant_metrics"
            )
            if pick_f:
                plot_f = (
                    fant[["DateMid"] + pick_f]
                    .melt(id_vars="DateMid", var_name="Metric", value_name="Value")
                    .dropna(subset=["Value"])
                )
                chart_f = (
                    alt.Chart(plot_f)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("DateMid:T", title="Data"),
                        y=alt.Y("Value:Q", title="Wartość"),
                        color=alt.Color("Metric:N", title="Metryka"),
                        tooltip=["DateMid:T", "Metric:N", "Value:Q"],
                    )
                    .properties(height=420)
                )
                st.altair_chart(chart_f, use_container_width=True)

            st.markdown("### Podsumowanie FANTASYPASY (średnie na mecz)")
            cols_num = [c for c in fant_metrics if c in fant.columns]
            if cols_num:
                summary_f = fant[cols_num].mean().to_frame("mean_per_game").round(2)
                st.dataframe(summary_f, use_container_width=True)

    # 4) Tabele i eksport
    with tabs_prof[3]:
        st.markdown("### Surowe dane – motoryka")
        if moto.empty:
            st.info("Brak danych motorycznych.")
        else:
            st.dataframe(
                moto.sort_values("DateMid", ascending=True),
                use_container_width=True
            )

        st.markdown("### Surowe dane – FANTASYPASY")
        if fant.empty:
            st.info("Brak danych FANTASYPASY.")
        else:
            st.dataframe(
                fant.sort_values("DateMid", ascending=True),
                use_container_width=True
            )

        if moto.empty and fant.empty:
            st.info("Brak danych do eksportu profilu zawodnika.")
        else:
            excel_buffer = build_player_excel_report(p, moto, fant)
            st.download_button(
                label="📊 Pobierz profil zawodnika (EXCEL)",
                data=excel_buffer,
                file_name=f"profil_{p}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="prof_all_excel_download",
            )


# ============================================================
#                     SEKCJA: OVER / UNDER
# ============================================================
elif sekcja == "Over/Under":
    st.title("Over/Under – analiza PII i metryk względem zespołu")

    try:
        df = load_motoryka_all(None, None, None)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        st.info("Brak danych motorycznych.")
        st.stop()

    df = df.copy()
    df["PlayerIntensityIndex"] = pd.to_numeric(df.get("PlayerIntensityIndex"), errors="coerce")
    df["Minutes"] = pd.to_numeric(df.get("Minutes"), errors="coerce").replace(0, np.nan)
    df["DateStart"] = pd.to_datetime(df.get("DateStart"), errors="coerce").dt.date
    df["DateEnd"] = pd.to_datetime(df.get("DateEnd"), errors="coerce").dt.date

    df = df.dropna(subset=["Name", "Team"])

    tab_global_pii, tab_metrics, tab_periods = st.tabs([
        "Globalne Over/Under (PII)",
        "Metryki per minuta",
        "Okresy pomiarowe (PII – tabela)",
    ])

    # TAB 1
    with tab_global_pii:
        st.subheader("Over/Under na podstawie PII – zawodnik vs zespoły")

        pii_df = df.dropna(subset=["PlayerIntensityIndex"]).copy()

        if pii_df.empty:
            st.info("Brak danych PII.")
            st.stop()

        players_list = sorted(pii_df["Name"].unique().tolist())
        if not players_list:
            st.info("Brak zawodników z danymi PII.")
            st.stop()

        player = st.selectbox(
            "Zawodnik",
            players_list,
            key="ou_global_pii_player",
        )

        player_pii_all = (
            pii_df[pii_df["Name"] == player]
            .dropna(subset=["DateStart", "DateEnd"])
            .copy()
        )

        if player_pii_all.empty:
            st.info("Brak danych PII dla wybranego zawodnika (brak okresów pomiarowych).")
            st.stop()

        st.markdown("### Widok 1 – wybrany okres: zawodnik vs wszystkie zespoły")

        periods = (
            player_pii_all[["DateStart", "DateEnd"]]
            .drop_duplicates()
            .sort_values(["DateStart", "DateEnd"])
        )
        periods["Label"] = periods["DateStart"].astype(str) + " → " + periods["DateEnd"].astype(str)

        period_option = st.selectbox(
            "Okres pomiaru (tylko daty, w których ten zawodnik ma pomiary motoryki)",
            options=periods["Label"].tolist(),
            key="ou_global_period_select",
        )

        sel = periods[periods["Label"] == period_option].iloc[0]
        p_start = sel["DateStart"]
        p_end = sel["DateEnd"]

        player_df = player_pii_all[
            (player_pii_all["DateStart"] == p_start) &
            (player_pii_all["DateEnd"] == p_end)
        ].copy()

        if player_df.empty:
            st.caption(f"Wybrany okres: **{p_start} → {p_end}**")
            st.info("Brak danych PII zawodnika w wybranym okresie.")
            st.stop()

        player_team = (
            player_df["Team"].dropna().mode().iloc[0]
            if not player_df["Team"].dropna().empty
            else None
        )

        player_mean_pii = player_df["PlayerIntensityIndex"].mean()

        team_pii = (
            pii_df[
                (pii_df["DateStart"] == p_start) &
                (pii_df["DateEnd"] == p_end)
            ]
            .groupby("Team")["PlayerIntensityIndex"]
            .mean()
            .reset_index()
            .rename(columns={"PlayerIntensityIndex": "PII_team"})
        )

        comp = team_pii.copy()
        comp["PII_player"] = player_mean_pii
        comp["diff_abs"] = comp["PII_player"] - comp["PII_team"]
        comp["diff_pct"] = (comp["PII_player"] / comp["PII_team"] - 1.0) * 100.0

        comp["diff_pct_abs"] = comp["diff_pct"].abs()
        closest_team_name = None
        closest_diff_pct = None
        if not comp.empty:
            idx_closest = comp["diff_pct_abs"].idxmin()
            closest_team_name = comp.loc[idx_closest, "Team"]
            closest_diff_pct = comp.loc[idx_closest, "diff_pct"]

        comp["Is_player_team"] = comp["Team"].apply(
            lambda t: "TAK" if (player_team is not None and t == player_team) else ""
        )

        threshold = st.slider(
            "Próg (%) – widok 1",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="ou_global_pii_threshold",
        )

        comp["Status"] = comp["diff_pct"].apply(
            lambda x: "Over" if x >= threshold else ("Under" if x <= -threshold else "Neutral")
        )

        team_order = ["C1", "C2", "U-17", "U-19"]

        comp["Team_order"] = comp["Team"].apply(
            lambda x: team_order.index(x) if x in team_order else 999
        )
        comp = comp.sort_values("Team_order").drop(columns=["Team_order"])

        st.caption(
            f"Wybrany okres: **{p_start} → {p_end}**  |  Zawodnik: **{player}**"
        )

        st.markdown(
            f"**Przypisany zespół zawodnika (na podstawie pomiarów w tym okresie):** "
            f"`{player_team or 'brak'}`"
        )

        if closest_team_name is not None:
            st.markdown(
                f"**Najbardziej zbliżony do wyniku zawodnika w tym pomiarze jest zespół:** "
                f"**{closest_team_name}** "
                f"(różnica: `{closest_diff_pct:.1f}` punktu procentowego)."
            )

        st.dataframe(
            comp[
                ["Team", "Is_player_team", "PII_player", "PII_team", "diff_abs", "diff_pct", "Status"]
            ]
            .rename(columns={
                "Team": "Zespół",
                "Is_player_team": "Zespół zawodnika?",
                "PII_player": "PII zawodnika",
                "PII_team": "Średnie PII zespołu",
                "diff_abs": "Różnica bezwzględna",
                "diff_pct": "Różnica (%)",
                "Status": "Status",
            })
            .round({
                "PII zawodnika": 3,
                "Średnie PII zespołu": 3,
                "Różnica bezwzględna": 3,
                "Różnica (%)": 1,
            }),
            use_container_width=True,
        )

        st.markdown(
            """
**Interpretacja różnic procentowych (Różnica (%)):**

- Wartość pokazuje, o ile procent średnie PII zawodnika różnią się od średniego PII danego zespołu.
- Obliczamy to jako:  
  **(PII zawodnika / PII zespołu − 1) × 100%**
- **Wartości dodatnie** (np. +40%) → zawodnik ma PII **wyższe** o 40% niż średnia tego zespołu.
- **Wartości ujemne** (np. −30%) → zawodnik ma PII **niższe** o 30% niż średnia tego zespołu.
- **Status (Over / Under / Neutral)** opisuje położenie względem zadanego progu.
"""
        )

        st.markdown("### Widok 2 – wszystkie okresy: zawodnik vs wybrany zespół")

        team_options = sorted(pii_df["Team"].unique().tolist())
        compare_team = st.selectbox(
            "Zespół referencyjny",
            team_options,
            key="ou_hist_team",
        )

        team_rows = (
            pii_df[pii_df["Team"] == compare_team]
            .dropna(subset=["DateStart", "DateEnd"])
            .copy()
        )
        player_rows = player_pii_all.copy()

        periods_hist = pd.merge(
            player_rows[["DateStart", "DateEnd"]].drop_duplicates(),
            team_rows[["DateStart", "DateEnd"]].drop_duplicates(),
            on=["DateStart", "DateEnd"],
            how="inner",
        )

        if periods_hist.empty:
            st.info("Brak wspólnych okresów pomiarowych dla tego zawodnika i wybranego zespołu.")
        else:
            player_per_period = (
                player_rows
                .groupby(["DateStart", "DateEnd"])["PlayerIntensityIndex"]
                .mean()
                .reset_index()
                .rename(columns={"PlayerIntensityIndex": "PII_player"})
            )

            team_per_period = (
                team_rows
                .groupby(["DateStart", "DateEnd"])["PlayerIntensityIndex"]
                .mean()
                .reset_index()
                .rename(columns={"PlayerIntensityIndex": "PII_team"})
            )

            hist = (
                periods_hist
                .merge(player_per_period, on=["DateStart", "DateEnd"], how="left")
                .merge(team_per_period, on=["DateStart", "DateEnd"], how="left")
            )

            hist = hist.dropna(subset=["PII_player", "PII_team"]).copy()

            if hist.empty:
                st.info("Brak kompletnych danych PII dla wspólnych okresów.")
            else:
                hist["diff_abs"] = hist["PII_player"] - hist["PII_team"]
                hist["diff_pct"] = (hist["PII_player"] / hist["PII_team"] - 1.0) * 100.0
                hist["Label"] = (
                    hist["DateStart"].astype(str) + " → " + hist["DateEnd"].astype(str)
                )

                threshold_hist = st.slider(
                    "Próg (%) – widok 2 (historia)",
                    min_value=0.0,
                    max_value=50.0,
                    value=5.0,
                    step=1.0,
                    key="ou_hist_threshold",
                )

                hist["Status"] = hist["diff_pct"].apply(
                    lambda x: "Over" if x >= threshold_hist
                    else ("Under" if x <= -threshold_hist else "Neutral")
                )

                hist = hist.sort_values(["DateStart", "DateEnd"])

                st.dataframe(
                    hist[
                        [
                            "Label",
                            "PII_player",
                            "PII_team",
                            "diff_abs",
                            "diff_pct",
                            "Status",
                        ]
                    ]
                    .rename(columns={
                        "Label": "Okres",
                        "PII_player": "PII zawodnika",
                        "PII_team": "Średnie PII zespołu",
                        "diff_abs": "Różnica bezwzględna",
                        "diff_pct": "Różnica (%)",
                        "Status": "Status",
                    })
                    .round({
                        "PII zawodnika": 3,
                        "Średnie PII zespołu": 3,
                        "Różnica bezwzględna": 3,
                        "Różnica (%)": 1,
                    }),
                    use_container_width=True,
                )

        st.markdown("### Widok 3 – przypisany zespół dla każdego okresu pomiarowego zawodnika")

        summary_rows = []

        periods_all = (
            player_pii_all[["DateStart", "DateEnd"]]
            .drop_duplicates()
            .sort_values(["DateStart", "DateEnd"])
        )

        for _, row_p in periods_all.iterrows():
            ds = row_p["DateStart"]
            de = row_p["DateEnd"]

            p_df = player_pii_all[
                (player_pii_all["DateStart"] == ds) &
                (player_pii_all["DateEnd"] == de)
            ]
            if p_df.empty:
                continue

            p_mean = p_df["PlayerIntensityIndex"].mean()

            t_df = (
                pii_df[
                    (pii_df["DateStart"] == ds) &
                    (pii_df["DateEnd"] == de)
                ]
                .groupby("Team")["PlayerIntensityIndex"]
                .mean()
                .reset_index()
                .rename(columns={"PlayerIntensityIndex": "PII_team"})
            )

            if t_df.empty:
                continue

            t_df["PII_player"] = p_mean
            t_df["diff_pct"] = (t_df["PII_player"] / t_df["PII_team"] - 1.0) * 100.0
            t_df["diff_pct_abs"] = t_df["diff_pct"].abs()

            idx_best = t_df["diff_pct_abs"].idxmin()
            best = t_df.loc[idx_best]

            summary_rows.append({
                "Okres": f"{ds} → {de}",
                "Zespół najbliżej zawodnika": best["Team"],
                "PII zawodnika": p_mean,
                "Średnie PII zespołu": best["PII_team"],
                "Różnica (%)": best["diff_pct"],
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df = summary_df.sort_values("Okres")

            st.dataframe(
                summary_df[
                    [
                        "Okres",
                        "Zespół najbliżej zawodnika",
                        "PII zawodnika",
                        "Średnie PII zespołu",
                        "Różnica (%)",
                    ]
                ].round({
                    "PII zawodnika": 3,
                    "Średnie PII zespołu": 3,
                    "Różnica (%)": 1,
                }),
                use_container_width=True,
            )
        else:
            st.info("Brak danych do zbudowania tabeli przypisanych zespołów dla zawodnika.")

    # TAB 2 – METRYKI PER MINUTA
    with tab_metrics:
        st.subheader("Over/Under – metryki per minuta względem zespołu")

        per_min_cols = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

        df2 = df.copy()
        for c in per_min_cols:
            df2[c + "_per_min"] = (
                pd.to_numeric(df2.get(c), errors="coerce") / df2["Minutes"]
            ).replace([np.inf, -np.inf], np.nan)

        metrics = [m + "_per_min" for m in per_min_cols]
        metrics_pick = st.multiselect(
            "Metryki",
            metrics,
            default=metrics,
            key="ou_metrics_pick",
        )

        if not metrics_pick:
            st.info("Wybierz przynajmniej jedną metrykę.")
            st.stop()

        pm = (
            df2.groupby(["Team", "Name"])[metrics_pick]
            .mean()
            .reset_index()
        )

        tm = (
            df2.groupby("Team")[metrics_pick]
            .mean()
            .reset_index()
            .rename(columns={m: m + "_team" for m in metrics_pick})
        )

        merged_m = pm.merge(tm, on="Team", how="left")

        rows = []
        for _, row in merged_m.iterrows():
            for m in metrics_pick:
                t = row[m + "_team"]
                p = row[m]
                if pd.isna(t) or pd.isna(p) or t == 0:
                    continue
                rows.append({
                    "Team": row["Team"],
                    "Player": row["Name"],
                    "Metric": m.replace("_per_min", ""),
                    "Player_mean": p,
                    "Team_mean": t,
                    "diff_abs": p - t,
                    "diff_pct": (p / t - 1) * 100,
                })

        out = pd.DataFrame(rows)

        if out.empty:
            st.info("Brak danych do analizy metryk per minuta.")
            st.stop()

        threshold_m = st.slider(
            "Próg (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="ou_metrics_threshold",
        )

        out["Status"] = out["diff_pct"].apply(
            lambda x: "Over" if x >= threshold_m else ("Under" if x <= -threshold_m else "Neutral")
        )

        st.dataframe(
            out.sort_values(["Team", "Metric", "diff_pct"], ascending=[True, True, False])
            .rename(columns={
                "Team": "Zespół",
                "Player": "Zawodnik",
                "Metric": "Metryka",
                "Player_mean": "Średnia zawodnika (na min)",
                "Team_mean": "Średnia zespołu (na min)",
                "diff_abs": "Różnica bezwzględna",
                "diff_pct": "Różnica (%)",
                "Status": "Status",
            })
            .round({
                "Średnia zawodnika (na min)": 3,
                "Średnia zespołu (na min)": 3,
                "Różnica bezwzględna": 3,
                "Różnica (%)": 1,
            }),
            use_container_width=True,
        )

    # TAB 3 – OKRESY POMIAROWE
    with tab_periods:
        st.subheader("PII – Over/Under w wybranych okresach pomiarowych (tabela)")

        df_pii = df.dropna(subset=["PlayerIntensityIndex", "DateStart", "DateEnd"]).copy()

        if df_pii.empty:
            st.info("Brak danych PII z kompletnymi okresami pomiarowymi.")
            st.stop()

        periods = (
            df_pii[["DateStart", "DateEnd"]]
            .drop_duplicates()
            .sort_values(["DateStart", "DateEnd"])
        )
        periods["Label"] = periods["DateStart"].astype(str) + " → " + periods["DateEnd"].astype(str)

        option = st.selectbox(
            "Okres pomiaru (tylko daty, w których SĄ pomiary motoryki)",
            options=periods["Label"].tolist(),
            key="ou_period_select",
        )

        row_sel = periods[periods["Label"] == option].iloc[0]
        d_start = row_sel["DateStart"]
        d_end = row_sel["DateEnd"]

        st.caption(f"Wybrany okres: **{d_start} → {d_end}**")

        df_period = df_pii[
            (df_pii["DateStart"] == d_start) &
            (df_pii["DateEnd"] == d_end)
        ].copy()

        if df_period.empty:
            st.info("Brak danych PII dla wybranego okresu.")
            st.stop()

        player_pii_p = (
            df_period
            .groupby(["Team", "Name"])["PlayerIntensityIndex"]
            .mean()
            .reset_index()
            .rename(columns={"PlayerIntensityIndex": "PII_player"})
        )

        team_pii_p = (
            df_period
            .groupby("Team")["PlayerIntensityIndex"]
            .mean()
            .reset_index()
            .rename(columns={"PlayerIntensityIndex": "PII_team"})
        )

        merged_p = player_pii_p.merge(team_pii_p, on="Team", how="left")

        merged_p["diff_abs"] = merged_p["PII_player"] - merged_p["PII_team"]
        merged_p["diff_pct"] = (
            merged_p["PII_player"] / merged_p["PII_team"] - 1.0
        ) * 100.0

        threshold_p = st.slider(
            "Próg Over/Under dla PII w tym okresie (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="ou_period_threshold",
        )

        merged_p["Status"] = merged_p["diff_pct"].apply(
            lambda x: "Over" if x >= threshold_p else ("Under" if x <= -threshold_p else "Neutral")
        )

        status_filter = st.multiselect(
            "Których zawodników pokazać?",
            options=["Over", "Under", "Neutral"],
            default=["Over", "Under"],
            key="ou_period_status_filter",
        )

        team_filter = st.multiselect(
            "Filtr zespołów",
            options=sorted(merged_p["Team"].unique().tolist()),
            key="ou_period_team_filter",
        )

        view_p = merged_p.copy()
        if status_filter:
            view_p = view_p[view_p["Status"].isin(status_filter)]
        if team_filter:
            view_p = view_p[view_p["Team"].isin(team_filter)]

        if view_p.empty:
            st.info("Brak zawodników spełniających kryteria w tym okresie.")
            st.stop()

        view_p = view_p.sort_values(["Team", "diff_pct"], ascending=[True, False])

        st.markdown("### Tabela – zawodnicy Over/Under w wybranym okresie (PII vs zespół)")
        st.dataframe(
            view_p[
                ["Team", "Name", "PII_player", "PII_team", "diff_abs", "diff_pct", "Status"]
            ]
            .rename(columns={
                "Team": "Zespół",
                "Name": "Zawodnik",
                "PII_player": "PII zawodnika",
                "PII_team": "Średnie PII zespołu",
                "diff_abs": "Różnica bezwzględna",
                "diff_pct": "Różnica (%)",
                "Status": "Status",
            })
            .round({
                "PII zawodnika": 3,
                "Średnie PII zespołu": 3,
                "Różnica bezwzględna": 3,
                "Różnica (%)": 1,
            }),
            use_container_width=True,
        )

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO

# ===================== STAŁE =====================
POS_OPTIONS = [
    "ŚO", "LŚO", "PŚO", "LO",
    "ŚPD", "8", "ŚP", "ŚPO", "10",
    "PW", "LW", "WAHADŁO", "NAPASTNIK"
]

DEFAULT_TEAMS = ["C1", "C2", "U-19", "U-17"]

# ===================== DANE Z GITHUBA =====================
BASE_CSV_URL = "https://raw.githubusercontent.com/GrzegorzTarasek/cracovia-app/main/dane/"
TABLES: dict[str, pd.DataFrame] = {}

def _load_csv(fname: str) -> pd.DataFrame:
    url = BASE_CSV_URL + fname
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_motoryka_table():
    if "motoryka_stats" not in TABLES:
        TABLES["motoryka_stats"] = _load_csv("motoryka_stats.csv")
    return TABLES["motoryka_stats"]

@st.cache_data(show_spinner=False)
def load_fantasy_table():
    if "fantasypasy_stats" not in TABLES:
        TABLES["fantasypasy_stats"] = _load_csv("fantasypasy_stats.csv")
    return TABLES["fantasypasy_stats"]

@st.cache_data(show_spinner=False)
def load_players_table():
    if "players" not in TABLES:
        TABLES["players"] = _load_csv("players_table.csv")
    return TABLES["players"]

@st.cache_data(show_spinner=False)
def load_teams_table():
    if "teams" not in TABLES:
        TABLES["teams"] = _load_csv("teams.csv")
    return TABLES["teams"]

@st.cache_data(show_spinner=False)
def load_periods_table():
    if "measurement_periods" not in TABLES:
        TABLES["measurement_periods"] = _load_csv("measurement_periods.csv")
    return TABLES["measurement_periods"]

@st.cache_data(show_spinner=False)
def get_periods_df():
    """
    Zwraca okresy pomiarowe measurement_periods w formacie:
    Label, DateStart, DateEnd
    (PeriodID jest opcjonalne – jeśli go nie ma w CSV, aplikacja i tak działa)
    """
    try:
        df = load_periods_table().copy()
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["Label", "DateStart", "DateEnd"])

    # konwersja dat
    if "DateStart" in df.columns:
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce").dt.date
    if "DateEnd" in df.columns:
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce").dt.date

    # kolumny dostępne — bez wymuszania PeriodID!
    expected = ["PeriodID", "Label", "DateStart", "DateEnd"]
    cols = [c for c in expected if c in df.columns]

    df = df[cols].dropna(subset=["DateStart", "DateEnd"]).drop_duplicates()
    df = df.sort_values(["DateStart", "DateEnd"], ascending=[False, False])

    return df

def build_player_excel_report(player_name: str, moto: pd.DataFrame, fant: pd.DataFrame):
    """
    Buduje raport EXCEL dla profilu zawodnika.
    Zwraca bytes buffer gotowy do download_button.
    """
    import io
    import numpy as np
    from pandas import ExcelWriter

    def safe_int(x):
        """Konwertuje sumę na int, traktując NaN jako 0."""
        try:
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return 0
            return int(x)
        except Exception:
            return 0

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # ARKUSZ 1 – MOTORYKA
        if moto is not None and not moto.empty:
            moto_sorted = moto.sort_values("DateMid", ascending=True)
            moto_sorted.to_excel(writer, index=False, sheet_name="Motoryka")

        # ARKUSZ 2 – FANTASYPASY
        if fant is not None and not fant.empty:
            fant_sorted = fant.sort_values("DateMid", ascending=True)
            fant_sorted.to_excel(writer, index=False, sheet_name="FANTASYPASY")

        # ARKUSZ 3 – PODSUMOWANIE
        summary_rows = []

        # MOTORYKA - MECZE + MINUTY
        if moto is not None and not moto.empty:
            mecze = pd.to_numeric(moto.get("NumberOfGames"), errors="coerce").sum()
            minuty = pd.to_numeric(moto.get("Minutes"), errors="coerce").sum()

            summary_rows.append({
                "Kategoria": "Motoryka",
                "Mecze": safe_int(mecze),
                "Minuty": safe_int(minuty)
            })

        # FANTASYPASY - MECZE + MINUTY
        if fant is not None and not fant.empty:
            mecze = pd.to_numeric(fant.get("NumberOfGames"), errors="coerce").sum()
            minuty = pd.to_numeric(fant.get("Minutes"), errors="coerce").sum()

            summary_rows.append({
                "Kategoria": "FANTASYPASY",
                "Mecze": safe_int(mecze),
                "Minuty": safe_int(minuty)
            })

        # Zapis do arkusza
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, index=False, sheet_name="Podsumowanie")

    output.seek(0)
    return output




# ============================================================
#                 POPRAWIONA FUNKCJA fetch_df
# ============================================================
def fetch_df(sql: str, params=None):

    if sql is None:
        return pd.DataFrame()

    params = params or {}
    s = " ".join(sql.split()).lower()

    # --- TEAMS ---
    if "from teams" in s and "team" in s:
        df = load_teams_table()
        if "Team" not in df.columns:
            return pd.DataFrame(columns=["Team"])
        return df[["Team"]].dropna().drop_duplicates().sort_values("Team")

    # --- PLAYERS (NAME ONLY) ---
    if "from players" in s and "team" not in s and "name" in s:
        df = load_players_table()
        if "Name" not in df.columns:
            return pd.DataFrame(columns=["Name"])
        return df[["Name"]].dropna().drop_duplicates().sort_values("Name")

    # ---- LISTA ZAWODNIKÓW Z TEAM & POSITION ----
    # SELECT Name, Team, Position FROM players ORDER BY Name;
    if "from players" in s and "select" in s and "team" in s:
        df = load_players_table().copy()
        # bierzemy tylko te kolumny, które faktycznie istnieją
        cols = [c for c in ["Name", "Team", "Position"] if c in df.columns]
        if not cols:
            # w ogóle nie ma znanych kolumn → zwróć pustą ramkę
            return pd.DataFrame(columns=["Name", "Team", "Position"])

        out = df[cols].copy()

        # jeśli mamy kolumnę Name – filtrujemy po niej i sortujemy po Name
        if "Name" in out.columns:
            out = out.dropna(subset=["Name"]).sort_values("Name")
        else:
            # awaryjnie sortujemy po pierwszej dostępnej kolumnie
            out = out.dropna().sort_values(cols[0])

        return out

    # --- PERIODS ---
    if "from measurement_periods" in s:
        df = load_periods_table()
        df = df.copy()
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        return df.dropna().drop_duplicates().sort_values(["DateStart", "DateEnd"], ascending=[False, False])

    # --- DISTINCT DATES ---
    if "distinct datestart" in s and "dateend" in s:
        if "motoryka" in s:
            df = load_motoryka_table()
        else:
            df = load_fantasy_table()

        df = df.copy()
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
        df = df[["DateStart", "DateEnd"]].dropna().drop_duplicates()
        return df.sort_values(["DateStart", "DateEnd"], ascending=[False, False])

    # --- MOTORYKA ---
    if "from motoryka_stats" in s:
        df = load_motoryka_table().copy()

        if "where name" in s and "name" in params:
            df = df[df["Name"] == params["name"]]

        if "ds" in params:
            df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
            df = df[df["DateStart"] >= pd.to_datetime(params["ds"])]

        if "de" in params:
            df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
            df = df[df["DateEnd"] <= pd.to_datetime(params["de"])]

        if "team in :teams" in s and "teams" in params:
            df = df[df["Team"].isin(params["teams"])]

        return df

    # --- FANTASY DATA ---
    if "from fantasypasy_stats" in s:
        df = load_fantasy_table().copy()

        if "where name" in s and "name" in params:
            df = df[df["Name"] == params["name"]]

        if "ds" in params:
            df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
            df = df[df["DateStart"] >= pd.to_datetime(params["ds"])]

        if "de" in params:
            df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")
            df = df[df["DateEnd"] <= pd.to_datetime(params["de"])]

        if "team in :teams" in s and "teams" in params:
            df = df[df["Team"].isin(params["teams"])]

        return df

    return pd.DataFrame()

 
# ============================================================
#         POPRAWIONE POBIERANIE OKRESÓW TYLKO Z DANYMI
# ============================================================
def get_periods_for_table(source: str, teams=None):
    """
    Zwraca tylko okresy, które faktycznie występują w motoryce/fantasy
    i opcjonalnie tylko dla wybranych zespołów.
    """
    periods = load_periods_table().copy()
    if periods.empty:
        return periods

    # wybór tabeli źródłowej
    src = load_fantasy_table() if source == "FANTASYPASY" else load_motoryka_table()
    if src.empty:
        return periods.iloc[0:0]

    if teams and "Team" in src.columns:
        src = src[src["Team"].isin(teams)]
        if src.empty:
            return periods.iloc[0:0]

    # konwersja dat
    for df in [periods, src]:
        df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce").dt.date
        df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce").dt.date

    used_pairs = src[["DateStart", "DateEnd"]].dropna().drop_duplicates()

    out = periods.merge(used_pairs, on=["DateStart", "DateEnd"], how="inner")
    return out.sort_values(["DateStart", "DateEnd"], ascending=[False, False])


# ============================================================
#         FUNKCJE POMOCNICZE DLA ANALIZ
# ============================================================

def extract_positions(series: pd.Series):
    out = set()
    for v in series.dropna():
        for p in str(v).replace("\\", "/").split("/"):
            p = p.strip().upper()
            if p:
                out.add(p)
    return sorted(out)

def _explode_positions(df: pd.DataFrame):
    rows = []
    for _, r in df.iterrows():
        if "Position" not in df.columns:
            continue
        for p in str(r["Position"]).replace("\\", "/").split("/"):
            p = p.strip().upper()
            if p:
                new_r = r.copy()
                new_r["Position"] = p
                rows.append(new_r)
    return pd.DataFrame(rows) if rows else df.iloc[0:0].copy()

def _q25(x): return x.quantile(0.25)
def _q75(x): return x.quantile(0.75)

def flat_agg(df, group_cols, num_cols):
    df2 = df.copy()
    for c in num_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    agg = df2.groupby(group_cols, dropna=False)[num_cols].agg(
        ['count', 'sum', 'mean', 'median', 'std', 'min', 'max', _q25, _q75]
    )

    agg.columns = [f"{c}__{s}" for c, s in agg.columns.to_flat_index()]
    return agg.reset_index()
# ============================================================
#                 NAWIGACJA – SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("---")
    st.header("Nawigacja")

    sekcja = st.radio(
        "Sekcja",
        ["Analiza", "Profil zawodnika", "Over/Under"],
        key="nav_section",
    )

    if sekcja == "Analiza":
        page = st.radio(
            "Strona",
            [
                "Porównania",
                "Analiza (pozycje & zespoły)",
                "Wykresy zmian",
                "Indeks – porównania",
                "Fantasy – przegląd graficzny"
            ],
            key="nav_page_ana",
        )
    elif sekcja == "Over/Under":
        page = "Over/Under"
    else:
        page = "Profil zawodnika"


st.title("Cracovia – rejestry i analizy")


# ============================================================
#                    STRONA: PORÓWNANIA
# ============================================================

if page == "Porównania":
    st.subheader("Porównanie młodzieży do pierwszego zespołu C1")

    mode_cmp = st.radio(
        "Wybór zakresu",
        ["Okres/Test z rejestru", "Z istniejących par dat", "Ręcznie"],
        horizontal=True,
        key="cmp_mode",
    )

    ds_f, de_f = None, None
    periods_all = load_periods_table()
    periods_all["DateStart"] = pd.to_datetime(periods_all["DateStart"], errors="coerce")
    periods_all["DateEnd"] = pd.to_datetime(periods_all["DateEnd"], errors="coerce")

    # ===============================
    # TRYB 1: OKRES/TEST Z REJESTRU
    # ===============================
    if mode_cmp == "Okres/Test z rejestru":
        if periods_all.empty:
            st.info("Brak zapisanych okresów – wybierz inny tryb.")
        else:
            labels = [
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]"
                for _, r in periods_all.iterrows()
            ]
            pick = st.selectbox("Okres/Test", labels, index=0, key="cmp_pick_period")
            sel = periods_all.iloc[labels.index(pick)]
            ds_f, de_f = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_f.date()} → {de_f.date()}")

    # =====================================
    # TRYB 2: Z ISTNIEJĄCYCH PAR DAT (MOTO)
    # =====================================
    elif mode_cmp == "Z istniejących par dat":
        pairs = fetch_df(
            """
            SELECT DISTINCT DateStart, DateEnd
            FROM motoryka_stats
            ORDER BY DateStart DESC, DateEnd DESC
            """
        )
        if pairs.empty:
            st.info("Brak danych – wybierz „Ręcznie”.")
        else:
            pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
            pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")
            opts = [
                f"{r.DateStart.date()} → {r.DateEnd.date()}"
                for _, r in pairs.iterrows()
            ]
            pick = st.selectbox("Para dat", opts, index=0, key="cmp_pick_pair")
            sel = pairs.iloc[opts.index(pick)]
            ds_f, de_f = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_f.date()} → {de_f.date()}")

    # ===============================
    # TRYB 3: RĘCZNIE
    # ===============================
    else:
        c1, c2 = st.columns(2)
        ds_f = c1.date_input("Od (DateStart)", key="cmp_ds")
        de_f = c2.date_input("Do (DateEnd)", key="cmp_de")

    # ======================================================
    # WCZYTANIE DANYCH DO PORÓWNAŃ
    # ======================================================
    try:
        df_m = fetch_df(
            """
            SELECT Name, Team, Position, DateStart, DateEnd,
                   Minutes, HSR_m, Sprint_m, ACC, DECEL, PlayerIntensityIndex
            FROM motoryka_stats
            WHERE 1=1
            """,
            {}
        )
    except Exception:
        df_m = pd.DataFrame()

    df_m = df_m.copy()
    df_m["DateStart"] = pd.to_datetime(df_m["DateStart"], errors="coerce")
    df_m["DateEnd"] = pd.to_datetime(df_m["DateEnd"], errors="coerce")

    if ds_f:
        df_m = df_m[df_m["DateStart"] >= pd.to_datetime(ds_f)]
    if de_f:
        df_m = df_m[df_m["DateEnd"] <= pd.to_datetime(de_f)]

    if df_m.empty:
        st.info("Brak danych w wybranym zakresie.")
        st.stop()

    # ======================================================
    # FILTRY – POZYCJE I ZAWODNICY
    # ======================================================
    df_exp = _explode_positions(df_m)
    pos_all = sorted(df_exp["Position"].dropna().unique().tolist())

    pos_sel = st.multiselect(
        "Pozycje (opcjonalnie)",
        pos_all,
        default=[],
        key="cmp_pos_multi"
    )
    if pos_sel:
        df_exp = df_exp[df_exp["Position"].isin(pos_sel)]

    players_all = sorted(df_exp["Name"].dropna().unique())
    p_sel = st.multiselect(
        "Zawodnicy (opcjonalnie)",
        players_all,
        default=players_all[:8],
        key="cmp_players_multi"
    )
    if p_sel:
        df_exp = df_exp[df_exp["Name"].isin(p_sel)]

    # ======================================================
    # REFERENCJA – C1
    # ======================================================
    scope = st.radio(
        "Zakres referencji C1",
        ["Globalna średnia C1", "Średnie C1 per pozycja"],
        horizontal=True,
        key="cmp_scope",
    )

    df_c1 = df_m[df_m["Team"] == "C1"].copy()
    if df_c1.empty:
        st.info("Brak danych referencyjnych z C1.")
        st.stop()

    if scope == "Globalna średnia C1":
        ref = {
            "HSR_m": df_c1["HSR_m"].mean(),
            "Sprint_m": df_c1["Sprint_m"].mean(),
            "ACC": df_c1["ACC"].mean(),
            "DECEL": df_c1["DECEL"].mean(),
            "PlayerIntensityIndex": df_c1["PlayerIntensityIndex"].mean(),
        }
        by_pos = False
    else:
        df_c1_exp = _explode_positions(df_c1)
        ref = (
            df_c1_exp.groupby("Position")[["HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]]
            .mean()
            .rename(columns={
                "HSR_m": "HSR_m_C1",
                "Sprint_m": "Sprint_m_C1",
                "ACC": "ACC_C1",
                "DECEL": "DECEL_C1",
                "PlayerIntensityIndex": "PII_C1"
            })
            .reset_index()
        )
        by_pos = True

    # ======================================================
    # OBLICZENIA
    # ======================================================
    df_base = df_exp.copy()
    metrics = ["HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]

    def add_diffs(df, ref, by_pos):
        df = df.copy()
        if by_pos:
            df = df.merge(ref, on="Position", how="left")
            for m in metrics:
                ref_col = f"{m}_C1" if f"{m}_C1" in df.columns else m
                df[m + "_diff"] = df[m] - df[ref_col]
                df[m + "_pct"] = df[m] / df[ref_col]
        else:
            for m in metrics:
                df[m + "_diff"] = df[m] - ref[m]
                df[m + "_pct"] = df[m] / ref[m]
        return df

    df_cmp = add_diffs(df_base, ref, by_pos)

    # ======================================================
    # WYŚWIETLENIE TABELI
    # ======================================================
    show = df_cmp[
        ["Name", "Team", "Position", "DateStart", "DateEnd"]
        + metrics
        + [m + "_diff" for m in metrics]
        + [m + "_pct" for m in metrics]
    ].sort_values(["Position", "Name", "DateStart"], ascending=[True, True, False])

    show = show.rename(columns={
        "Name": "Zawodnik",
        "Team": "Zespół",
        "Position": "Pozycja",
        "DateStart": "Data od",
        "DateEnd": "Data do",
    })

    st.dataframe(show, use_container_width=True)

    st.markdown(
        """
        **Legenda:**
        - metryki surowe: HSR_m, Sprint_m, ACC, DECEL, PII  
        - _diff = różnica do referencji C1  
        - _pct = procent wartości referencyjnej  
        """
    )
# ============================================================
#        STRONA: ANALIZA (pozycje & zespoły)
# ============================================================

elif page == "Analiza (pozycje & zespoły)":

    st.subheader("Analiza statystyk – per pozycja i per zespół")

    # ------------------------------
    # WYBÓR ŹRÓDŁA: FANTASY / MOTORYKA
    # ------------------------------
    src = st.radio(
        "Źródło danych",
        ["FANTASYPASY", "MOTORYKA"],
        horizontal=True,
        key="an_src"
    )

    # ------------------------------
    # WYBÓR ZESPOŁÓW
    # ------------------------------
    teams_pick = st.multiselect(
        "Zespoły (opcjonalnie)",
        sorted(load_teams_table()["Team"].dropna().unique().tolist()),
        default=[],
        key="an_team_multi"
    )

    # ----------------------------------------------------------
    # WYBÓR ZAKRESU — TYLKO TAKIE OKRESY, KTÓRE ISTNIEJĄ W DANYCH
    # ----------------------------------------------------------
    mode_a = st.radio(
        "Wybór zakresu",
        ["Okres/Test z rejestru", "Z istniejących par dat", "Ręcznie"],
        horizontal=True,
        key="an_mode"
    )

    ds_a, de_a = None, None

    # -------- TRYB 1: OKRES Z REJESTRU ---------
    if mode_a == "Okres/Test z rejestru":
        periods = get_periods_for_table(src, teams_pick)
        if periods.empty:
            st.info("Brak okresów przyporządkowanych do tych danych.")
        else:
            periods["Label2"] = periods["DateStart"].astype(str) + " → " + periods["DateEnd"].astype(str)
            pick = st.selectbox("Okres/Test", periods["Label2"], key="an_pick_period")
            row = periods[periods["Label2"] == pick].iloc[0]
            ds_a, de_a = row["DateStart"], row["DateEnd"]
            st.caption(f"Wybrany zakres: {ds_a} → {de_a}")

    # -------- TRYB 2: Z ISTNIEJĄCYCH PAR DAT ---------
    elif mode_a == "Z istniejących par dat":
        table_name = "motoryka_stats" if src == "MOTORYKA" else "fantasypasy_stats"
        pairs = fetch_df(
            f"""
            SELECT DISTINCT DateStart, DateEnd
            FROM {table_name}
            ORDER BY DateStart DESC, DateEnd DESC
            """
        )
        if pairs.empty:
            st.info("Brak par dat.")
        else:
            pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
            pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")
            options = [
                f"{r.DateStart.date()} → {r.DateEnd.date()}"
                for _, r in pairs.iterrows()
            ]
            pick = st.selectbox("Para dat", options, key="an_pick_pair")
            sel = pairs.iloc[options.index(pick)]
            ds_a, de_a = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Wybrany zakres: {ds_a.date()} → {de_a.date()}")

    # -------- TRYB 3: RĘCZNIE --------------
    else:
        c1, c2 = st.columns(2)
        ds_a = c1.date_input("Od (DateStart)", key="an_ds")
        de_a = c2.date_input("Do (DateEnd)", key="an_de")

    # =======================================================================
    # WCZYTANIE DANYCH
    # =======================================================================

    if src == "FANTASYPASY":
        df = load_fantasy_table().copy()
    else:
        df = load_motoryka_table().copy()

    if df.empty:
        st.info("Brak danych.")
        st.stop()

    # FILTR ZESPOŁÓW
    if teams_pick and "Team" in df.columns:
        df = df[df["Team"].isin(teams_pick)]

    # FILTR DAT
    df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
    df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")

    if ds_a:
        df = df[df["DateStart"] >= pd.to_datetime(ds_a)]
    if de_a:
        df = df[df["DateEnd"] <= pd.to_datetime(de_a)]

    if df.empty:
        st.info("Brak danych po zastosowaniu filtrów.")
        st.stop()

    # =======================================================================
    # PRZYGOTOWANIE DANYCH
    # =======================================================================

    df_pos = _explode_positions(df)
    pos_all = sorted(df_pos["Position"].dropna().unique().tolist())

    pos_pick = st.multiselect(
        "Filtr pozycji (hybrydy wliczone)",
        pos_all,
        default=[],
        key="an_pos"
    )
    if pos_pick:
        df_pos = df_pos[df_pos["Position"].isin(pos_pick)]

    # =======================================================================
    # METRYKI DO AGREGACJI
    # =======================================================================

    if src == "FANTASYPASY":
        num_cols = [
            "NumberOfGames", "Minutes",
            "Goal", "Assist", "ChanceAssist",
            "KeyPass", "KeyLoss", "DuelLossInBox",
            "MissBlockShot", "Finalization",
            "KeyIndividualAction", "KeyRecover",
            "DuelWinInBox", "BlockShot",
            "PktOff", "PktDef"
        ]
    else:
        num_cols = [
            "Minutes", "TD_m", "HSR_m",
            "Sprint_m", "ACC", "DECEL",
            "PlayerIntensityIndex"
        ]

    # =======================================================================
    # ANALIZA — POZYCJA
    # =======================================================================

    st.markdown("### Analiza per pozycja")

    agg_pos = flat_agg(df_pos, ["Position"], num_cols)
    agg_pos_disp = agg_pos.rename(columns={"Position": "Pozycja"})
    st.dataframe(agg_pos_disp, use_container_width=True)

    # =======================================================================
    # ANALIZA — ZESPÓŁ
    # =======================================================================

    st.markdown("### Analiza per zespół")

    if "Team" in df.columns:
        agg_team = flat_agg(df, ["Team"], num_cols)
        agg_team_disp = agg_team.rename(columns={"Team": "Zespół"})
        st.dataframe(agg_team_disp, use_container_width=True)

    # =======================================================================
    # ANALIZA — POZYCJA × ZESPÓŁ
    # =======================================================================

    st.markdown("### Analiza: Zespół × Pozycja")

    if "Team" in df_pos.columns:
        agg_team_pos = flat_agg(df_pos, ["Team", "Position"], num_cols)
        agg_team_pos_disp = agg_team_pos.rename(columns={"Team": "Zespół", "Position": "Pozycja"})
        st.dataframe(agg_team_pos_disp, use_container_width=True)

    st.markdown(
        """
        **Opis kolumn:**
        - __count — liczba wpisów  
        - __sum — suma  
        - __mean — średnia  
        - __median — mediana  
        - __std — odchylenie  
        - __min / __max — wartości min/max  
        - __q25 / __q75 — kwartyle  
        """
    )
# ============================================================
#                     STRONA: WYKRESY ZMIAN
# ============================================================

elif page == "Wykresy zmian":

    st.subheader("Wykresy zmian po dacie")
    st.caption("MOTORYKA – metryki zliczeniowe w przeliczeniu na minutę.")

    # ----------------------------------------
    # WCZYTANIE PEŁNYCH DANYCH MOTORYCZNYCH
    # ----------------------------------------
    dfw = load_motoryka_table().copy()

    if dfw.empty:
        st.info("Brak danych motorycznych.")
        st.stop()

    dfw["Minutes"] = pd.to_numeric(dfw["Minutes"], errors="coerce").replace(0, np.nan)
    dfw["DateStart"] = pd.to_datetime(dfw["DateStart"], errors="coerce")
    dfw["DateEnd"] = pd.to_datetime(dfw["DateEnd"], errors="coerce")

    # ------------------------------
    # Metryki bazowe do per_min
    # ------------------------------
    per_minute_base = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

    # =====================================================
    #             WYBÓR ZAWODNIKÓW
    # =====================================================
    mode_players = st.radio(
        "Zakres zawodników",
        ["Cała drużyna", "Pozycja", "Wybrani gracze"],
        horizontal=True,
        key="plot_players_mode"
    )

    teams_list = sorted(dfw["Team"].dropna().unique().tolist())
    positions_all = sorted(extract_positions(dfw["Position"]))

    # ----------------------------------------------
    # 1) CAŁA DRUŻYNA
    # ----------------------------------------------
    if mode_players == "Cała drużyna":
        t = st.selectbox(
            "Drużyna",
            teams_list,
            key="plot_pick_team"
        )
        dff = dfw[dfw["Team"] == t].copy()
        selected_names = sorted(dff["Name"].dropna().unique().tolist())

    # ----------------------------------------------
    # 2) POZYCJA
    # ----------------------------------------------
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
            dff = pd.DataFrame(rows)
        else:
            dff = dfw.iloc[0:0].copy()

        selected_names = sorted(dff["Name"].dropna().unique().tolist())

    # ----------------------------------------------
    # 3) WYBRANI GRACZE
    # ----------------------------------------------
    else:
        all_names = sorted(dfw["Name"].dropna().unique().tolist())
        selected_names = st.multiselect(
            "Zawodnicy",
            all_names,
            default=all_names[:3],
            key="plot_pick_names"
        )
        dff = dfw[dfw["Name"].isin(selected_names)].copy()

    if dff.empty:
        st.info("Brak danych dla wybranych kryteriów.")
        st.stop()

    if not selected_names:
        st.info("Nie wybrano zawodników.")
        st.stop()

    # =================================================================
    #           PER MINUTE – DODAJEMY KOLUMNY
    # =================================================================
    dff = dff.copy()

    for m in per_minute_base:
        dff[m + "_per_min"] = (
            pd.to_numeric(dff[m], errors="coerce") / dff["Minutes"]
        ).replace([np.inf, -np.inf], np.nan)

    # =================================================================
    #           DATA ŚRODKOWA (uśrednienie okresu)
    # =================================================================
    dff["DateMid"] = dff["DateStart"] + (dff["DateEnd"] - dff["DateStart"]) / 2
    dff["DateMid"] = dff["DateMid"].dt.date

    # =================================================================
    #           OZNACZENIE OKRESÓW (etykiety)
    # =================================================================
    periods = load_periods_table().copy()
    periods["DateStart"] = pd.to_datetime(periods["DateStart"], errors="coerce")
    periods["DateEnd"] = pd.to_datetime(periods["DateEnd"], errors="coerce")

    dff["RangeFallback"] = dff["DateStart"].astype(str) + " → " + dff["DateEnd"].astype(str)

    dff = dff.merge(
        periods[["Label", "DateStart", "DateEnd"]],
        on=["DateStart", "DateEnd"],
        how="left"
    )

    dff["RangeLabel"] = dff["Label"].where(dff["Label"].notna(), dff["RangeFallback"])

    # =================================================================
    #           WYBÓR METRYKI
    # =================================================================
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

    # =================================================================
    #           WYKRES – LINIA + PUNKTY
    # =================================================================
    if base.empty:
        st.info("Brak wartości do wykresu.")
        st.stop()

    plot_df = base.copy()

    # etykieta legendy zależnie od trybu
    if mode_players in ["Pozycja", "Wybrani gracze"]:
        plot_df["LegendLabel"] = plot_df["Team"] + " | " + plot_df["Name"]
        color_field = "LegendLabel"
        legend_title = "Zespół | Zawodnik"
    else:
        color_field = "Name"
        legend_title = "Zawodnik"

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("DateMid:T", title="Data (środek okresu)"),
            y=alt.Y("Value:Q", title=metric),
            color=alt.Color(color_field, title=legend_title),
            tooltip=[
                alt.Tooltip("DateMid:T", title="Data środkowa"),
                alt.Tooltip("RangeLabel:N", title="Zakres"),
                alt.Tooltip("Team:N", title="Zespół"),
                alt.Tooltip("Name:N", title="Zawodnik"),
                alt.Tooltip("Value:Q", title=metric, format=".3f")
            ],
        )
        .properties(height=420)
    )

    st.altair_chart(chart, use_container_width=True)

    # =================================================================
    #           TABELA POD WYKRESEM
    # =================================================================
    table_view = base.sort_values(
        ["Team", "Name", "RangeLabel"]
    )

    st.markdown("### Dane źródłowe")
    st.dataframe(table_view, use_container_width=True)
# ============================================================
#            STRONA: FANTASY – PRZEGLĄD GRAFICZNY
# ============================================================

elif page == "Fantasy – przegląd graficzny":

    st.subheader("FANTASYPASY – przegląd graficzny (po drużynie / meczu / obu)")

    # ============================================
    # WYBÓR TRYBU ZAKRESU DAT
    # ============================================
    mode_fx = st.radio(
        "Wybór zakresu",
        [
            "Z istniejących par dat (FANTASYPASY)",
            "Okres/Test z rejestru",
            "Ręcznie"
        ],
        horizontal=True,
        key="fx_mode"
    )

    selected_pairs = None
    ds_fv, de_fv = None, None

    periods = load_periods_table().copy()
    periods["DateStart"] = pd.to_datetime(periods["DateStart"], errors="coerce")
    periods["DateEnd"] = pd.to_datetime(periods["DateEnd"], errors="coerce")

    # ============================================================
    # TRYB 1: Z ISTNIEJĄCYCH PAR DAT (FANTASYPASY)
    # ============================================================
    if mode_fx == "Z istniejących par dat (FANTASYPASY)":
        pairs = fetch_df("""
            SELECT DISTINCT DateStart, DateEnd
            FROM fantasypasy_stats
            ORDER BY DateStart DESC, DateEnd DESC
        """)

        if pairs.empty:
            st.info("Brak zapisanych par dat.")
        else:
            pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
            pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")

            options = [
                f"{r.DateStart.date()} → {r.DateEnd.date()}"
                for _, r in pairs.iterrows()
            ]
            tuple_map = {
                f"{r.DateStart.date()} → {r.DateEnd.date()}":
                (r.DateStart, r.DateEnd) for _, r in pairs.iterrows()
            }

            if "fx_pick_pairs" not in st.session_state:
                st.session_state["fx_pick_pairs"] = []

            c1, c2 = st.columns([1, 3])
            if c1.button("Wybierz wszystkie"):
                st.session_state["fx_pick_pairs"] = options

            picked = c2.multiselect("Pary dat", options, key="fx_pick_pairs")
            selected_pairs = {tuple_map[x] for x in picked} if picked else None

    # ============================================================
    # TRYB 2: OKRES Z REJESTRU
    # ============================================================
    elif mode_fx == "Okres/Test z rejestru":
        if periods.empty:
            st.info("Brak okresów.")
        else:
            labels = [
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]"
                for _, r in periods.iterrows()
            ]
            tuple_map = {
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]":
                (r.DateStart, r.DateEnd) for _, r in periods.iterrows()
            }

            if "fx_pick_periods" not in st.session_state:
                st.session_state["fx_pick_periods"] = []

            c1, c2 = st.columns([1, 3])
            if c1.button("Wybierz wszystkie okresy"):
                st.session_state["fx_pick_periods"] = labels

            picked = c2.multiselect("Okresy/testy", labels, key="fx_pick_periods")
            selected_pairs = {tuple_map[x] for x in picked} if picked else None

    # ============================================================
    # TRYB 3: RĘCZNIE
    # ============================================================
    else:
        c1, c2 = st.columns(2)
        ds_fv = c1.date_input("Od (DateStart)", value=None, key="fx_ds")
        de_fv = c2.date_input("Do (DateEnd)", value=None, key="fx_de")

    # ============================================================
    # WCZYTANIE DANYCH FANTASY
    # ============================================================
    df_fx = load_fantasy_table().copy()
    df_fx["DateStart"] = pd.to_datetime(df_fx["DateStart"], errors="coerce")
    df_fx["DateEnd"] = pd.to_datetime(df_fx["DateEnd"], errors="coerce")

    if selected_pairs:
        pick_df = pd.DataFrame(list(selected_pairs), columns=["DateStart", "DateEnd"])
        df_fx = df_fx.merge(pick_df, on=["DateStart", "DateEnd"], how="inner")
        st.caption(f"Wybrano {len(selected_pairs)} zakresów dat.")
    else:
        if ds_fv:
            df_fx = df_fx[df_fx["DateStart"] >= pd.to_datetime(ds_fv)]
        if de_fv:
            df_fx = df_fx[df_fx["DateEnd"] <= pd.to_datetime(de_fv)]

    if df_fx.empty:
        st.info("Brak danych FANTASYPASY po filtrach.")
        st.stop()

    # ============================================================
    # PRZYGOTOWANIE KOLUMNY DATE-MID
    # ============================================================
    df_fx["DateMid"] = (
        df_fx["DateStart"] + (df_fx["DateEnd"] - df_fx["DateStart"]) / 2
    ).dt.date

    df_fx["DateMidStr"] = df_fx["DateMid"].astype(str)
    df_fx["FacetBoth"] = df_fx["Team"] + " | " + df_fx["DateMidStr"]

    metric = st.selectbox(
        "Metryka do analizy",
        [
            "PktOff", "PktDef", "Goal", "Assist", "ChanceAssist",
            "KeyPass", "KeyLoss", "DuelLossInBox", "MissBlockShot",
            "Finalization", "KeyIndividualAction", "KeyRecover",
            "DuelWinInBox", "BlockShot"
        ],
        key="fx_metric"
    )

    seg = st.radio(
        "Segregacja",
        ["po drużynie", "po meczu", "po obu (siatka)"],
        horizontal=True,
        key="fx_seg2"
    )

    # ============================================================
    # FILTR ZESPOŁÓW
    # ============================================================
    teams = sorted(df_fx["Team"].dropna().unique().tolist())
    team_pick = st.multiselect(
        "Filtr drużyn (opcjonalnie)",
        teams,
        default=[],
        key="fx_teams2"
    )
    if team_pick:
        df_fx = df_fx[df_fx["Team"].isin(team_pick)]

    if df_fx.empty:
        st.info("Brak danych po odfiltrowaniu drużyn.")
        st.stop()

    # ============================================================
    # WYKRES 1 – PktOff vs PktDef (scatter)
    # ============================================================
    st.markdown("### Wykres: PktOff vs PktDef")

    scatter = (
        alt.Chart(df_fx)
        .mark_circle(size=80, opacity=0.75)
        .encode(
            x=alt.X("PktOff:Q", title="PktOff"),
            y=alt.Y("PktDef:Q", title="PktDef"),
            color=alt.Color("Team:N", title="Drużyna"),
            tooltip=["Name", "Team", "DateMid", "PktOff", "PktDef"]
        )
        .properties(height=320)
    )

    if seg == "po drużynie":
        chart_scatter = scatter.facet("Team:N", columns=3)

    elif seg == "po meczu":
        chart_scatter = scatter.facet("DateMidStr:N", columns=3)

    else:  # siatka: drużyna × dzień
        chart_scatter = scatter.facet("FacetBoth:N", columns=3)

    st.altair_chart(
        chart_scatter.resolve_scale(x="independent", y="independent"),
        use_container_width=True
    )

    # ============================================================
    # WYKRES 2 – ROZKŁAD METRYKI (BOXPLOT)
    # ============================================================
    st.markdown(f"### Rozkład metryki: {metric}")

    box = (
        alt.Chart(df_fx)
        .mark_boxplot(size=40)
        .encode(
            y=alt.Y(f"{metric}:Q", title=metric),
            color=alt.Color("Team:N", title="Drużyna"),
            tooltip=["Team", "Name", "DateMid", metric]
        )
        .properties(height=300)
    )

    if seg == "po drużynie":
        chart_box = box.facet("Team:N", columns=3)
    elif seg == "po meczu":
        chart_box = box.facet("DateMidStr:N", columns=3)
    else:
        chart_box = box.facet("FacetBoth:N", columns=3)

    st.altair_chart(
        chart_box.resolve_scale(y="independent"),
        use_container_width=True
    )

    # ============================================================
    # RANKING ZAWODNIKÓW wg METRYKI
    # ============================================================
    st.markdown(f"### Ranking zawodników wg: {metric}")

    rank_df = (
        df_fx.groupby(["Name", "Team"], as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
    )

    st.dataframe(rank_df.head(30), use_container_width=True)
# ============================================================
#                  STRONA: INDEKS – PORÓWNANIA
# ============================================================

elif page == "Indeks – porównania":

    st.subheader("Indeks – porównania i rankingi PII")

    # =======================================================
    #           TRYB WYBORU ZAKRESU MOTORYKI
    # =======================================================
    mode_idx = st.radio(
        "Wybór zakresu",
        ["Okres/Test z rejestru", "Z istniejących par dat (MOTORYKA)"],
        horizontal=True,
        key="idx_range_mode"
    )

    df_periods = load_periods_table().copy()
    df_periods["DateStart"] = pd.to_datetime(df_periods["DateStart"], errors="coerce")
    df_periods["DateEnd"] = pd.to_datetime(df_periods["DateEnd"], errors="coerce")

    ds_i, de_i = None, None

    # ----------------------------------------
    # TRYB 1: OKRES Z REJESTRU
    # ----------------------------------------
    if mode_idx == "Okres/Test z rejestru":
        if df_periods.empty:
            st.info("Brak okresów w rejestrze.")
            st.stop()
        else:
            options = [
                f"{r.Label} [{r.DateStart.date()}→{r.DateEnd.date()}]"
                for _, r in df_periods.iterrows()
            ]
            pick = st.selectbox("Okres/Test", options, key="idx_pick_period")
            sel = df_periods.iloc[options.index(pick)]
            ds_i, de_i = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_i.date()} → {de_i.date()}")

    # ----------------------------------------
    # TRYB 2: Z ISTNIEJĄCYCH PAR DAT
    # ----------------------------------------
    else:
        pairs = fetch_df(
            """
            SELECT DISTINCT DateStart, DateEnd
            FROM motoryka_stats
            ORDER BY DateStart DESC, DateEnd DESC
            """
        )
        if pairs.empty:
            st.info("Brak par dat w danych motorycznych.")
            st.stop()

        pairs["DateStart"] = pd.to_datetime(pairs["DateStart"], errors="coerce")
        pairs["DateEnd"] = pd.to_datetime(pairs["DateEnd"], errors="coerce")

        opts = [
            f"{r.DateStart.date()} → {r.DateEnd.date()}"
            for _, r in pairs.iterrows()
        ]
        pick = st.selectbox("Para dat", opts, key="idx_pick_pair")
        sel = pairs.iloc[opts.index(pick)]
        ds_i, de_i = sel["DateStart"], sel["DateEnd"]
        st.caption(f"Zakres: {ds_i.date()} → {de_i.date()}")

    # ============================================================
    #             WCZYTYWANIE DANYCH W ZAKRESIE
    # ============================================================
    df = load_motoryka_table().copy()

    if df.empty:
        st.info("Brak danych motorycznych.")
        st.stop()

    df["PlayerIntensityIndex"] = pd.to_numeric(df["PlayerIntensityIndex"], errors="coerce")
    df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
    df["DateEnd"] = pd.to_datetime(df["DateEnd"], errors="coerce")

    if ds_i:
        df = df[df["DateStart"] >= ds_i]
    if de_i:
        df = df[df["DateEnd"] <= de_i]

    if df.empty:
        st.info("Brak danych w wybranym zakresie.")
        st.stop()

    df = df.copy()
    df = df.dropna(subset=["PlayerIntensityIndex"])

    # Data środkowa
    df["DateMid"] = (df["DateStart"] + (df["DateEnd"] - df["DateStart"]) / 2).dt.date

    # ============================================================
    #             PII vs średnia zespołu
    # ============================================================
    if "PlayerIntensityIndexComparingToTeamAverage" in df.columns:
        if df["PlayerIntensityIndexComparingToTeamAverage"].notna().any():
            df["PII_vs_team_avg"] = pd.to_numeric(
                df["PlayerIntensityIndexComparingToTeamAverage"],
                errors="coerce"
            )
        else:
            df["PII_vs_team_avg"] = np.nan
    else:
        # własne liczenie
        team_mean = (
            df.groupby(["Team", "DateStart", "DateEnd"])["PlayerIntensityIndex"]
            .mean()
            .rename("team_avg")
        )

        df = df.merge(
            team_mean,
            on=["Team", "DateStart", "DateEnd"],
            how="left"
        )

        df["PII_vs_team_avg"] = df["PlayerIntensityIndex"] - df["team_avg"]

    # ============================================================
    #          TRYB ANALIZY
    # ============================================================
    mode = st.radio(
        "Tryb analizy",
        [
            "Ranking ogólny",
            "Ranking per data",
            "Ranking per zespół",
            "Porównanie graczy"
        ],
        horizontal=True,
        key="idx_mode_main"
    )

    # ============================================================
    #       RANKING OGÓLNY
    # ============================================================
    if mode == "Ranking ogólny":
        top_n = st.slider("Top N", 5, 50, 20, key="idx_top_all")

        view = (
            df[["Name", "Team", "DateMid", "PlayerIntensityIndex", "PII_vs_team_avg"]]
            .sort_values("PlayerIntensityIndex", ascending=False)
            .head(top_n)
        )

        view_disp = view.rename(columns={
            "Name": "Zawodnik",
            "Team": "Zespół",
            "DateMid": "Data",
            "PlayerIntensityIndex": "Indeks PII",
            "PII_vs_team_avg": "Różnica PII vs średnia zespołu",
        })

        st.dataframe(view_disp, use_container_width=True)

    # ============================================================
    #         RANKING PER DATA
    # ============================================================
    elif mode == "Ranking per data":

        dates = sorted(df["DateMid"].unique(), reverse=True)
        pick = st.multiselect("Wybierz daty", dates, default=dates[:3], key="idx_dates")

        if not pick:
            st.info("Wybierz przynajmniej jedną datę.")
            st.stop()

        top_n = st.slider("Top N na datę", 3, 20, 10, key="idx_top_dates")

        combined = []
        for d in pick:
            tmp = (
                df[df["DateMid"] == d]
                .sort_values("PlayerIntensityIndex", ascending=False)
                .head(top_n)
            )
            tmp["_Date"] = d
            combined.append(tmp)

        view = pd.concat(combined, ignore_index=True)

        view_disp = view.rename(columns={
            "Name": "Zawodnik",
            "Team": "Zespół",
            "DateMid": "Data",
            "PlayerIntensityIndex": "Indeks PII",
            "PII_vs_team_avg": "Różnica PII vs średnia zespołu",
        })

        st.dataframe(view_disp, use_container_width=True)

        chart = (
            alt.Chart(view.rename(columns={"PlayerIntensityIndex": "Value"}))
            .mark_bar()
            .encode(
                y=alt.Y("Name:N", sort="-x", title="Zawodnik"),
                x=alt.X("Value:Q", title="Indeks PII"),
                color=alt.Color("Team:N", title="Zespół"),
                column=alt.Column("_Date:N", header=alt.Header(labelAngle=0)),
            )
            .properties(height=25 * min(top_n, len(view["Name"].unique())))
        )
        st.altair_chart(chart, use_container_width=True)

    # ============================================================
    #       RANKING PER ZESPÓŁ
    # ============================================================
    elif mode == "Ranking per zespół":

        teams = sorted(df["Team"].dropna().unique().tolist())
        pick = st.multiselect("Zespoły", teams, default=teams[:3], key="idx_teams_pick")

        if not pick:
            st.info("Wybierz co najmniej jeden zespół.")
            st.stop()

        top_n = st.slider("Top N na zespół", 3, 20, 10, key="idx_top_team")

        combined = []
        for t in pick:
            tmp = (
                df[df["Team"] == t]
                .sort_values("PlayerIntensityIndex", ascending=False)
                .head(top_n)
            )
            combined.append(tmp)

        view = pd.concat(combined, ignore_index=True)

        view_disp = view.rename(columns={
            "Name": "Zawodnik",
            "Team": "Zespół",
            "DateMid": "Data",
            "PlayerIntensityIndex": "Indeks PII",
            "PII_vs_team_avg": "Różnica PII vs średnia zespołu",
        })

        st.dataframe(view_disp, use_container_width=True)

        chart = (
            alt.Chart(view.rename(columns={"PlayerIntensityIndex": "Value"}))
            .mark_bar()
            .encode(
                y=alt.Y("Name:N", sort="-x"),
                x=alt.X("Value:Q", title="Indeks PII"),
                color=alt.Color("Team:N"),
                column=alt.Column("Team:N", header=alt.Header(labelAngle=0)),
            )
            .properties(height=25 * min(top_n, len(view["Name"].unique())))
        )
        st.altair_chart(chart, use_container_width=True)

    # ============================================================
    #         PORÓWNANIE GRACZY
    # ============================================================
    else:
        teams = sorted(df["Team"].dropna().unique())
        t_sel = st.selectbox(
            "Zespół (opcjonalnie)",
            ["(wszystkie)"] + teams,
            key="idx_cmp_team_main"
        )

        base = df if t_sel == "(wszystkie)" else df[df["Team"] == t_sel]

        all_names = sorted(base["Name"].dropna().unique().tolist())
        names = st.multiselect(
            "Zawodnicy",
            all_names,
            default=all_names[:3],
            key="idx_cmp_names"
        )

        if not names:
            st.info("Wybierz zawodników.")
            st.stop()

        subset = base[base["Name"].isin(names)].copy()

        metric = st.radio(
            "Metryka",
            ["PlayerIntensityIndex", "PII_vs_team_avg"],
            horizontal=True,
            key="idx_cmp_metric"
        )

        plot_data = subset[["Name", "Team", "DateMid", metric]].copy()
        plot_data = plot_data.rename(columns={metric: "Value"}).dropna(subset=["Value"])

        line = (
            alt.Chart(plot_data)
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
            "PII_vs_team_avg": "Różnica PII vs średnia zespołu",
        })

        st.dataframe(table_disp, use_container_width=True)
# ============================================================
#                 STRONA: PROFIL ZAWODNIKA
# ============================================================

elif page == "Profil zawodnika":
    st.subheader("Profil zawodnika – pełny przegląd")

    # 1. Spróbuj wczytać listę zawodników z players_table.csv
    players_df = load_players_table().copy()
    players_list: list[str] = []

    if not players_df.empty and "Name" in players_df.columns:
        # standardowy wariant – mamy normalny plik players_table.csv
        players_df = players_df.dropna(subset=["Name"])
        players_list = sorted(players_df["Name"].astype(str).unique().tolist())
    else:
        # 2. Fallback: budujemy listę zawodników z motoryka_stats i fantasypasy_stats
        moto_raw = load_motoryka_table().copy()
        fant_raw = load_fantasy_table().copy()

        names = set()

        if not moto_raw.empty and "Name" in moto_raw.columns:
            names.update(moto_raw["Name"].dropna().astype(str).unique().tolist())

        if not fant_raw.empty and "Name" in fant_raw.columns:
            names.update(fant_raw["Name"].dropna().astype(str).unique().tolist())

        players_list = sorted(names)

        # Zbuduj przybliżony players_df (Name, Team, Position) na podstawie istniejących danych
        guessed_rows = []

        if not moto_raw.empty:
            cols_m = [c for c in ["Name", "Team", "Position"] if c in moto_raw.columns]
            if cols_m:
                guessed_rows.append(moto_raw[cols_m].copy())

        if not fant_raw.empty:
            cols_f = [c for c in ["Name", "Team", "Position"] if c in fant_raw.columns]
            if cols_f:
                guessed_rows.append(fant_raw[cols_f].copy())

        if guessed_rows:
            merged = pd.concat(guessed_rows, ignore_index=True)
            merged = merged.dropna(subset=["Name"])
            # dla każdego zawodnika bierzemy najczęściej występujący Team/Position
            players_df = (
                merged.groupby("Name", as_index=False)
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            )
        else:
            # totalny fallback – żadnych danych
            players_df = pd.DataFrame(columns=["Name", "Team", "Position"])

    # jeśli dalej nie ma zawodników – kończymy sekcję
    if not players_list:
        st.info("Brak zawodników w danych (motoryka / fantasypasy / players_table.csv).")
        st.stop()

    # wybór zawodnika z listy
    p = st.selectbox(
        "Zawodnik",
        players_list,
        key="prof_all_player",
    )

    if not p:
        st.info("Wybierz zawodnika.")
        st.stop()

    # próbujemy wyciągnąć Team / Position dla podpisu pod nagłówkiem
    prow = None
    if not players_df.empty and "Name" in players_df.columns:
        mask = players_df["Name"].astype(str) == str(p)
        if mask.any():
            prow = players_df[mask].iloc[0]

    team_label = str(prow["Team"]) if (prow is not None and "Team" in prow) else "—"
    pos_str = str(prow["Position"]) if (prow is not None and "Position" in prow) else "—"

    st.caption(f"**Zespół:** {team_label}   •   **Domyślne pozycje:** {pos_str}")

    # ===================== Wczytanie danych MOTORYKA / FANTASYPASY dla zawodnika =====================

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

    # ---------------- MOTORYKA – przygotowanie ----------------
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

    # ---------------- FANTASYPASY – przygotowanie ----------------
    if not fant.empty:
        fant = fant.copy()
        fant["Minutes"] = pd.to_numeric(fant.get("Minutes"), errors="coerce")
        mid_f = pd.to_datetime(fant["DateStart"]) + (
            pd.to_datetime(fant["DateEnd"]) - pd.to_datetime(fant["DateStart"])
        ) / 2
        fant["DateMid"] = mid_f.dt.date

    # krótkie podsumowanie liczby meczów/minut
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
                label="Pobierz profil zawodnika (EXCEL)",
                data=excel_buffer,
                file_name=f"profil_{p}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="prof_all_excel_download",
            )
# ============================================================
#                 SEKCJA: OVER / UNDER
# ============================================================
elif sekcja == "Over/Under":
    st.title("Over/Under – analiza PII i metryk względem zespołu (mediany)")

    # ===== 1) Pobranie danych motorycznych Z CSV (bez SQL) =====
    try:
        # zamiast load_motoryka_all(None, None, None) – tutaj bezpośrednio tabela z CSV
        df = load_motoryka_table()
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

    # usuwamy wiersze bez zawodnika / teamu
    df = df.dropna(subset=["Name", "Team"])

    # ==================================================================
    #                        ZAKŁADKI
    # ==================================================================
    tab_global_pii, tab_metrics, tab_periods = st.tabs([
        "Globalne Over/Under (PII – mediany)",
        "Metryki per minuta (mediany)",
        "Okresy pomiarowe PII (mediany)",
    ])

    # ==================================================================
    #    TAB 1 – GLOBALNE PII: ZAWODNIK vs ZESPOŁY (MEDIANY)
    # ==================================================================
    with tab_global_pii:
        st.subheader("Over/Under – PII zawodnika vs zespoły (mediany)")

        pii_df = df.dropna(subset=["PlayerIntensityIndex"]).copy()
        if pii_df.empty:
            st.info("Brak danych PII.")
            st.stop()

        # wybór zawodnika
        players_list = sorted(pii_df["Name"].unique().tolist())
        player = st.selectbox(
            "Zawodnik",
            players_list,
            key="ou_global_pii_player",
        )

        # dane TYLKO dla wybranego zawodnika (z kompletnymi datami)
        player_pii_all = (
            pii_df[pii_df["Name"] == player]
            .dropna(subset=["DateStart", "DateEnd"])
            .copy()
        )

        if player_pii_all.empty:
            st.info("Brak danych PII dla wybranego zawodnika (brak okresów pomiarowych).")
            st.stop()

        # --------------------------------------------------------------
        # WIDOK 1: JEDEN OKRES – zawodnik vs wszystkie zespoły
        # --------------------------------------------------------------
        st.markdown("### Widok 1 – wybrany okres: zawodnik vs wszystkie zespoły (mediany)")

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

        # filtr zawodnika po wybranym okresie
        player_df = player_pii_all[
            (player_pii_all["DateStart"] == p_start) &
            (player_pii_all["DateEnd"] == p_end)
        ].copy()

        if player_df.empty:
            st.caption(f"Wybrany okres: **{p_start} → {p_end}**")
            st.info("Brak danych PII zawodnika w wybranym okresie.")
            st.stop()

        # główny team zawodnika (najczęściej występujący w tym okresie)
        player_team = (
            player_df["Team"].dropna().mode().iloc[0]
            if not player_df["Team"].dropna().empty
            else None
        )

        # MEDIANA PII zawodnika w wybranym okresie
        player_median_pii = player_df["PlayerIntensityIndex"].median()

        # MEDIANA PII zespołów w wybranym okresie (wszyscy zawodnicy)
        team_pii = (
            pii_df[
                (pii_df["DateStart"] == p_start) &
                (pii_df["DateEnd"] == p_end)
            ]
            .groupby("Team")["PlayerIntensityIndex"]
            .median()
            .reset_index()
            .rename(columns={"PlayerIntensityIndex": "PII_team_median"})
        )

        # tabela: jeden zawodnik vs wszystkie zespoły
        comp = team_pii.copy()
        comp["PII_player_median"] = player_median_pii
        comp["diff_abs"] = comp["PII_player_median"] - comp["PII_team_median"]
        comp["diff_pct"] = (comp["PII_player_median"] / comp["PII_team_median"] - 1.0) * 100.0

        # zespół „najbliżej” zawodnika (po |diff_pct|)
        comp["diff_pct_abs"] = comp["diff_pct"].abs()
        closest_team_name = None
        closest_diff_pct = None
        if not comp.empty:
            idx_closest = comp["diff_pct_abs"].idxmin()
            closest_team_name = comp.loc[idx_closest, "Team"]
            closest_diff_pct = comp.loc[idx_closest, "diff_pct"]

        # zaznaczenie zespołu zawodnika w tabeli
        comp["Is_player_team"] = comp["Team"].apply(
            lambda t: "TAK" if (player_team is not None and t == player_team) else ""
        )

        # próg i status
        threshold = st.slider(
            "Próg (%) – Widok 1",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="ou_global_pii_threshold",
        )

        comp["Status"] = comp["diff_pct"].apply(
            lambda x: "Over" if x >= threshold else ("Under" if x <= -threshold else "Neutral")
        )

        # pożądana kolejność zespołów
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
                f"**Najbardziej zbliżony do mediany PII zawodnika w tym pomiarze jest zespół:** "
                f"**{closest_team_name}** "
                f"(różnica: `{closest_diff_pct:.1f}` punktu procentowego)."
            )

        st.dataframe(
            comp[
                ["Team", "Is_player_team", "PII_player_median", "PII_team_median", "diff_abs", "diff_pct", "Status"]
            ]
            .rename(columns={
                "Team": "Zespół",
                "Is_player_team": "Zespół zawodnika?",
                "PII_player_median": "Mediana PII zawodnika",
                "PII_team_median": "Mediana PII zespołu",
                "diff_abs": "Różnica bezwzględna",
                "diff_pct": "Różnica (%)",
                "Status": "Status",
            })
            .round({
                "Mediana PII zawodnika": 3,
                "Mediana PII zespołu": 3,
                "Różnica bezwzględna": 3,
                "Różnica (%)": 1,
            }),
            use_container_width=True,
        )

        st.markdown(
            """
**Interpretacja różnic procentowych (Różnica (%)) – MEDIANY:**

- Wartość pokazuje, o ile procent **mediana PII zawodnika** różni się od **mediany PII danego zespołu**.
- Obliczamy to jako: **(mediana PII zawodnika / mediana PII zespołu − 1) × 100%**.
- **Wartości dodatnie** (np. +40%) → zawodnik ma medianę PII **wyższą** o 40% niż medianowa wartość zespołu.
- **Wartości ujemne** (np. −30%) → zawodnik ma medianę PII **niższą** o 30% niż medianowa wartość zespołu.
- **Status (Over / Under / Neutral)** opisuje położenie względem zadanego progu.
"""
        )

        # --------------------------------------------------------------
        # WIDOK 2: HISTORIA – zawodnik vs WYBRANY ZESPÓŁ (wszystkie daty)
        # --------------------------------------------------------------
        st.markdown("### Widok 2 – wszystkie okresy: zawodnik vs wybrany zespół (mediany)")

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
            # MEDIANA PII zawodnika per okres
            player_per_period = (
                player_rows
                .groupby(["DateStart", "DateEnd"])["PlayerIntensityIndex"]
                .median()
                .reset_index()
                .rename(columns={"PlayerIntensityIndex": "PII_player_median"})
            )

            # MEDIANA PII zespołu per okres
            team_per_period = (
                team_rows
                .groupby(["DateStart", "DateEnd"])["PlayerIntensityIndex"]
                .median()
                .reset_index()
                .rename(columns={"PlayerIntensityIndex": "PII_team_median"})
            )

            hist = (
                periods_hist
                .merge(player_per_period, on=["DateStart", "DateEnd"], how="left")
                .merge(team_per_period, on=["DateStart", "DateEnd"], how="left")
            )

            hist = hist.dropna(subset=["PII_player_median", "PII_team_median"]).copy()

            if hist.empty:
                st.info("Brak kompletnych danych PII (mediany) dla wspólnych okresów.")
            else:
                hist["diff_abs"] = hist["PII_player_median"] - hist["PII_team_median"]
                hist["diff_pct"] = (hist["PII_player_median"] / hist["PII_team_median"] - 1.0) * 100.0
                hist["Label"] = (
                    hist["DateStart"].astype(str) + " → " + hist["DateEnd"].astype(str)
                )

                threshold_hist = st.slider(
                    "Próg (%) – Widok 2 (historia, mediany)",
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
                            "PII_player_median",
                            "PII_team_median",
                            "diff_abs",
                            "diff_pct",
                            "Status",
                        ]
                    ]
                    .rename(columns={
                        "Label": "Okres",
                        "PII_player_median": "Mediana PII zawodnika",
                        "PII_team_median": "Mediana PII zespołu",
                        "diff_abs": "Różnica bezwzględna",
                        "diff_pct": "Różnica (%)",
                        "Status": "Status",
                    })
                    .round({
                        "Mediana PII zawodnika": 3,
                        "Mediana PII zespołu": 3,
                        "Różnica bezwzględna": 3,
                        "Różnica (%)": 1,
                    }),
                    use_container_width=True,
                )

        # --------------------------------------------------------------
        # WIDOK 3: TABELA – przypisany zespół dla KAŻDEGO pomiaru
        # --------------------------------------------------------------
        st.markdown("### Widok 3 – przypisany zespół dla każdego okresu (mediana PII)")

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

            p_median = p_df["PlayerIntensityIndex"].median()

            t_df = (
                pii_df[
                    (pii_df["DateStart"] == ds) &
                    (pii_df["DateEnd"] == de)
                ]
                .groupby("Team")["PlayerIntensityIndex"]
                .median()
                .reset_index()
                .rename(columns={"PlayerIntensityIndex": "PII_team_median"})
            )

            if t_df.empty:
                continue

            t_df["PII_player_median"] = p_median
            t_df["diff_pct"] = (t_df["PII_player_median"] / t_df["PII_team_median"] - 1.0) * 100.0
            t_df["diff_pct_abs"] = t_df["diff_pct"].abs()

            idx_best = t_df["diff_pct_abs"].idxmin()
            best = t_df.loc[idx_best]

            summary_rows.append({
                "Okres": f"{ds} → {de}",
                "Zespół najbliżej zawodnika": best["Team"],
                "Mediana PII zawodnika": p_median,
                "Mediana PII zespołu": best["PII_team_median"],
                "Różnica (%)": best["diff_pct"],
            })

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows).sort_values("Okres")

            st.dataframe(
                summary_df[
                    [
                        "Okres",
                        "Zespół najbliżej zawodnika",
                        "Mediana PII zawodnika",
                        "Mediana PII zespołu",
                        "Różnica (%)",
                    ]
                ].round({
                    "Mediana PII zawodnika": 3,
                    "Mediana PII zespołu": 3,
                    "Różnica (%)": 1,
                }),
                use_container_width=True,
            )
        else:
            st.info("Brak danych do zbudowania tabeli przypisanych zespołów (mediany PII).")

    # ==================================================================
    #                 TAB 2 – METRYKI PER MINUTA (MEDIANY)
    # ==================================================================
    with tab_metrics:
        st.subheader("Over/Under – metryki per minuta względem zespołu (mediany)")

        per_min_cols = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

        df2 = df.copy()
        for c in per_min_cols:
            df2[c + "_per_min"] = (
                pd.to_numeric(df2.get(c), errors="coerce") / df2["Minutes"]
            ).replace([np.inf, -np.inf], np.nan)

        metrics = [m + "_per_min" for m in per_min_cols]
        metrics_pick = st.multiselect(
            "Metryki (na minutę)",
            metrics,
            default=metrics,
            key="ou_metrics_pick",
        )

        if not metrics_pick:
            st.info("Wybierz przynajmniej jedną metrykę.")
            st.stop()

        # MEDIANY metryk per minuta – zawodnik
        pm = (
            df2.groupby(["Team", "Name"])[metrics_pick]
            .median()
            .reset_index()
        )

        # MEDIANY metryk per minuta – team
        tm = (
            df2.groupby("Team")[metrics_pick]
            .median()
            .reset_index()
            .rename(columns={m: m + "_team_median" for m in metrics_pick})
        )

        merged_m = pm.merge(tm, on="Team", how="left")

        rows = []
        for _, row in merged_m.iterrows():
            for m in metrics_pick:
                t = row[m + "_team_median"]
                p = row[m]
                if pd.isna(t) or pd.isna(p) or t == 0:
                    continue
                rows.append({
                    "Team": row["Team"],
                    "Player": row["Name"],
                    "Metric": m.replace("_per_min", ""),
                    "Player_median": p,
                    "Team_median": t,
                    "diff_abs": p - t,
                    "diff_pct": (p / t - 1) * 100,
                })

        out = pd.DataFrame(rows)

        if out.empty:
            st.info("Brak danych do analizy metryk per minuta (mediany).")
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
                "Player_median": "Mediana zawodnika (na min)",
                "Team_median": "Mediana zespołu (na min)",
                "diff_abs": "Różnica bezwzględna",
                "diff_pct": "Różnica (%)",
                "Status": "Status",
            })
            .round({
                "Mediana zawodnika (na min)": 3,
                "Mediana zespołu (na min)": 3,
                "Różnica bezwzględna": 3,
                "Różnica (%)": 1,
            }),
            use_container_width=True,
        )

    # ==================================================================
    #       TAB 3 – OKRESY POMIAROWE (PII – MEDIANY)
    # ==================================================================
    with tab_periods:
        st.subheader("PII – Over/Under w wybranych okresach pomiarowych (mediany)")

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

        # filtr na dany okres
        df_period = df_pii[
            (df_pii["DateStart"] == d_start) &
            (df_pii["DateEnd"] == d_end)
        ].copy()

        if df_period.empty:
            st.info("Brak danych PII dla wybranego okresu.")
            st.stop()

        # MEDIANA PII per zawodnik w tym okresie
        player_pii_p = (
            df_period
            .groupby(["Team", "Name"])["PlayerIntensityIndex"]
            .median()
            .reset_index()
            .rename(columns={"PlayerIntensityIndex": "PII_player_median"})
        )

        # MEDIANA PII per team w tym okresie
        team_pii_p = (
            df_period
            .groupby("Team")["PlayerIntensityIndex"]
            .median()
            .reset_index()
            .rename(columns={"PlayerIntensityIndex": "PII_team_median"})
        )

        merged_p = player_pii_p.merge(team_pii_p, on="Team", how="left")

        merged_p["diff_abs"] = merged_p["PII_player_median"] - merged_p["PII_team_median"]
        merged_p["diff_pct"] = (
            merged_p["PII_player_median"] / merged_p["PII_team_median"] - 1.0
        ) * 100.0

        threshold_p = st.slider(
            "Próg Over/Under dla PII (mediany) w tym okresie (%)",
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

        st.markdown("### Tabela – zawodnicy Over/Under w wybranym okresie (mediana PII vs zespół)")
        st.dataframe(
            view_p[
                ["Team", "Name", "PII_player_median", "PII_team_median", "diff_abs", "diff_pct", "Status"]
            ]
            .rename(columns={
                "Team": "Zespół",
                "Name": "Zawodnik",
                "PII_player_median": "Mediana PII zawodnika",
                "PII_team_median": "Mediana PII zespołu",
                "diff_abs": "Różnica bezwzględna",
                "diff_pct": "Różnica (%)",
                "Status": "Status",
            })
            .round({
                "Mediana PII zawodnika": 3,
                "Mediana PII zespołu": 3,
                "Różnica bezwzględna": 3,
                "Różnica (%)": 1,
            }),
            use_container_width=True,
        )




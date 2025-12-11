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

    Arkusze:
      1) Dane_surowe – MOTORYKA (u góry) i FANTASYPASY (pod spodem), osobne bloki
         - motoryka: usunięte kolumny 1stTeam / PII_vs_team
         - tylko wiersze, w których zawodnik grał (Minutes > 0 lub NumberOfGames > 0)
      2) OverUnder – cztery widoki Over/Under dla zawodnika:
         - Widok 1 – PII: zawodnik vs wszystkie zespoły (wszystkie okresy)
         - Widok 2 – PII: zawodnik vs domyślny zespół (historia)
         - Widok 3 – metryki per minuta: zawodnik vs mediana zespołu
         - Widok 4 – przypisany zespół dla każdego okresu (mediana PII)
      3) Podsumowanie – liczba pomiarów + mecze + minuty
         - MOTORYKA: liczba pomiarów + minuty (meczów brak)
         - FANTASYPASY: liczba pomiarów, mecze (NumberOfGames), minuty

    Zwraca bytes buffer gotowy do download_button.
    """
    import io
    import numpy as np
    import pandas as pd
    from pandas import ExcelWriter

    # kolumny do usunięcia z motoryki
    MOTO_DROP_COLS = [
        "TD_km_1stTeam",
        "HSR_1stTeam",
        "Sprint_1stTeam",
        "ACC_1stTeam",
        "DECEL_1stTeam",
        "PlayerIntensityIndexComparingToTeamAverage",
    ]

    def safe_int_from_series(s: pd.Series) -> int:
        """Suma serii jako int, NaN → 0."""
        if s is None:
            return 0
        val = pd.to_numeric(s, errors="coerce").sum()
        try:
            return int(val) if pd.notna(val) else 0
        except Exception:
            return 0

    def filter_played(df: pd.DataFrame) -> pd.DataFrame:
        """
        Zostawia tylko wiersze, w których zawodnik grał:
        Minutes > 0 LUB NumberOfGames > 0 (warunek OR).
        Działa również, gdy którejś kolumny nie ma.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if "Minutes" in df.columns:
            mins = pd.to_numeric(df["Minutes"], errors="coerce")
        else:
            mins = pd.Series(0, index=df.index, dtype="float")

        if "NumberOfGames" in df.columns:
            games = pd.to_numeric(df["NumberOfGames"], errors="coerce")
        else:
            games = pd.Series(0, index=df.index, dtype="float")

        mins = mins.fillna(0)
        games = games.fillna(0)

        mask = (mins > 0) | (games > 0)
        return df[mask]

    output = io.BytesIO()

    with ExcelWriter(output, engine="xlsxwriter") as writer:

        # =======================================================
        #  ARKUSZ 1: DANE_SUROWE – MOTORYKA NAD FANTASY
        # =======================================================
        sheet_raw = "Dane_surowe"
        ws_raw = None
        next_row = 0

        moto_used = pd.DataFrame()
        fant_used = pd.DataFrame()

        # ---------- MOTORYKA ----------
        if moto is not None and not moto.empty:
            moto_used = filter_played(moto)

            if not moto_used.empty:
                # usuwamy kolumny 1stTeam / PII_vs_team
                existing_to_drop = [c for c in MOTO_DROP_COLS if c in moto_used.columns]
                if existing_to_drop:
                    moto_used = moto_used.drop(columns=existing_to_drop)

                # sort wg DateMid jeśli jest
                if "DateMid" in moto_used.columns:
                    moto_used["DateMid"] = pd.to_datetime(moto_used["DateMid"], errors="coerce")
                    moto_used = moto_used.sort_values("DateMid", ascending=True)

                # nagłówek + tabela MOTORYKI
                moto_used.to_excel(
                    writer,
                    sheet_name=sheet_raw,
                    index=False,
                    startrow=next_row + 1,
                )
                ws_raw = writer.sheets[sheet_raw]
                ws_raw.write(next_row, 0, "MOTORYKA")
                next_row = next_row + len(moto_used) + 3  # odstęp przed FANTASY

        # ---------- FANTASYPASY ----------
        if fant is not None and not fant.empty:
            fant_used = filter_played(fant)

            if not fant_used.empty:
                if "DateMid" in fant_used.columns:
                    fant_used["DateMid"] = pd.to_datetime(fant_used["DateMid"], errors="coerce")
                    fant_used = fant_used.sort_values("DateMid", ascending=True)

                # jeśli nie było motoryki, trzeba stworzyć arkusz
                if ws_raw is None:
                    fant_used.to_excel(
                        writer,
                        sheet_name=sheet_raw,
                        index=False,
                        startrow=next_row + 1,
                    )
                    ws_raw = writer.sheets[sheet_raw]
                    ws_raw.write(next_row, 0, "FANTASYPASY")
                    next_row = next_row + len(fant_used) + 3
                else:
                    fant_used.to_excel(
                        writer,
                        sheet_name=sheet_raw,
                        index=False,
                        startrow=next_row + 1,
                    )
                    ws_raw.write(next_row, 0, "FANTASYPASY")
                    next_row = next_row + len(fant_used) + 3

        # jeśli oba puste – stworzymy chociaż pusty arkusz
        if ws_raw is None:
            pd.DataFrame().to_excel(writer, sheet_name=sheet_raw, index=False)

        # =======================================================
        #  ARKUSZ 2: OVER / UNDER – 4 WIDOKI DLA ZAWODNIKA
        # =======================================================
        try:
            df_all = load_motoryka_table().copy()
        except Exception:
            df_all = pd.DataFrame()

        view1 = pd.DataFrame()
        view2 = pd.DataFrame()
        view3 = pd.DataFrame()
        view4 = pd.DataFrame()  # przypisany zespół na każdy okres

        if df_all is not None and not df_all.empty:
            df = df_all.copy()
            df["PlayerIntensityIndex"] = pd.to_numeric(
                df.get("PlayerIntensityIndex"), errors="coerce"
            )
            df["Minutes"] = pd.to_numeric(df.get("Minutes"), errors="coerce").replace(0, np.nan)
            df["DateStart"] = pd.to_datetime(df.get("DateStart"), errors="coerce").dt.date
            df["DateEnd"] = pd.to_datetime(df.get("DateEnd"), errors="coerce").dt.date
            df = df.dropna(subset=["Name", "Team", "DateStart", "DateEnd"])

            # dane tylko dla tego zawodnika
            df_player = df[df["Name"] == player_name].copy()

            if not df_player.empty:
                # ===================================================
                # Widok 1: PII – zawodnik vs wszystkie zespoły (wszystkie okresy, mediany)
                # ===================================================
                player_period = (
                    df_player.groupby(["DateStart", "DateEnd"])["PlayerIntensityIndex"]
                    .median()
                    .reset_index()
                    .rename(columns={"PlayerIntensityIndex": "PII_player"})
                )

                team_period = (
                    df.groupby(["Team", "DateStart", "DateEnd"])["PlayerIntensityIndex"]
                    .median()
                    .reset_index()
                    .rename(columns={"PlayerIntensityIndex": "PII_team"})
                )

                view1 = player_period.merge(
                    team_period,
                    on=["DateStart", "DateEnd"],
                    how="inner",
                )

                if not view1.empty:
                    view1["diff_abs"] = view1["PII_player"] - view1["PII_team"]
                    view1["diff_pct"] = (
                        view1["PII_player"] / view1["PII_team"] - 1.0
                    ) * 100.0
                    view1["Okres"] = (
                        view1["DateStart"].astype(str)
                        + " → "
                        + view1["DateEnd"].astype(str)
                    )
                    view1 = view1[
                        ["Okres", "Team", "PII_player", "PII_team", "diff_abs", "diff_pct"]
                    ].rename(columns={
                        "Team": "Zespół",
                        "PII_player": "PII zawodnika (mediana)",
                        "PII_team": "Mediana PII zespołu",
                        "Różnica bezwzględna": "Różnica bezwzględna",   # nazwa już ok
                        "Różnica (%)": "Różnica (%)",                 # nazwa już ok
                    })
                    # powyżej zamieniliśmy nazwy ręcznie, więc poprawmy:
                    view1 = view1.rename(columns={
                        "diff_abs": "Różnica bezwzględna",
                        "diff_pct": "Różnica (%)",
                    })
                    view1 = view1.sort_values(["Okres", "Zespół"])

                # ===================================================
                # Widok 2: PII – historia vs domyślny zespół (mediany)
                # ===================================================
                player_team = (
                    df_player["Team"].dropna().mode().iloc[0]
                    if not df_player["Team"].dropna().empty
                    else None
                )

                if player_team and not view1.empty:
                    view2 = view1[view1["Zespół"] == player_team].copy()
                    view2 = view2.sort_values("Okres")

                # ===================================================
                # Widok 3: metryki per minuta – zawodnik vs mediana zespołu (mediany)
                # ===================================================
                per_min_cols = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

                df2 = df.copy()
                for c in per_min_cols:
                    df2[c + "_per_min"] = (
                        pd.to_numeric(df2.get(c), errors="coerce") / df2["Minutes"]
                    ).replace([np.inf, -np.inf], np.nan)

                metrics = [m + "_per_min" for m in per_min_cols]

                pm = (
                    df2.groupby(["Team", "Name"])[metrics]
                    .median()
                    .reset_index()
                )

                tm = (
                    df2.groupby("Team")[metrics]
                    .median()
                    .reset_index()
                    .rename(columns={m: m + "_team" for m in metrics})
                )

                merged_m = pm.merge(tm, on="Team", how="left")

                rows_m = []
                for _, r in merged_m[merged_m["Name"] == player_name].iterrows():
                    for m in metrics:
                        team_val = r.get(m + "_team")
                        player_val = r.get(m)
                        if pd.isna(team_val) or pd.isna(player_val) or team_val == 0:
                            continue
                        base_metric = m.replace("_per_min", "")
                        rows_m.append({
                            "Zespół": r["Team"],
                            "Metryka": base_metric,
                            "Mediana zawodnika (na min)": player_val,
                            "Mediana zespołu (na min)": team_val,
                            "Różnica bezwzględna": player_val - team_val,
                            "Różnica (%)": (player_val / team_val - 1.0) * 100.0,
                        })

                if rows_m:
                    view3 = pd.DataFrame(rows_m).sort_values(["Zespół", "Metryka"])

                # ===================================================
                # Widok 4: przypisany zespół dla każdego okresu (mediana PII)
                # ===================================================
                summary_rows = []

                periods_all = (
                    df_player[["DateStart", "DateEnd"]]
                    .drop_duplicates()
                    .sort_values(["DateStart", "DateEnd"])
                )

                for _, row_p in periods_all.iterrows():
                    ds = row_p["DateStart"]
                    de = row_p["DateEnd"]

                    # dane zawodnika w tym okresie
                    p_df = df_player[
                        (df_player["DateStart"] == ds) &
                        (df_player["DateEnd"] == de)
                    ]
                    if p_df.empty:
                        continue

                    p_median = p_df["PlayerIntensityIndex"].median()

                    # dane wszystkich zespołów w tym okresie
                    t_df = (
                        df[
                            (df["DateStart"] == ds) &
                            (df["DateEnd"] == de)
                        ]
                        .groupby("Team")["PlayerIntensityIndex"]
                        .median()
                        .reset_index()
                        .rename(columns={"PlayerIntensityIndex": "PII_team_median"})
                    )

                    if t_df.empty:
                        continue

                    t_df["PII_player_median"] = p_median
                    t_df["diff_pct"] = (
                        t_df["PII_player_median"] / t_df["PII_team_median"] - 1.0
                    ) * 100.0
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
                    view4 = (
                        pd.DataFrame(summary_rows)
                        .sort_values("Okres")
                        .round({
                            "Mediana PII zawodnika": 3,
                            "Mediana PII zespołu": 3,
                            "Różnica (%)": 1,
                        })
                    )

        # zapis arkusza OverUnder
        if (view1 is not None and not view1.empty) or \
           (view2 is not None and not view2.empty) or \
           (view3 is not None and not view3.empty) or \
           (view4 is not None and not view4.empty):

            start_row = 0
            sheet_ou = "OverUnder"
            ws_ou = None

            if view1 is not None and not view1.empty:
                view1.to_excel(
                    writer,
                    sheet_name=sheet_ou,
                    index=False,
                    startrow=start_row + 1,
                )
                ws_ou = writer.sheets[sheet_ou]
                ws_ou.write(start_row, 0, "Widok 1 – PII: zawodnik vs wszystkie zespoły (wszystkie okresy, mediany)")
                start_row += len(view1) + 3

            if view2 is not None and not view2.empty:
                if ws_ou is None:
                    view2.to_excel(
                        writer,
                        sheet_name=sheet_ou,
                        index=False,
                        startrow=start_row + 1,
                    )
                    ws_ou = writer.sheets[sheet_ou]
                else:
                    view2.to_excel(
                        writer,
                        sheet_name=sheet_ou,
                        index=False,
                        startrow=start_row + 1,
                    )
                ws_ou.write(start_row, 0, "Widok 2 – PII: zawodnik vs domyślny zespół (historia, mediany)")
                start_row += len(view2) + 3

            if view3 is not None and not view3.empty:
                if ws_ou is None:
                    view3.to_excel(
                        writer,
                        sheet_name=sheet_ou,
                        index=False,
                        startrow=start_row + 1,
                    )
                    ws_ou = writer.sheets[sheet_ou]
                else:
                    view3.to_excel(
                        writer,
                        sheet_name=sheet_ou,
                        index=False,
                        startrow=start_row + 1,
                    )
                ws_ou.write(start_row, 0, "Widok 3 – metryki per minuta: zawodnik vs mediana zespołu (mediany)")
                start_row += len(view3) + 3

            if view4 is not None and not view4.empty:
                if ws_ou is None:
                    view4.to_excel(
                        writer,
                        sheet_name=sheet_ou,
                        index=False,
                        startrow=start_row + 1,
                    )
                    ws_ou = writer.sheets[sheet_ou]
                else:
                    view4.to_excel(
                        writer,
                        sheet_name=sheet_ou,
                        index=False,
                        startrow=start_row + 1,
                    )
                ws_ou.write(start_row, 0, "Widok 4 – przypisany zespół dla każdego okresu (mediana PII)")

        # =======================================================
        #  ARKUSZ 3: PODSUMOWANIE POMIARÓW / MECZÓW
        # =======================================================
        summary_rows = []

        if moto_used is not None and not moto_used.empty:
            summary_rows.append({
                "Kategoria": "Motoryka",
                "Liczba pomiarów": len(moto_used),
                "Mecze": np.nan,  # brak meczów w motoryce
                "Minuty": safe_int_from_series(moto_used.get("Minutes")),
            })

        if fant_used is not None and not fant_used.empty:
            summary_rows.append({
                "Kategoria": "FANTASYPASY",
                "Liczba pomiarów": len(fant_used),
                "Mecze": safe_int_from_series(fant_used.get("NumberOfGames")),
                "Minuty": safe_int_from_series(fant_used.get("Minutes")),
            })

        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_excel(writer, index=False, sheet_name="Podsumowanie")

    output.seek(0)
    return output









# ============================================================
#                 FUNKCJA fetch_df
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
#         POBIERANIE OKRESÓW TYLKO Z DANYMI
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
        [
            "Motoryka",
            "Fantasy",
            "Porównania z C1",
            "Benchmark zawodnika",
            "Klasteryzacja",
            "Profil zawodnika",
        ],
        key="nav_section",
    )

    # ------------------ MOTORYKA ------------------
    if sekcja == "Motoryka":
        page = st.radio(
            "Widok",
            [
                "Wykresy zmian",
                "Indeks – porównania",
                "Player Intensity Index",
            ],
            key="nav_page_mot",
        )

    # ------------------ FANTASY -------------------
    elif sekcja == "Fantasy":
        page = "Fantasy – przegląd graficzny"

    # ------------- PORÓWNANIA Z C1 ----------------
    elif sekcja == "Porównania z C1":
        page = "Porównania"

    # ------------ ZNORMALIZOWANY PII --------------
    elif sekcja == "Benchmark zawodnika":
        page = "Benchmark zawodnika"

    # ---------------- KLASTERYZACJA ---------------
    elif sekcja == "Klasteryzacja":
        page = st.radio(
            "Widok",
            [
                "Over/Under",
                "Powtarzalne Over/Under",
            ],
            key="nav_page_cluster",
        )

    # --------------- PROFIL ZAWODNIKA -------------
    else:  # "Profil zawodnika"
        page = "Profil zawodnika"


st.title("Cracovia – analiza")


# ============================================================
#                    STRONA: PORÓWNANIA
# ============================================================

if page == "Porównania":
    st.subheader("Porównanie młodzieży do pierwszego zespołu C1")

    # ======================================================
    # WCZYTANIE OKRESÓW Z REJESTRU
    # ======================================================
    periods_all = load_periods_table().copy()
    periods_all["DateStart"] = pd.to_datetime(periods_all["DateStart"], errors="coerce")
    periods_all["DateEnd"]   = pd.to_datetime(periods_all["DateEnd"],   errors="coerce")

    # ======================================================
    # WCZYTANIE CAŁEJ TABELI motoryka_stats (JEDEN RAZ)
    # ======================================================
    df_all = fetch_df("""
        SELECT Name, Team, Position, DateStart, DateEnd,
               Minutes, HSR_m, Sprint_m, ACC, DECEL, PlayerIntensityIndex
        FROM motoryka_stats
    """)

    df_all["DateStart"] = pd.to_datetime(df_all["DateStart"], errors="coerce")
    df_all["DateEnd"]   = pd.to_datetime(df_all["DateEnd"],   errors="coerce")

    # ======================================================
    # WYDZIELAMY C1 I WYLICZAMY DOSTĘPNE DATY C1
    # ======================================================
    c1_all = df_all[df_all["Team"] == "C1"].copy()
    if c1_all.empty:
        st.info("Brak jakichkolwiek pomiarów C1 w bazie.")
        st.stop()

    c1_dates = (
        c1_all[["DateStart", "DateEnd"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ======================================================
    # FUNKCJA: okres jest OK, jeśli C1 ma pomiar z DateStart w środku
    # ======================================================
    def period_is_valid(ds, de):
        cond = (c1_dates["DateStart"] >= ds) & (c1_dates["DateStart"] <= de)
        return cond.any()

    # ======================================================
    # FILTRUJEMY OKRESY – ZOSTAJĄ TYLKO Z DANYMI C1
    # ======================================================
    periods_all = periods_all[
        periods_all.apply(lambda r: period_is_valid(r["DateStart"], r["DateEnd"]), axis=1)
    ]

    if periods_all.empty:
        st.info("Brak okresów pokrywających się z realnymi pomiarami C1.")
        st.stop()

    labels = [
        f"{r.Label} [{r.DateStart.date()} → {r.DateEnd.date()}]"
        for _, r in periods_all.iterrows()
    ]

    pick = st.selectbox("Wybierz okres testowy", labels, index=0, key="cmp_pick_period")
    sel = periods_all.iloc[labels.index(pick)]

    ds_f, de_f = sel["DateStart"], sel["DateEnd"]
    st.caption(f"Zakres analizy: **{ds_f.date()} → {de_f.date()}**")

    # ======================================================
    # 1) DANE C1 – DateStart musi wypaść w środku okresu
    # ======================================================
    df_c1 = c1_all[
        (c1_all["DateStart"] >= ds_f) &
        (c1_all["DateStart"] <= de_f)
    ].copy()

    if df_c1.empty:
        st.info("Brak danych referencyjnych C1 w wybranym okresie (sprawdź dane w bazie).")
        st.stop()

    # ======================================================
    # 2) DANE MŁODZIEŻY – overlap z okresem
    # ======================================================
    df_youth = df_all[
        (df_all["Team"] != "C1") &
        (df_all["DateEnd"]   >= ds_f) &
        (df_all["DateStart"] <= de_f)
    ].copy()

    if df_youth.empty:
        st.info("Brak danych zawodników w wybranym okresie.")
        st.stop()

    df_m = df_youth.copy()

    # ======================================================
    # FILTRY POZYCJI I ZAWODNIKÓW
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
        default=players_all[:10],
        key="cmp_players_multi"
    )
    if p_sel:
        df_exp = df_exp[df_exp["Name"].isin(p_sel)]

    # ======================================================
    # REFERENCJA – GLOBALNA / PER POZYCJA
    # ======================================================
    scope = st.radio(
        "Zakres referencji C1",
        ["Globalna średnia C1", "Średnie C1 per pozycja"],
        horizontal=True,
        key="cmp_scope",
    )

    metrics = ["HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]

    if scope == "Globalna średnia C1":
        ref = {m: df_c1[m].mean() for m in metrics}
        by_pos = False
    else:
        df_c1_exp = _explode_positions(df_c1)
        ref = (
            df_c1_exp.groupby("Position")[metrics]
            .mean()
            .rename(columns={
                "HSR_m": "HSR_m_C1",
                "Sprint_m": "Sprint_m_C1",
                "ACC": "ACC_C1",
                "DECEL": "DECEL_C1",
                "PlayerIntensityIndex": "PII_C1",
            })
            .reset_index()
        )
        by_pos = True

    # ======================================================
    # OBLICZENIE RÓŻNIC DO C1
    # ======================================================
    import numpy as np

    def add_diffs(df, ref, by_pos):
        df = df.copy()

        if by_pos:
            df = df.merge(ref, on="Position", how="left")
            for m in metrics:
                ref_col = f"{m}_C1" if f"{m}_C1" in df.columns else m
                df[f"{m}_diff"] = df[m] - df[ref_col]
                df[f"{m}_pct"] = np.where(
                    df[ref_col].astype(float) == 0,
                    np.nan,
                    df[m] / df[ref_col],
                )
        else:
            for m in metrics:
                base = ref[m]
                df[f"{m}_diff"] = df[m] - base
                df[f"{m}_pct"] = np.where(
                    base == 0,
                    np.nan,
                    df[m] / base,
                )

        return df

    df_cmp = add_diffs(df_exp, ref, by_pos)

    # ======================================================
    # WYŚWIETLENIE TABELI
    # ======================================================
    show = df_cmp[
        ["Name", "Team", "Position", "DateStart", "DateEnd"]
        + metrics
        + [m + "_diff" for m in metrics]
        + [m + "_pct"  for m in metrics]
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
        - _diff = różnica względem referencji C1  
        - _pct = procent wartości referencyjnej (1.00 = 100%)  
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

        # daty jako date (żeby ładnie merge'ować po DateStart/DateEnd)
        moto["DateStart"] = pd.to_datetime(moto["DateStart"], errors="coerce").dt.date
        moto["DateEnd"] = pd.to_datetime(moto["DateEnd"], errors="coerce").dt.date

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
            periods = periods.copy()
            periods["DateStart"] = pd.to_datetime(periods["DateStart"], errors="coerce").dt.date
            periods["DateEnd"] = pd.to_datetime(periods["DateEnd"], errors="coerce").dt.date

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

        # globalne średnie PII zespołów po okresach – z całego CSV
        all_m = load_motoryka_table().copy()
        if not all_m.empty and "PlayerIntensityIndex" in all_m.columns:
            all_m = all_m.copy()
            all_m["PlayerIntensityIndex"] = pd.to_numeric(
                all_m["PlayerIntensityIndex"], errors="coerce"
            )
            all_m["DateStart"] = pd.to_datetime(all_m["DateStart"], errors="coerce").dt.date
            all_m["DateEnd"] = pd.to_datetime(all_m["DateEnd"], errors="coerce").dt.date

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
        fant["DateStart"] = pd.to_datetime(fant["DateStart"], errors="coerce")
        fant["DateEnd"] = pd.to_datetime(fant["DateEnd"], errors="coerce")
        mid_f = fant["DateStart"] + (fant["DateEnd"] - fant["DateStart"]) / 2
        fant["DateMid"] = mid_f.dt.date

    # krótkie podsumowanie liczby meczów/minut
    if not moto.empty:
        mecze_m_raw = pd.to_numeric(moto.get("NumberOfGames"), errors="coerce").sum()
        min_m_raw = pd.to_numeric(moto.get("Minutes"), errors="coerce").sum()
        mecze_m = int(mecze_m_raw) if pd.notna(mecze_m_raw) else 0
        min_m = int(min_m_raw) if pd.notna(min_m_raw) else 0
        st.caption(f"MOTORYKA: mecze = {mecze_m}, minuty = {min_m}")

    if not fant.empty:
        mecze_f_raw = pd.to_numeric(fant.get("NumberOfGames"), errors="coerce").sum()
        min_f_raw = pd.to_numeric(fant.get("Minutes"), errors="coerce").sum()
        mecze_f = int(mecze_f_raw) if pd.notna(mecze_f_raw) else 0
        min_f = int(min_f_raw) if pd.notna(min_f_raw) else 0
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

            # --- PODSUMOWANIE METRYK NA ZAKRES POMIARU (GLOBALNE) ---
            st.markdown("### Podsumowanie metryk na zakres pomiaru")

            all_m = load_motoryka_table().copy()
            if all_m.empty:
                st.info("Brak globalnych danych motorycznych.")
            else:
                all_m["Minutes"] = pd.to_numeric(all_m["Minutes"], errors="coerce").replace(0, np.nan)

                for m in per_min_cols:
                    if m in all_m.columns:
                        all_m[m + "_per_min"] = (
                            pd.to_numeric(all_m[m], errors="coerce") / all_m["Minutes"]
                        )

                all_m["DateStart"] = pd.to_datetime(all_m["DateStart"], errors="coerce").dt.date
                all_m["DateEnd"] = pd.to_datetime(all_m["DateEnd"], errors="coerce").dt.date

                periods2 = get_periods_df().copy()
                if not periods2.empty:
                    periods2["DateStart"] = pd.to_datetime(
                        periods2["DateStart"], errors="coerce"
                    ).dt.date
                    periods2["DateEnd"] = pd.to_datetime(
                        periods2["DateEnd"], errors="coerce"
                    ).dt.date

                    all_m = all_m.merge(
                        periods2[["DateStart", "DateEnd", "Label"]],
                        on=["DateStart", "DateEnd"],
                        how="left",
                    )

                    all_m["RangeLabel"] = all_m["Label"].fillna(
                        all_m["DateStart"].astype(str) + " → " + all_m["DateEnd"].astype(str)
                    )
                else:
                    all_m["RangeLabel"] = (
                        all_m["DateStart"].astype(str) + " → " + all_m["DateEnd"].astype(str)
                    )

                agg_cols = [c for c in all_m.columns if c.endswith("_per_min")]

                if agg_cols:
                    summary_global = (
                        all_m.groupby("RangeLabel")[agg_cols]
                        .agg(["mean", "median"])
                        .round(2)
                    )
                    st.dataframe(summary_global, use_container_width=True)

            st.markdown("---")
            st.subheader("Zawodnik vs C1 – per minuta (globalnie)")

            # referencja C1 liczona z całej tabeli motorycznej
            df_all_for_ref = load_motoryka_table().copy()

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
                if agg_pii.empty:
                    st.info("Brak danych do wyznaczenia najlepszych/najsłabszych okresów.")
                else:
                    c1, c2 = st.columns(2)
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
elif page == "Over/Under":
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


# ============================================================
#           SEKCJA: POWTARZALNE OVER / UNDER
# ============================================================
elif page == "Powtarzalne Over/Under":
    st.title("Powtarzalne Over/Under – analiza stabilności zawodników")

    # ===== 1) Pobranie danych motorycznych =====
    try:
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
    df = df.dropna(subset=["Name", "Team", "PlayerIntensityIndex", "DateStart", "DateEnd"])

    # ==================================================================
    # MEDIANY PII w okresach: zawodnik vs zespół
    # ==================================================================
    player_period = (
        df.groupby(["Name", "Team", "DateStart", "DateEnd"], as_index=False)["PlayerIntensityIndex"]
        .median()
        .rename(columns={"PlayerIntensityIndex": "PII_player"})
    )

    team_period = (
        df.groupby(["Team", "DateStart", "DateEnd"], as_index=False)["PlayerIntensityIndex"]
        .median()
        .rename(columns={"PlayerIntensityIndex": "PII_team"})
    )

    merged = player_period.merge(
        team_period,
        on=["Team", "DateStart", "DateEnd"],
        how="left",
    ).dropna(subset=["PII_player", "PII_team"])

    if merged.empty:
        st.info("Brak danych PII per okres.")
        st.stop()

    merged["diff_abs"] = merged["PII_player"] - merged["PII_team"]
    merged["diff_pct"] = (merged["PII_player"] / merged["PII_team"] - 1.0) * 100
    merged["Okres"] = merged["DateStart"].astype(str) + " → " + merged["DateEnd"].astype(str)

    # ==================================================================
    # WYBÓR WIDOKU
    # ==================================================================
    view_mode = st.radio(
        "Widok",
        ["Powtarzalni w okresie", "Ranking powtarzalności"],
        horizontal=True,
        key="rep_view_mode",
    )

    # ==================================================================
    #     WIDOK 1 – POWTARZALNI W OKRESIE
    # ==================================================================
    if view_mode == "Powtarzalni w okresie":

        # dostępne okresy
        periods_all = (
            merged[["DateStart", "DateEnd"]]
            .drop_duplicates()
            .sort_values(["DateStart", "DateEnd"])
        )
        periods_all["Label"] = periods_all["DateStart"].astype(str) + " → " + periods_all["DateEnd"].astype(str)

        okres_sel = st.selectbox(
            "Okres referencyjny",
            periods_all["Label"].tolist(),
            key="rep_period_select",
        )

        row_sel = periods_all[periods_all["Label"] == okres_sel].iloc[0]
        ds = row_sel["DateStart"]
        de = row_sel["DateEnd"]

        st.caption(f"Wybrany okres: {ds} → {de}")

        threshold = st.slider(
            "Próg (%) Over/Under (vs mediana zespołu)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="rep_threshold",
        )

        sel_df = merged[
            (merged["DateStart"] == ds) &
            (merged["DateEnd"] == de)
        ].copy()

        if sel_df.empty:
            st.info("Brak danych w wybranym okresie.")
            st.stop()

        # wyliczenia
        over_df = sel_df[sel_df["diff_pct"] >= threshold].copy()
        under_df = sel_df[sel_df["diff_pct"] <= -threshold].copy()

        c1, c2 = st.columns(2)

        # ------------------ OVER ------------------
        with c1:
            st.markdown("## Powtarzalni OVER")
            st.caption(
                "Zawodnicy, których mediana PII w tym okresie jest wyższa niż mediana PII zespołu "
                "o co najmniej zadany próg."
            )

            if over_df.empty:
                st.info("Brak zawodników Over w tym okresie.")
            else:
                st.dataframe(
                    over_df[["Team", "Name"]]
                    .sort_values(["Team", "Name"])
                    .rename(columns={"Team": "Zespół", "Name": "Zawodnik"}),
                    use_container_width=True,
                )

        # ------------------ UNDER ------------------
        with c2:
            st.markdown("## Powtarzalni UNDER")
            st.caption(
                "Zawodnicy, których mediana PII w tym okresie jest niższa niż mediana PII zespołu "
                "o co najmniej zadany próg."
            )

            if under_df.empty:
                st.info("Brak zawodników Under w tym okresie.")
            else:
                st.dataframe(
                    under_df[["Team", "Name"]]
                    .sort_values(["Team", "Name"])
                    .rename(columns={"Team": "Zespół", "Name": "Zawodnik"}),
                    use_container_width=True,
                )

    # ==================================================================
    #     WIDOK 2 – RANKING POWTARZALNOŚCI (CAŁA HISTORIA)
    # ==================================================================
    else:
        st.subheader("Ranking powtarzalności Over/Under – cała historia (vs mediana zespołu)")

        threshold_rank = st.slider(
            "Próg (%) Over/Under (vs mediana zespołu)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            key="rep_rank_threshold",
        )

        merged_rank = merged.copy()
        merged_rank["flag_over"] = merged_rank["diff_pct"] >= threshold_rank
        merged_rank["flag_under"] = merged_rank["diff_pct"] <= -threshold_rank

        # agregacja po zawodnikach
        agg_rank = (
            merged_rank
            .groupby(["Team", "Name"], as_index=False)
            .agg(
                Over=("flag_over", "sum"),
                Under=("flag_under", "sum"),
                Periods=("Okres", "nunique"),
            )
        )

        # zawodnicy, którzy mieli choć 1 Over albo 1 Under
        agg_rank = agg_rank[(agg_rank["Over"] > 0) | (agg_rank["Under"] > 0)]

        if agg_rank.empty:
            st.info("Brak zawodników z historii Over/Under.")
            st.stop()

        min_rep = st.slider(
            "Minimalna liczba powtórzeń Over/Under",
            min_value=1,
            max_value=int(agg_rank[["Over", "Under"]].values.max()),
            value=2,
            step=1,
            key="rep_min_rep",
        )

        agg_rank_f = agg_rank[
            (agg_rank["Over"] >= min_rep) | (agg_rank["Under"] >= min_rep)
        ]

        if agg_rank_f.empty:
            st.info("Brak zawodników spełniających minimalne kryterium powtórzeń.")
            st.stop()

        c1, c2 = st.columns(2)

        # ------------------ OVER ranking ------------------
        with c1:
            st.markdown("### Najczęściej OVER")
            df_over = agg_rank_f.sort_values(["Over", "Under"], ascending=[False, False])
            st.dataframe(
                df_over.rename(columns={
                    "Team": "Zespół",
                    "Name": "Zawodnik",
                    "Over": "Liczba Over",
                    "Under": "Liczba Under",
                    "Periods": "Okresy pomiarowe",
                }),
                use_container_width=True,
            )

        # ------------------ UNDER ranking ------------------
        with c2:
            st.markdown("### Najczęściej UNDER")
            df_under = agg_rank_f.sort_values(["Under", "Over"], ascending=[False, False])
            st.dataframe(
                df_under.rename(columns={
                    "Team": "Zespół",
                    "Name": "Zawodnik",
                    "Under": "Liczba Under",
                    "Over": "Liczba Over",
                    "Periods": "Okresy pomiarowe",
                }),
                use_container_width=True,
            )

        st.markdown(
            """
**Interpretacja rankingu:**

- Każdy Over/Under oznacza okres, w którym zawodnik był istotnie powyżej lub poniżej mediany zespołu.  
- Ranking pokazuje, którzy zawodnicy **regularnie przekraczają medianę zespołu (Over)**,  
  a którzy częściej są **poniżej typowego poziomu drużyny (Under)**.  
"""
        )

# ============================================================
#          STRONA: BENCHMARK ZAWODNIKA – WIDOK
# ============================================================

elif page == "Benchmark zawodnika":

    st.subheader("Benchmark zawodnika")
    st.caption(
        "Porównanie zawodników względem wzorca – zawodnicy z tej samej pozycji lub z tego samego zespołu, "
        "metryki motoryczne w czasie oraz statystyki zbiorcze względem wzorca."
    )

    # ---------------------------------------------------------
    # ŁADOWANIE I PRZYGOTOWANIE DANYCH
    # ---------------------------------------------------------
    df = load_motoryka_table().copy()

    # usuwamy team C1 z analizy
    if "Team" in df.columns:
        df = df[df["Team"] != "C1"]

    if df.empty or "Name" not in df.columns:
        st.info("Brak danych motorycznych do porównań.")
        st.stop()

    df["Minutes"] = pd.to_numeric(df.get("Minutes"), errors="coerce").replace(0, np.nan)
    df["DateStart"] = pd.to_datetime(df.get("DateStart"), errors="coerce")
    df["DateEnd"] = pd.to_datetime(df.get("DateEnd"), errors="coerce")

    df = df.dropna(subset=["Name", "Minutes", "DateStart", "DateEnd"])
    if df.empty:
        st.info("Brak pełnych danych (daty / minuty) do porównania.")
        st.stop()

    # środek okresu jako datetime (nie .date()) – altair to lubi
    df["DateMid"] = df["DateStart"] + (df["DateEnd"] - df["DateStart"]) / 2

    # przeliczenia per minuta
    per_min_base = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]
    for m in per_min_base:
        if m in df.columns:
            df[m + "_per_min"] = (
                pd.to_numeric(df[m], errors="coerce") / df["Minutes"]
            ).replace([np.inf, -np.inf], np.nan)

    per_min_cols = [m + "_per_min" for m in per_min_base if m + "_per_min" in df.columns]
    if not per_min_cols:
        st.info("Brak metryk per minuta.")
        st.stop()

    # ---------------------------------------------------------
    # WYBÓR WZORCA
    # ---------------------------------------------------------
    players_all = sorted(df["Name"].dropna().unique())
    ref_player = st.selectbox("Zawodnik referencyjny (wzorzec)", players_all, key="bm_ref_player")

    base_df = df[df["Name"] == ref_player].copy()
    if base_df.empty:
        st.info("Brak danych dla wzorca.")
        st.stop()

    # zespół i pozycje wzorca
    base_team = base_df["Team"].dropna().mode().iloc[0] if "Team" in base_df.columns else "—"

    positions = set()
    if "Position" in base_df.columns:
        for val in base_df["Position"].dropna():
            for p in str(val).replace("\\", "/").split("/"):
                p = p.strip().upper()
                if p:
                    positions.add(p)

    pos_label = ", ".join(sorted(positions)) if positions else "—"
    st.caption(f"Wzorzec – **{ref_player}**, zespół: **{base_team}**, pozycje: **{pos_label}**")

    # ---------------------------------------------------------
    # TRYB PORÓWNANIA
    # ---------------------------------------------------------
    group_mode = st.radio(
        "Grupa porównawcza",
        ["Zawodnicy na tej samej pozycji", "Zawodnicy z tego samego zespołu"],
        horizontal=True,
        key="bm_group_mode",
    )

    df_np = df[["Name", "Team", "Position"]].drop_duplicates()

    def parse_pos(val):
        if pd.isna(val):
            return set()
        out = set()
        for p in str(val).replace("\\", "/").split("/"):
            p = p.strip().upper()
            if p:
                out.add(p)
        return out

    candidates = set()

    if group_mode == "Zawodnicy na tej samej pozycji":
        for _, r in df_np.iterrows():
            if parse_pos(r["Position"]).intersection(positions):
                candidates.add(r["Name"])
    else:
        for _, r in df_np.iterrows():
            if r["Team"] == base_team:
                candidates.add(r["Name"])

    candidates.discard(ref_player)
    candidates = sorted(candidates)

    if not candidates:
        st.info("Brak zawodników spełniających kryteria (pozycja/zespół).")
        st.stop()

    cmp_players = st.multiselect(
        "Zawodnicy do porównania",
        candidates,
        default=candidates[:5] if len(candidates) > 5 else candidates,
        key="bm_cmp_players",
    )

    if not cmp_players:
        st.info("Wybierz co najmniej jednego zawodnika do porównania.")
        st.stop()

    # ---------------------------------------------------------
    # WYBÓR METRYKI
    # ---------------------------------------------------------
    metric = st.selectbox(
        "Metryka (per min)",
        per_min_cols,
        format_func=lambda x: x.replace("_per_min", "") + " (per min)",
        key="bm_metric",
    )

    # ---------------------------------------------------------
    # WYCIĘCIE DANYCH DO ANALIZY
    # ---------------------------------------------------------
    df_cmp = df[df["Name"].isin([ref_player] + cmp_players)].copy()
    df_cmp = df_cmp.dropna(subset=["DateMid", metric])

    if df_cmp.empty:
        st.info("Brak danych do analizy tej metryki.")
        st.stop()

    # ---------------------------------------------------------
    # DWA TABY
    # ---------------------------------------------------------
    tab_time, tab_stats = st.tabs(["Wykresy i daty", "Statystyki zbiorcze"])


    # =========================================================
    # TAB 1 — wykres + tabelki per data
    # =========================================================
    with tab_time:
        st.markdown("### Wykres w czasie – metryka per minuta")

        plot_df = df_cmp[["Name", "Team", "DateMid", metric]].copy()
        plot_df = plot_df.rename(columns={metric: "Value"})
        plot_df["DateMid"] = pd.to_datetime(plot_df["DateMid"], errors="coerce")

        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("DateMid:T", title="Data"),
                y=alt.Y("Value:Q", title=metric),
                color=alt.Color("Name:N", title="Zawodnik"),
                tooltip=[
                    alt.Tooltip("DateMid:T", title="Data"),
                    alt.Tooltip("Team:N", title="Zespół"),
                    alt.Tooltip("Name:N", title="Zawodnik"),
                    alt.Tooltip("Value:Q", title=metric, format=".1f"),
                ],
            )
            .properties(height=420)
        )

        st.altair_chart(chart, use_container_width=True)

        # --------- tabelki per data vs wzorzec ---------
        st.markdown("### Tabelki po datach – różnice względem zawodnika wzorcowego")

        base_vals = (
            base_df[["DateMid", metric]]
            .dropna(subset=["DateMid", metric])
            .groupby("DateMid", as_index=False)[metric]
            .mean()
            .rename(columns={metric: "BaseValue"})
        )

        cmp_vals = (
            df_cmp[df_cmp["Name"].isin(cmp_players)][["Name", "Team", "DateMid", metric]]
            .dropna(subset=["DateMid", metric])
            .copy()
        )

        cmp_vals = cmp_vals.merge(base_vals, on="DateMid", how="left")
        cmp_vals = cmp_vals.dropna(subset=["BaseValue"])

        if cmp_vals.empty:
            st.info("Brak wspólnych dat pomiarowych między wzorcem a wybranymi zawodnikami.")
        else:
            cmp_vals["diff_abs"] = cmp_vals[metric] - cmp_vals["BaseValue"]
            cmp_vals["diff_pct"] = np.where(
                cmp_vals["BaseValue"].astype(float) == 0,
                np.nan,
                (cmp_vals[metric] / cmp_vals["BaseValue"] - 1) * 100,
            )

            table_view = (
                cmp_vals
                .sort_values(["DateMid", "Team", "Name"])
                .rename(columns={
                    "DateMid": "Data",
                    "Team": "Zespół",
                    "Name": "Zawodnik",
                    metric: "Wartość zawodnika",
                    "BaseValue": f"Wartość wzorca ({ref_player})",
                    "diff_abs": "Różnica bezwzględna",
                    "diff_pct": "Różnica (%)",
                })
            )

            for d in sorted(table_view["Data"].unique()):
                st.markdown(f"#### Data: {d}")
                df_day = table_view[table_view["Data"] == d].drop(columns=["Data"])

                st.dataframe(
                    df_day.round({
                        "Wartość zawodnika": 1,
                        f"Wartość wzorca ({ref_player})": 1,
                        "Różnica bezwzględna": 1,
                        "Różnica (%)": 1,
                    }),
                    use_container_width=True,
                )

            st.markdown(
                """
                **Jak czytać te tabelki:**  
                - Każda tabelka odpowiada jednej dacie (środek okresu pomiarowego).  
                - „Wartość wzorca” to ta sama metryka dla zawodnika referencyjnego.  
                - „Różnica (%)” = (wartość zawodnika / wartość wzorca – 1) × 100%.  
                """
            )

    # =========================================================
    # TAB 2 — statystyki zbiorcze
    # =========================================================
    with tab_stats:
        st.markdown("### Statystyki zbiorcze — średnie, mediany, odchylenia")

        base_metric_vals = base_df[metric].dropna()
        base_mean = base_metric_vals.mean()
        base_median = base_metric_vals.median()
        base_std = base_metric_vals.std()
        base_n = base_metric_vals.count()

        # agregaty dla zawodników porównywanych
        agg_cmp = (
            df_cmp[df_cmp["Name"].isin(cmp_players)][["Name", "Team", metric]]
            .dropna(subset=[metric])
            .groupby(["Name", "Team"], as_index=False)
            .agg(
                Liczba_pomiarów=("Name", "size"),
                Średnia=(metric, "mean"),
                Mediana=(metric, "median"),
                Odchylenie_std=(metric, "std"),
            )
        )

        if agg_cmp.empty:
            st.info("Brak danych do policzenia statystyk zbiorczych.")
        else:
            # dodajemy wartości wzorca i różnice
            agg_cmp["Średnia wzorca"] = base_mean
            agg_cmp["Mediana wzorca"] = base_median

            agg_cmp["Różnica średniej"] = agg_cmp["Średnia"] - base_mean
            agg_cmp["Różnica mediany"] = agg_cmp["Mediana"] - base_median

            agg_cmp["Różnica średniej (%)"] = np.where(
                base_mean == 0, np.nan, (agg_cmp["Średnia"] / base_mean - 1) * 100
            )
            agg_cmp["Różnica mediany (%)"] = np.where(
                base_median == 0, np.nan, (agg_cmp["Mediana"] / base_median - 1) * 100
            )

            # sortowanie po różnicy średniej
            agg_cmp = agg_cmp.sort_values("Różnica średniej", ascending=False)

            # wiersz z wzorcem na górze tabeli
            ref_row = pd.DataFrame({
                "Name": [ref_player],
                "Team": [base_team],
                "Liczba_pomiarów": [base_n],
                "Średnia": [base_mean],
                "Mediana": [base_median],
                "Odchylenie_std": [base_std],
                "Średnia wzorca": [base_mean],
                "Mediana wzorca": [base_median],
                "Różnica średniej": [0.0],
                "Różnica mediany": [0.0],
                "Różnica średniej (%)": [0.0],
                "Różnica mediany (%)": [0.0],
            })

            agg_show = pd.concat([ref_row, agg_cmp], ignore_index=True)
            agg_show = agg_show.rename(columns={"Name": "Zawodnik", "Team": "Zespół"})

            # kolumna Typ: Wzorzec / Zawodnik
            typ_col = ["Wzorzec"] + ["Zawodnik"] * (len(agg_show) - 1)
            agg_show.insert(0, "Typ", typ_col)

            # zaokrąglenia – WSZYSTKO DO 1 MIEJSCA PO PRZECINKU
            numeric_round = {
                "Średnia": 1,
                "Mediana": 1,
                "Odchylenie_std": 1,
                "Średnia wzorca": 1,
                "Mediana wzorca": 1,
                "Różnica średniej": 1,
                "Różnica mediany": 1,
                "Różnica średniej (%)": 1,
                "Różnica mediany (%)": 1,
            }
            agg_show = agg_show.round(numeric_round)

            # wyróżnienie wiersza wzorca
            def highlight_ref(row):
                if row["Typ"] == "Wzorzec":
                    return ["background-color: #303030; font-weight: bold"] * len(row)
                return [""] * len(row)

            styled = agg_show.style.apply(highlight_ref, axis=1)

            st.dataframe(styled, use_container_width=True)

            st.markdown(
                """
                **Interpretacja:**  
                - Wiersz „Wzorzec” to zawodnik referencyjny.  
                - Średnia / mediana – wartości metryki per minuta dla zawodnika.  
                - Różnice pokazują odchylenie od wzorca w jednostkach metryki i procentowo.  
                """
            )



# ============================================================
#          STRONA: PLAYER INTENSITY INDEX (PII)
# ============================================================

elif page == "Player Intensity Index":

    st.subheader("Player Intensity Index – kategorie i normalizacja")
    st.caption(
        "Normalizacja wskaźnika PII do skali 0–100 względem maksymalnej wartości w wybranym okresie "
        "oraz podział na kategorie wysiłkowe (U17 / U19 / C2)."
    )

    # ---------------------------------------------------------
    # WCZYTANIE DANYCH
    # ---------------------------------------------------------
    df = load_motoryka_table().copy()

    # WYKLUCZENIE C1
    if "Team" in df.columns:
        df = df[df["Team"] != "C1"]

    if df.empty or "PlayerIntensityIndex" not in df.columns:
        st.info("Brak danych PII w tabeli motorycznej.")
        st.stop()

    # tylko realne występy – zawodnik musi mieć minuty > 0
    df["Minutes"] = pd.to_numeric(df.get("Minutes"), errors="coerce")
    df = df[df["Minutes"] > 0]

    if df.empty:
        st.info("Brak zawodników z rozegranymi minutami (Minutes > 0).")
        st.stop()

    df["DateStart"] = pd.to_datetime(df["DateStart"], errors="coerce")
    df["DateEnd"]   = pd.to_datetime(df["DateEnd"], errors="coerce")
    df["DateMid"]   = df["DateStart"] + (df["DateEnd"] - df["DateStart"]) / 2

    df = df.dropna(subset=["Name", "Team", "PlayerIntensityIndex", "DateStart", "DateEnd"])
    if df.empty:
        st.info("Brak kompletnych danych do analizy PII.")
        st.stop()

    # ---------------------------------------------------------
    # WYBÓR OKRESU Z REJESTRU (LABEL + DATY)
    # ---------------------------------------------------------
    periods = load_periods_table().copy()
    periods["DateStart"] = pd.to_datetime(periods["DateStart"], errors="coerce")
    periods["DateEnd"]   = pd.to_datetime(periods["DateEnd"], errors="coerce")

    # tylko okresy, które faktycznie występują w danych PII
    used_pairs = (
        df[["DateStart", "DateEnd"]]
        .dropna()
        .drop_duplicates()
    )

    periods = periods.merge(
        used_pairs,
        on=["DateStart", "DateEnd"],
        how="inner",
    )

    if periods.empty:
        st.info("Brak okresów pomiarowych wspólnych dla rejestru i danych PII.")
        st.stop()

    periods = periods.sort_values(["DateStart", "DateEnd"], ascending=[False, False])

    labels = [
        f"{r.Label} [{r.DateStart.date()} → {r.DateEnd.date()}]"
        for _, r in periods.iterrows()
    ]
    label_to_dates = {
        f"{r.Label} [{r.DateStart.date()} → {r.DateEnd.date()}]": (r.DateStart, r.DateEnd)
        for _, r in periods.iterrows()
    }

    picked_label = st.selectbox(
        "Okres pomiarowy",
        labels,
        index=0,
        key="pii_period_pick",
    )

    ds_sel, de_sel = label_to_dates[picked_label]

    # filtr danych na wybrany okres
    df = df[(df["DateStart"] == ds_sel) & (df["DateEnd"] == de_sel)]
    if df.empty:
        st.info("Brak danych PII dla wybranego okresu.")
        st.stop()

    # ---------------------------------------------------------
    # NORMALIZACJA PII DO 0–100 WZGLĘDEM MAKSIMUM W TYM OKRESIE
    # ---------------------------------------------------------
    pii_vals = pd.to_numeric(df["PlayerIntensityIndex"], errors="coerce").fillna(0)
    pii_max = float(pii_vals.max())

    if pii_max <= 0:
        df["PII_norm"] = 0.0
    else:
        df["PII_norm"] = (pii_vals / pii_max) * 100.0

    # ---------------------------------------------------------
    # PROGI KATEGORII WG DRUŻYNY (U17 / U19 / C2) – NA PII_norm
    # ---------------------------------------------------------
    import numpy as np

    thresholds = {
        "U17": [
            (0, 25, "Bardzo niska"),
            (25, 40, "Niska"),
            (40, 55, "Optymalna"),
            (55, 65, "Wysoka"),
            (65, np.inf, "Bardzo wysoka / Maksymalna"),
        ],
        "U19": [
            (0, 35, "Bardzo niska"),
            (35, 50, "Niska"),
            (50, 70, "Optymalna"),
            (70, 80, "Wysoka"),
            (80, np.inf, "Bardzo wysoka / Maksymalna"),
        ],
        "C2": [
            (0, 40, "Bardzo niska"),
            (40, 60, "Niska"),
            (60, 75, "Optymalna"),
            (75, 85, "Wysoka"),
            (85, np.inf, "Bardzo wysoka / Maksymalna"),
        ],
    }

    def infer_band(team: str) -> str:
        t = str(team).upper()
        if "U17" in t or "U-17" in t:
            return "U17"
        if "U19" in t or "U-19" in t:
            return "U19"
        # reszta traktowana jako C2
        if "C2" in t:
            return "C2"
        return "C2"

    def classify(row):
        band = infer_band(row["Team"])
        val = row["PII_norm"]

        if pd.isna(val):
            return pd.Series({
                "Grupa progowa": band,
                "Kategoria wysiłku": "Nieznana"
            })

        for low, high, label in thresholds[band]:
            if low <= val < high:
                return pd.Series({
                    "Grupa progowa": band,
                    "Kategoria wysiłku": label
                })

        return pd.Series({
            "Grupa progowa": band,
            "Kategoria wysiłku": "Nieznana"
        })

    df = df.join(df.apply(classify, axis=1))

    # ---------------------------------------------------------
    # FILTR ZESPOŁÓW: WSZYSTKIE / JEDEN ZESPÓŁ
    # ---------------------------------------------------------
    teams_all = sorted(df["Team"].unique())
    mode_teams = st.radio(
        "Zakres zespołów",
        ["Wszystkie", "Jeden zespół"],
        horizontal=True,
        key="pii_team_mode",
    )

    if mode_teams == "Jeden zespół":
        team_pick = st.selectbox("Wybierz zespół", teams_all, key="pii_team_one")
        df = df[df["Team"] == team_pick]

    if df.empty:
        st.info("Brak danych po filtrowaniu zespołów.")
        st.stop()

    # ---------------------------------------------------------
    # WIDOK – PODZIAŁ NA KATEGORIE OD NAJGORSZYCH DO NAJLEPSZYCH
    # ---------------------------------------------------------
    st.markdown("### Kategorie wysiłku i PII (0–100) – wybrany okres")
    st.caption(f"Okres pomiaru: **{picked_label}**")

    med_raw = df["PlayerIntensityIndex"].median()
    med_norm = df["PII_norm"].median()

    base_show = df[[
        "Name", "Team",
        "PlayerIntensityIndex", "PII_norm",
        "Grupa progowa", "Kategoria wysiłku",
    ]].rename(columns={
        "Name": "Zawodnik",
        "Team": "Zespół",
        "PlayerIntensityIndex": "PII (surowy)",
        "PII_norm": "PII CN (0–100)",
    }).round({
        "PII (surowy)": 1,
        "PII CN (0–100)": 1,
    })

    cat_order = [
        "Bardzo niska",
        "Niska",
        "Optymalna",
        "Wysoka",
        "Bardzo wysoka / Maksymalna",
        "Nieznana",
    ]

    for cat in cat_order:
        sub = base_show[base_show["Kategoria wysiłku"] == cat].copy()
        if sub.empty:
            continue

        st.markdown(f"#### {cat}")

        # ta kolumna jest wspólna w ramach tabelki – można ją wyrzucić
        sub = sub.drop(columns=["Kategoria wysiłku"])

        st.dataframe(
            sub.sort_values(["Zespół", "Zawodnik", "PII CN (0–100)"]),
            use_container_width=True,
        )

    st.caption(
        f"Mediana PII (surowy): **{med_raw:.1f}**, "
        f"Mediana PII CN (0–100): **{med_norm:.1f}**."
    )

    st.markdown(
        """
        **Interpretacja PII CN (0–100):**
        - 0 oznacza PII = 0 (brak zarejestrowanej intensywności w danym pomiarze),
        - 100 odpowiada najwyższej wartości PII w wybranym okresie,
        - wartości pośrednie są udziałem (% maksymalnej intensywności w tym okresie).
        """
    )











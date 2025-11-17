# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import altair as alt
from datetime import date
from io import BytesIO
import sqlite3
import os

# ===================== USTAWIENIA & DB =====================
st.set_page_config(page_title="Cracovia – rejestry, wprowadzanie danych i analizy", layout="wide")

DB_FILE = "cracovia.sqlite" # Nazwa pliku lokalnej bazy SQLite
DEFAULT_TEAMS = ["C1", "C2", "U-19", "U-17"]


@st.cache_resource
def get_sqlite_engine():
    """Tworzy silnik SQLite i ładuje dane z cracovia.sql (jeśli plik DB nie istnieje)."""
    if not os.path.exists(DB_FILE):
        st.info("Inicjalizacja lokalnej bazy danych SQLite z pliku cracovia.sql...")
        try:
            # Tworzymy puste połączenie
            conn = sqlite3.connect(DB_FILE)
            
            # Wczytujemy zrzut SQL
            with open("cracovia.sql", "r", encoding="utf-8") as f:
                sql_script = f.read()

            # --- Czyszczenie skryptu SQL dla kompatybilności z SQLite ---
            sql_clean = []
            skip_lines = False
            for line in sql_script.splitlines():
                # Pomijanie komentarzy i sekcji MariaDB/MySQL
                if line.startswith("/*M!") or line.startswith("/*!") or line.startswith("--"):
                    continue
                
                # Agresywne pomijanie definicji WIDOKÓW
                if "VIEW" in line or "Temporary table structure for view" in line or "Final view structure for view" in line:
                    skip_lines = True
                    continue
                if skip_lines and line.strip().endswith(";"):
                    skip_lines = False
                    continue
                    
                if skip_lines:
                    continue

                # Czyszczenie składni
                line = line.replace("int(11)", "INTEGER")
                line = line.replace("decimal(6,3)", "REAL")
                line = line.replace("decimal(8,3)", "REAL")
                line = line.replace("timestamp", "TEXT")
                line = line.replace("date", "TEXT")
                line = line.replace("varchar(100)", "TEXT")
                line = line.replace("varchar(120)", "TEXT")
                line = line.replace("varchar(50)", "TEXT")
                
                # Usuń klauzule MySQL
                line = line.replace("ENGINE=InnoDB", "")
                line = line.replace("COLLATE=utf8mb4_uca1400_ai_ci", "")
                line = line.replace("DEFAULT CHARSET=utf8mb4", "")
                line = line.replace("AUTO_INCREMENT", "")
                
                # Usuń CHECK, FOREIGN KEY, INDEXY
                line = line.replace("CHECK (`DuelLossOutBox` <= 0),", ",")
                line = line.replace("CHECK (`RescueAction` >= 0),", ",")
                line = line.replace("CONSTRAINT `chk_KeyLoss_le_zero` CHECK (`KeyLoss` <= 0),", "")
                line = line.replace("CONSTRAINT `chk_DuelLossInBox_le_zero` CHECK (`DuelLossInBox` <= 0),", "")
                line = line.replace("CONSTRAINT `chk_MissBlockShot_le_zero` CHECK (`MissBlockShot` <= 0)", "")
                line = line.replace("CONSTRAINT `fk_players_team` FOREIGN KEY (`Team`) REFERENCES `teams` (`Team`) ON DELETE SET NULL ON UPDATE CASCADE", "")
                line = line.replace("KEY `fk_players_team` (`Team`),", ",")
                line = line.replace("KEY `idx_fant_team_dates` (`Team`,`DateStart`,`DateEnd`),", ",")
                line = line.replace("KEY `idx_fant_pos_dates` (`Position`,`DateStart`,`DateEnd`),", ",")
                line = line.replace("KEY `idx_moto_team_dates` (`Team`,`DateStart`,`DateEnd`),", ",")
                line = line.replace("KEY `idx_moto_pos_dates` (`Position`,`DateStart`,`DateEnd`)", "")
                line = line.replace("UNIQUE KEY `uniq_label_dates` (`Label`,`DateStart`,`DateEnd`)", "UNIQUE (`Label`,`DateStart`,`DateEnd`)")
                line = line.replace("UNIQUE KEY `Name` (`Name`),", "UNIQUE (`Name`),")
                
                # Zamień GENERATED ALWAYS AS na zwykłe kolumny INTEGER/REAL
                line = line.replace("GENERATED ALWAYS AS", "AS")
                line = line.replace("STORED", "")
                line = line.replace("`PktOff` AS (`Goal` + `Assist` + `ChanceAssist` + `KeyPass` + `KeyLoss` + `Finalization` + `KeyIndividualAction`)", "`PktOff` INTEGER")
                line = line.replace("`PktDef` AS (`KeyRecover` + `DuelWinInBox` + `DuelLossInBox` + `DuelLossOutBox` + `BlockShot` + `MissBlockShot` + `RescueAction`)", "`PktDef` INTEGER")
                line = line.replace("`PlayerIntensityIndex` REAL AS (`HSR_m` * 1.0 + `Sprint_m` * 1.5 + `ACC` * 2.0 + `DECEL` * 2.0)", "`PlayerIntensityIndex` REAL")
                line = line.replace("`PlayerIntensityIndexComparingToTeamAverage` REAL NOT NULL DEFAULT 0.000", "`PlayerIntensityIndexComparingToTeamAverage` REAL NOT NULL DEFAULT 0.0")

                # Dodaj do czystego skryptu
                if line.strip():
                    sql_clean.append(line.strip())
            
            sql_clean_str = "\n".join(sql_clean)
            
            # Wykonaj skrypt
            conn.executescript(sql_clean_str)
            conn.commit()
            conn.close()
            st.success("Baza danych SQLite zainicjalizowana pomyślnie.")
            
        except Exception as e:
            st.error(f"Błąd inicjalizacji bazy danych SQLite z cracovia.sql: {e}")
            
    # Teraz tworzymy silnik SQLAlchemy dla SQLite
    DB_URL = f"sqlite:///{DB_FILE}"
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
    
    return engine

engine = get_sqlite_engine() # Inicjalizacja silnika


# ===================== AUTORYZACJA =====================
USERNAME = "admin"
PASSWORD = "Cracovia"

def init_auth_state():
    if "auth" not in st.session_state:
        st.session_state["auth"] = False

__SIDEBAR_RENDERED = False
def sidebar_auth():
    """Panel logowania w sidebarze (wywołaj raz na start)."""
    global __SIDEBAR_RENDERED
    if __SIDEBAR_RENDERED:
        return
    __SIDEBAR_RENDERED = True
    with st.sidebar:
        st.markdown("  Dostęp do edycji")
        if st.session_state.get("auth", False):
            st.success("Zalogowano jako **admin**")
            if st.button(" Wyloguj", key="btn_logout_sidebar"):
                st.session_state["auth"] = False
                st.toast("Wylogowano.")
        else:
            with st.expander("Zaloguj się, aby dodawać/edytować"):
                with st.form("login_form_sidebar_main", clear_on_submit=False):
                    u = st.text_input("Login", value="", key="login_user_sidebar_main")
                    p = st.text_input("Hasło", value="", type="password", key="login_pass_sidebar_main")
                    ok = st.form_submit_button("Zaloguj", type="primary")
                if ok:
                    if u == USERNAME and p == PASSWORD:
                        st.session_state["auth"] = True
                        st.success("Zalogowano.")
                        st.toast("Zalogowano.")
                    else:
                        st.error("Błędny login lub hasło.")

def render_login_inline(suffix: str):
    st.info("Ta sekcja wymaga zalogowania. Użyj panelu ** Dostęp do edycji** w lewym sidebarze.")
    with st.expander(f"Albo zaloguj się tutaj ({suffix})"):
        with st.form(f"login_form_inline_{suffix}", clear_on_submit=False):
            u = st.text_input("Login", value="", key=f"login_user_inline_{suffix}")
            p = st.text_input("Hasło", value="", type="password", key=f"login_pass_inline_{suffix}")
            ok = st.form_submit_button("Zaloguj", type="primary")
        if ok:
            if u == USERNAME and p == PASSWORD:
                st.session_state["auth"] = True
                st.success("Zalogowano.")
            else:
                st.error("Błędny login lub hasło.")

init_auth_state()
sidebar_auth()

# ===================== STAŁE =====================
POS_OPTIONS = [
    "ŚO", "LŚO", "PŚO", "LO",
    "ŚPD", "8", "ŚP", "ŚPO", "10", "PW", "LW", "WAHADŁO", "NAPASTNIK",
]

# ===================== DB HELPERS =====================
def exec_sql(sql, params=None):
    with engine.begin() as con:
        return con.execute(text(sql), params or {})

def upsert(sql, params):
    if not st.session_state.get("auth", False):
        raise PermissionError("Brak uprawnień do zapisu – zaloguj się.")
    with engine.begin() as con:
        con.execute(text(sql), params)

def fetch_df(sql, params=None):
    return pd.read_sql(text(sql), con=engine, params=params or {})

def ensure_db_views(engine):
    """Tworzy widok 'all_stats' w SQLite, który zastępuje skomplikowany JOIN z MySQL."""
    with engine.begin() as conn:
        sql_all_stats = """
        CREATE VIEW IF NOT EXISTS all_stats AS 
        SELECT
            f.Name, f.Team, f.Position, f.DateStart, f.DateEnd,
            f.NumberOfGames, f.Minutes, f.Goal, f.Assist, f.ChanceAssist, f.KeyPass, 
            f.KeyLoss, f.Finalization, f.KeyIndividualAction, f.KeyRecover, 
            f.DuelWinInBox, f.BlockShot, f.DuelLossInBox, f.DuelLossOutBox, 
            f.MissBlockShot, f.RescueAction, 
            (f.Goal + f.Assist + f.ChanceAssist + f.KeyPass + f.KeyLoss + f.Finalization + f.KeyIndividualAction) AS PktOff, 
            (f.KeyRecover + f.DuelWinInBox + f.DuelLossInBox + f.DuelLossOutBox + f.BlockShot + f.MissBlockShot + f.RescueAction) AS PktDef,
            m.TD_m, m.HSR_m, m.Sprint_m, m.ACC, m.DECEL, 
            (m.HSR_m * 1.0 + m.Sprint_m * 1.5 + m.ACC * 2.0 + m.DECEL * 2.0) AS PlayerIntensityIndex
        FROM fantasypasy_stats f
        LEFT JOIN motoryka_stats m ON f.Name = m.Name AND f.DateStart = m.DateStart AND f.DateEnd = m.DateEnd;
        """
        try:
            conn.execute(text("DROP VIEW IF EXISTS all_stats;"))
            conn.execute(text(sql_all_stats))
        except Exception as e:
            # To jest w porządku, jeśli widok nie istnieje lub jest używany
            pass
            
# Wymuś utworzenie widoku all_stats (lub odtworzenie)
ensure_db_views(engine) 

sql = "SELECT * FROM all_stats WHERE Team='C1'"
df = fetch_df(sql)

# --- zabezpieczenie przed brakiem kolumn motoryki (np. HSR_m) ---
for c in ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]:
    if c not in df.columns:
        df[c] = 0

# --- Funkcje rejestrów (Poprawione dla SQLite) ---

def get_team_list():
    try:
        df = fetch_df("SELECT Team FROM teams ORDER BY Team;")
        return df["Team"].tolist() if not df.empty else DEFAULT_TEAMS
    except Exception:
        return DEFAULT_TEAMS

def upsert_player(name: str, team: str, position: str):
    # POPRAWKA: Zmiana na składnię SQLite ON CONFLICT
    upsert("""
        INSERT INTO players (Name, Team, Position)
        VALUES (:n, :t, :p)
        ON CONFLICT(Name) DO UPDATE SET Team=excluded.Team, Position=excluded.Position
    """, {"n": name.strip(), "t": team.strip() if team else None, "p": position.strip() if position else None})


# ==========================================
# Funkcja Excel (zawiera błędy w oryginalnym kodzie, ale dla spójności zachowujemy ogólną strukturę)
# ==========================================

def build_player_excel_report(player_name: str, moto: pd.DataFrame, fant: pd.DataFrame) -> BytesIO:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        workbook = writer.book
        
        # Inicjalizacja zmiennych do obsługi kolumn w arkuszu Porównanie
        excel_comparison_worksheet = "Porównanie"
        c1_mean_idx = 1
        num_fant_rows = 0
        comparison_data = pd.DataFrame()

        # ===== 1) PORÓWNANIE – metryki PII i porównanie z C1 (sekcja górna) =====
        if moto is not None and not moto.empty:
            moto = moto.copy()
            # Usuń godziny z dat
            moto["DateStart"] = pd.to_datetime(moto["DateStart"], errors="coerce").dt.date
            moto["DateEnd"] = pd.to_datetime(moto["DateEnd"], errors="coerce").dt.date
            moto["Minutes"] = pd.to_numeric(moto.get("Minutes"), errors="coerce").replace(0, np.nan)
            
            # Musimy załadować wszystkie dane motoryki, żeby policzyć średnią C1 dynamicznie
            try:
                # Wczytanie wszystkich danych motoryki
                df_all_moto = fetch_df("SELECT PlayerIntensityIndex, Team FROM motoryka_stats WHERE Team='C1'")
                c1_mean_idx = pd.to_numeric(df_all_moto["PlayerIntensityIndex"], errors="coerce").mean()
            except Exception:
                c1_mean_idx = 1 # Fallback, aby uniknąć ZeroDivisionError

            # Użyj kolumny PII, która powinna być obliczona przez load_motoryka_all lub widok
            moto["PlayerIntensityIndex"] = pd.to_numeric(moto["PlayerIntensityIndex"], errors="coerce")
            
            # Porównanie jako stosunek do średniej C1
            moto["PII_vs_team_avg"] = moto["PlayerIntensityIndex"] / c1_mean_idx 
            
            # Zapisz dane porównania
            comparison_data = moto[["DateStart", "DateEnd", "Name", "Team", "PlayerIntensityIndex", "PII_vs_team_avg"]]
            comparison_data.to_excel(writer, sheet_name=excel_comparison_worksheet, index=False, startrow=1)

            # Dodaj nagłówek
            worksheet = writer.sheets[excel_comparison_worksheet]
            worksheet.write(0, 0, "PORÓWNANIE – metryki PII i porównanie z C1")

        # ===== 2) FANTASYPASY – surowe dane =====
        if fant is not None and not fant.empty:
            fant = fant.copy()
            # Usuń godziny z dat
            fant["DateStart"] = pd.to_datetime(fant["DateStart"], errors="coerce").dt.date
            fant["DateEnd"] = pd.to_datetime(fant["DateEnd"], errors="coerce").dt.date

            # Zapisz surowe dane fantasy
            fant_full = fant[[
                "DateStart", "DateEnd", "Goal", "Assist", "ChanceAssist", "KeyPass", "KeyLoss",
                "DuelLossInBox", "DuelLossOutBox", "MissBlockShot", "Finalization", "KeyIndividualAction",
                "KeyRecover", "DuelWinInBox", "BlockShot", "PktOff", "PktDef"
            ]]
            fant_full.to_excel(writer, sheet_name="Fantasypasy", index=False, startrow=1)

            # Dodaj nagłówek
            worksheet = writer.sheets["Fantasypasy"]
            worksheet.write(0, 0, "FANTASYPASY – surowe dane")
            
            # Musimy znać długość tej tabeli, żeby kontynuować w Porównanie
            num_fant_rows = len(fant_full) + 1 # + nagłówek 

        # ===== 3) PORÓWNANIE – metryki per minutę i porównanie z C1 (Dodatkowe) =====
        if moto is not None and not moto.empty and c1_mean_idx != 1 and 'Minutes' in moto.columns:
            
            per_min_cols = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]
            
            # Musimy obliczyć metryki per minuta
            for col in per_min_cols:
                if col in moto.columns:
                    moto[col + "_per_min"] = pd.to_numeric(moto[col], errors="coerce") / moto["Minutes"]
                    # Obliczanie średniej C1 dla danej metryki per minuta (dla porównania)
                    c1_mean_per_min = moto[col + "_per_min"].loc[moto["Team"] == "C1"].mean()
                    moto[col + "_vs_c1"] = moto[col + "_per_min"] / (c1_mean_per_min if c1_mean_per_min not in (0, np.nan) else 1) 

            # Zapisz metryki per minutę i porównanie z C1
            all_per_min_cols = [c + "_per_min" for c in per_min_cols]
            all_vs_c1_cols = [c + "_vs_c1" for c in per_min_cols]
            
            additional_comparison = moto[["DateStart", "DateEnd", "Name", "Team"] + [c for c in all_per_min_cols + all_vs_c1_cols if c in moto.columns]]
            
            # Określenie rzędu startowego
            if excel_comparison_worksheet in writer.sheets:
                # Jeśli sekcja 1 była pusta, a to jest pierwsza sekcja do zapisu w Porównaniu, użyj wiersza 1
                if comparison_data.empty:
                     start_row = 1
                # Jeśli sekcja 1 była, kontynuuj po niej
                else:
                    start_row = len(comparison_data) + 3 # 1 wiersz danych + 2 nagłówki + 1 wolny wiersz = 4, ale + 3, bo startrow to index
            else:
                 # Jeśli arkusz jeszcze nie istnieje (co nie powinno się zdarzyć, jeśli moto nie jest puste, ale na wszelki wypadek)
                 start_row = 1
            
            worksheet = writer.sheets[excel_comparison_worksheet]
            
            # Dodaj nagłówek dla danych porównania
            worksheet.write(start_row - 1, 0, "PORÓWNANIE – metryki per minutę i porównanie z C1 (Dodatkowe)")
            
            additional_comparison.to_excel(writer, sheet_name=excel_comparison_worksheet, index=False, startrow=start_row)


        output.seek(0)
        return output


# ===================== NAGŁÓWEK + NAWIGACJA (pozostawione z Twojego kodu) =====================
with st.sidebar:
    st.markdown("---")
    st.header(" Nawigacja")
    sekcja = st.radio(
        "Sekcja",
        [" Dodawanie danych", " Dane", " Analiza"],
        key="nav_section",
    )
    if sekcja == " Dodawanie danych":
        page = st.radio(
            "Strona",
            ["Zawodnicy & Zespoły", "FANTASYPASY (wpis)", "MOTORYKA (wpis)", "Okresy/Testy"],
            key="nav_page_add",
        )
    elif sekcja == " Dane":
        page = st.radio(
            "Strona",
            ["Podgląd danych"],
            key="nav_page_data",
        )
    else:
    	page = st.radio(
        	"Strona",
        	["Porównania", "Analiza (pozycje & zespoły)", "Wykresy zmian", "Indeks – porównania", "Profil zawodnika"],
        	key="nav_page_ana",
    	)


st.title(" Cracovia – rejestry, wprowadzanie danych i analizy")

def extract_positions(series: pd.Series) -> list:
    all_pos = set()
    for val in series.dropna():
        parts = str(val).replace("\\", "/").split("/")
        for p in parts:
            p = p.strip().upper()
            if p:
                all_pos.add(p)
    return sorted(all_pos)

# ===================== ZAWODNICY & ZESPOŁY =====================
if page == "Zawodnicy & Zespoły":
    st.subheader("Rejestr zespołów i zawodników")
    if not st.session_state.get("auth", False):
        render_login_inline("regs")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Zespoły")
            team_to_add = st.text_input("Dodaj/zmień nazwę zespołu", key="reg_team_input")
            if st.button(" Zapisz zespół", use_container_width=True, key="reg_team_save"):
                if team_to_add.strip():
                    # POPRAWKA: Zmiana na składnię SQLite ON CONFLICT
                    upsert(
                        "INSERT INTO teams (Team) VALUES (:t) "
                        "ON CONFLICT(Team) DO UPDATE SET Team=excluded.Team;",
                        {"t": team_to_add.strip()},
                    )
                    st.success("Zapisano zespół.")
                else:
                    st.warning("Podaj nazwę.")
            st.dataframe(
                fetch_df("SELECT Team FROM teams ORDER BY Team;"),
                use_container_width=True,
                hide_index=True,
            )

        with c2:
            st.markdown("Zawodnik")
            teams_list = get_team_list()
            with st.form("player_form"):
                p_name = st.text_input("Imię i nazwisko *", key="reg_player_name")
                p_team = st.selectbox("Zespół", teams_list, index=0 if teams_list else 0, key="reg_team_select")
                p_pos  = st.multiselect(
                    "Domyślne pozycje (możesz wybrać kilka)",
                    POS_OPTIONS,
                    default=["ŚP"] if "ŚP" in POS_OPTIONS else [],
                    key="reg_pos_multi"
                )
                ok = st.form_submit_button(" Zapisz / nadpisz zawodnika", type="primary")
            if ok:
                if p_name.strip():
                    upsert_player(p_name, p_team, "/".join(p_pos) if p_pos else None)
                    st.success("Zawodnik zapisany/zmieniony.")
                else:
                    st.error("Podaj imię i nazwisko.")
            st.dataframe(
                fetch_df("SELECT Name, Team, Position, UpdatedAt FROM players ORDER BY Name;"),
                use_container_width=True,
            )

# ===================== OKRESY/TESTY =====================
elif page == "Okresy/Testy":
    st.subheader("Okresy / Testy (etykieta + zakres dat)")
    if not st.session_state.get("auth", False):
        render_login_inline("periods")
    else:
        with st.form("period_form"):
            label = st.text_input("Etykieta okresu/testu *", value="Test szybkości", key="per_label")
            ds = st.date_input("DateStart *", value=date.today(), key="per_ds")
            de = st.date_input("DateEnd *", value=date.today(), key="per_de")
            ok = st.form_submit_button(" Zapisz okres", type="primary")
        if ok:
            if label and ds and de:
                # POPRAWKA: Zmiana na składnię SQLite ON CONFLICT
                upsert("""
                    INSERT INTO measurement_periods (Label, DateStart, DateEnd)
                    VALUES (:l, :ds, :de)
                    ON CONFLICT(Label, DateStart, DateEnd) DO UPDATE SET Label=excluded.Label
                """, {"l": label.strip(), "ds": ds, "de": de})
                st.success("Okres zapisany.")
            else:
                st.error("Uzupełnij pola z gwiazdką.")
        st.dataframe(
            fetch_df("SELECT PeriodID, Label, DateStart, DateEnd FROM measurement_periods ORDER BY DateStart DESC, Label;"),
            use_container_width=True,
        )

# ===================== FANTASYPASY (WPIS) =====================
elif page == "FANTASYPASY (wpis)":
    st.subheader("Wpisz statystyki – FANTASYPASY")
    if not st.session_state.get("auth", False):
        render_login_inline("fant")
    else:
        players_df = fetch_df("SELECT Name, Team, Position FROM players ORDER BY Name;")
        if players_df.empty:
            st.info("Brak zawodników w rejestrze. Dodaj w „Zawodnicy & Zespoły”.")
        else:
            name = st.selectbox("Zawodnik *", players_df["Name"].tolist(), key="f_pick_player")
            prow = players_df[players_df["Name"] == name].iloc[0]
            team = (prow["Team"] or "")
            default_pos_list = [p.strip() for p in str(prow["Position"] or "").replace("\\","/").split("/") if p.strip()]
            st.caption(f"Zespół: **{team or '—'}** •  Domyślne pozycje: **{('/'.join(default_pos_list) or '—')}**")

            # Zakres
            mode_f = st.radio("Zakres dat do zapisu", ["Okres/Test z rejestru", "Z istniejących par dat (FANTASYPASY)", "Ręcznie"],
                              horizontal=True, key="f_mode")
            ds, de = None, None
            if mode_f == "Okres/Test z rejestru":
                periods = fetch_df("SELECT PeriodID, Label, DateStart, DateEnd FROM measurement_periods ORDER BY DateStart DESC;")
                if periods.empty:
                    st.info("Brak okresów – wybierz inną opcję.")
                else:
                    labels = [f"{r.Label} [{r.DateStart}→{r.DateEnd}]" for _, r in periods.iterrows()]
                    pick = st.selectbox("Okres/Test", labels, index=0, key="f_pick_period")
                    sel = periods.iloc[labels.index(pick)]
                    ds, de = sel["DateStart"], sel["DateEnd"]
                    st.caption(f"Zakres: {ds} → {de}")
            elif mode_f == "Z istniejących par dat (FANTASYPASY)":
                pairs = fetch_df("""SELECT DISTINCT DateStart, DateEnd FROM fantasypasy_stats
                                    ORDER BY DateStart DESC, DateEnd DESC""")
                if pairs.empty:
                    st.info("Brak zapisanych par dat – wybierz inną opcję.")
                else:
                    opts = [f"{r.DateStart} → {r.DateEnd}" for _, r in pairs.iterrows()]
                    pick = st.selectbox("Para dat", opts, index=0, key="f_pick_pair")
                    sel = pairs.iloc[opts.index(pick)]
                    ds, de = sel["DateStart"], sel["DateEnd"]
                    st.caption(f"Zakres: {ds} → {de}")
            else:
                c3, c4 = st.columns(2)
                ds = c3.date_input("DateStart *", key="f_ds")
                de = c4.date_input("DateEnd *",   key="f_de")

            # Pozycje tego wpisu
            pos_multi = st.multiselect(
                "Pozycje w tym wpisie",
                POS_OPTIONS,
                default=[p for p in default_pos_list if p in POS_OPTIONS] or ["ŚP" if "ŚP" in POS_OPTIONS else POS_OPTIONS[0]],
                key="f_pos_multi"
            )
            upd_default_pos = st.checkbox("Zaktualizuj domyślne pozycje w rejestrze na wybrane powyżej",
                                          value=False, key="f_update_default_pos")

            # Metryki
            st.markdown("Podstawy")
            c1, c2 = st.columns(2)
            number_of_games = c1.number_input("NumberOfGames", min_value=0, step=1, key="f_games")
            minutes        = c2.number_input("Minutes",        min_value=0, step=1, key="f_minutes")

            # --- OFENSYWA ---
            st.markdown("Ofensywa")
            c1, c2, c3, c4 = st.columns(4)
            goal          = c1.number_input("Goal",         min_value=0, step=3, key="f_goal")
            assist        = c2.number_input("Assist",       min_value=0, step=2, key="f_assist")
            chance_assist = c3.number_input("ChanceAssist", min_value=0, step=2, key="f_chance")
            key_pass      = c4.number_input("KeyPass",      min_value=0, step=2, key="f_keypass")

            c1, c2, c3 = st.columns(3)
            key_loss     = c1.number_input("KeyLoss (≤0)",         value=0, step=2, max_value=0, key="f_keyloss")
            finalization = c2.number_input("Finalization",         min_value=0, step=1, key="f_final")
            key_ind_act  = c3.number_input("KeyIndividualAction",  min_value=0, step=2, key="f_keyind")

# --- STRATY / OBRONA ---
            st.markdown("Straty / obrona")
            c1, c2, c3, c4 = st.columns(4)
            key_recover   = c1.number_input("KeyRecover",            min_value=0, step=2, key="f_keyrec")
            duel_win_box  = c2.number_input("DuelWinInBox",          min_value=0, step=2, key="f_duelwin")
            duel_loss_box = c3.number_input("DuelLossInBox (≤0)",    value=0, step=2, max_value=0, key="f_duelloss")
            block_shot    = c4.number_input("BlockShot",             min_value=0, step=1, key="f_block")

            c1, c2, c3 = st.columns(3)
            miss_block     = c2.number_input("MissBlockShot (≤0)",     value=0, step=1, max_value=0, key="f_missblock")
            duel_loss_out  = c1.number_input("DuelLossOutOfBox (≤0)", value=0, step=1, max_value=0, key="f_duellossout")
            rescue_action  = c3.number_input("RescueAction",           min_value=0, step=2, key="f_rescue")



            if st.button(" Zapisz do fantasypasy_stats", type="primary", key="f_save_btn"):
                if not (name and team is not None and pos_multi and ds and de):
                    st.error("Uzupełnij zawodnika, zakres dat i pozycje.")
                else:
                    pos_str = "/".join(pos_multi)
                    if upd_default_pos:
                        upsert_player(name, team, pos_str)  # nadpis domyślnych pozycji w rejestrze
                        
                    # POPRAWKA: Obliczanie i dodanie PktOff/PktDef do INSERT/UPDATE
                    pkt_off = int(goal) + int(assist) + int(chance_assist) + int(key_pass) + int(key_loss) + int(finalization) + int(key_ind_act)
                    pkt_def = int(key_recover) + int(duel_win_box) + int(duel_loss_box) + int(duel_loss_out) + int(block_shot) + int(miss_block) + int(rescue_action)


                    # POPRAWKA: Zmiana składni na SQLite ON CONFLICT
                    sql = """
                    INSERT INTO fantasypasy_stats
                      (Name, Team, Position, DateStart, DateEnd,
                       NumberOfGames, Minutes, Goal, Assist, ChanceAssist, KeyPass,
                       KeyLoss, DuelLossInBox, DuelLossOutBox, MissBlockShot,
                       Finalization, KeyIndividualAction, KeyRecover, DuelWinInBox, BlockShot, RescueAction,
                       PktOff, PktDef)
                    VALUES
                      (:Name, :Team, :Position, :DateStart, :DateEnd,
                       :NumberOfGames, :Minutes, :Goal, :Assist, :ChanceAssist, :KeyPass,
                       :KeyLoss, :DuelLossInBox, :DuelLossOutBox, :MissBlockShot,
                       :Finalization, :KeyIndividualAction, :KeyRecover, :DuelWinInBox, :BlockShot, :RescueAction,
                       :PktOff, :PktDef)
                    ON CONFLICT(Name, DateStart, DateEnd) DO UPDATE SET
                       Team=excluded.Team,
                       Position=excluded.Position,
                       NumberOfGames=excluded.NumberOfGames,
                       Minutes=excluded.Minutes,
                       Goal=excluded.Goal,
                       Assist=excluded.Assist,
                       ChanceAssist=excluded.ChanceAssist,
                       KeyPass=excluded.KeyPass,
                       KeyLoss=excluded.KeyLoss,
                       DuelLossInBox=excluded.DuelLossInBox,
                       DuelLossOutBox=excluded.DuelLossOutBox,
                       MissBlockShot=excluded.MissBlockShot,
                       Finalization=excluded.Finalization,
                       KeyIndividualAction=excluded.KeyIndividualAction,
                       KeyRecover=excluded.KeyRecover,
                       DuelWinInBox=excluded.DuelWinInBox,
                       BlockShot=excluded.BlockShot,
                       RescueAction=excluded.RescueAction,
                       PktOff=excluded.PktOff,
                       PktDef=excluded.PktDef
                    """

                    params = {
                        "Name": name, "Team": team, "Position": pos_str,
                        "DateStart": ds, "DateEnd": de,
                        "NumberOfGames": int(number_of_games), "Minutes": int(minutes),
                        "Goal": int(goal), "Assist": int(assist), "ChanceAssist": int(chance_assist),
                        "KeyPass": int(key_pass), "KeyLoss": int(key_loss),
                        "DuelLossInBox": int(duel_loss_box), "DuelLossOutBox": int(duel_loss_out),
                        "MissBlockShot": int(miss_block),
                        "Finalization": int(finalization), "KeyIndividualAction": int(key_ind_act),
                        "KeyRecover": int(key_recover), "DuelWinInBox": int(duel_win_box),
                        "BlockShot": int(block_shot), "RescueAction": int(rescue_action),
                        "PktOff": pkt_off, "PktDef": pkt_def
                    }

                    try:
                        upsert(sql, params)
                        st.success("Zapisano / zaktualizowano rekord (FANTASYPASY).")
                    except PermissionError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Błąd zapisu: {e}")            
# ===================== MOTORYKA (WPIS) =====================
elif page == "MOTORYKA (wpis)":
    st.subheader("Wpisz statystyki – MOTORYKA")
    if not st.session_state.get("auth", False):
        render_login_inline("moto")
    else:
        players_df = fetch_df("SELECT Name, Team, Position FROM players ORDER BY Name;")
        if players_df.empty:
            st.info("Brak zawodników w rejestrze. Dodaj w „Zawodnicy & Zespoły”.")
        else:
            name2 = st.selectbox("Zawodnik *", players_df["Name"].tolist(), key="m_pick_player")
            prow = players_df[players_df["Name"] == name2].iloc[0]
            team2 = (prow["Team"] or "")
            default_pos_list = [p.strip() for p in str(prow["Position"] or "").replace("\\","/").split("/") if p.strip()]
            st.caption(f"Zespół: **{team2 or '—'}** •  Domyślne pozycje: **{('/'.join(default_pos_list) or '—')}**")

            mode_m = st.radio("Zakres dat do zapisu", ["Okres/Test z rejestru", "Z istniejących par dat (MOTORYKA)", "Ręcznie"],
                              horizontal=True, key="m_mode")
            ds2, de2 = None, None
            if mode_m == "Okres/Test z rejestru":
                periods = fetch_df("SELECT PeriodID, Label, DateStart, DateEnd FROM measurement_periods ORDER BY DateStart DESC;")
                if periods.empty:
                    st.info("Brak okresów – wybierz inną opcję.")
                else:
                    labels = [f"{r.Label} [{r.DateStart}→{r.DateEnd}]" for _, r in periods.iterrows()]
                    pick = st.selectbox("Okres/Test", labels, index=0, key="m_pick_period")
                    sel = periods.iloc[labels.index(pick)]
                    ds2, de2 = sel["DateStart"], sel["DateEnd"]
                    st.caption(f"Zakres: {ds2} → {de2}")
            elif mode_m == "Z istniejących par dat (MOTORYKA)":
                pairs = fetch_df("""SELECT DISTINCT DateStart, DateEnd FROM motoryka_stats
                                    ORDER BY DateStart DESC, DateEnd DESC""")
                if pairs.empty:
                    st.info("Brak zapisanych par dat – wybierz inną opcję.")
                else:
                    opts = [f"{r.DateStart} → {r.DateEnd}" for _, r in pairs.iterrows()]
                    pick = st.selectbox("Para dat", opts, index=0, key="m_pick_pair")
                    sel = pairs.iloc[opts.index(pick)]
                    ds2, de2 = sel["DateStart"], sel["DateEnd"]
                    st.caption(f"Zakres: {ds2} → {de2}")
            else:
                c3, c4 = st.columns(2)
                ds2 = c3.date_input("DateStart *", key="m_ds")
                de2 = c4.date_input("DateEnd *",   key="m_de")

            pos2_multi = st.multiselect(
                "Pozycje w tym wpisie",
                POS_OPTIONS,
                default=[p for p in default_pos_list if p in POS_OPTIONS] or ["ŚP" if "ŚP" in POS_OPTIONS else POS_OPTIONS[0]],
                key="m_pos_multi"
            )
            upd_default_pos2 = st.checkbox("Zaktualizuj domyślne pozycje w rejestrze na wybrane powyżej",
                                           value=False, key="m_update_default_pos")

            minutes2 = st.number_input("Minutes (≥0)", min_value=0, step=1, key="m_minutes")
            c1, c2, c3 = st.columns(3)
            td_m      = c1.number_input("TD_m",     min_value=0, step=1, key="m_td")
            hsr_m     = c2.number_input("HSR_m",    min_value=0, step=1, key="m_hsr")
            sprint_m  = c3.number_input("Sprint_m", min_value=0, step=1, key="m_sprint")
            c1, c2 = st.columns(2)
            acc       = c1.number_input("ACC",   min_value=0, step=1, key="m_acc")
            decel     = c2.number_input("DECEL", min_value=0, step=1, key="m_decel")

            if st.button(" Zapisz do motoryka_stats", type="primary", key="m_save_btn"):
                if not (name2 and team2 is not None and pos2_multi and ds2 and de2):
                    st.error("Uzupełnij zawodnika, zakres dat i pozycje.")
                else:
                    pos_str2 = "/".join(pos2_multi)
                    if upd_default_pos2:
                        upsert_player(name2, team2, pos_str2)
                        
                    # POPRAWKA: Obliczenie i dodanie PlayerIntensityIndex do INSERT/UPDATE
                    pii = int(hsr_m) * 1.0 + int(sprint_m) * 1.5 + int(acc) * 2.0 + int(decel) * 2.0


                    # POPRAWKA: Zmiana składni na SQLite ON CONFLICT
                    sql = """
                    INSERT INTO motoryka_stats
                      (Name, Team, Position, DateStart, DateEnd,
                       Minutes, TD_m, HSR_m, Sprint_m, ACC, DECEL,
                       PlayerIntensityIndex, PlayerIntensityIndexComparingToTeamAverage)
                    VALUES
                      (:Name, :Team, :Position, :DateStart, :DateEnd,
                       :Minutes, :TD_m, :HSR_m, :Sprint_m, :ACC, :DECEL,
                       :PII, 0)
                    ON CONFLICT(Name, DateStart, DateEnd) DO UPDATE SET
                       Team=excluded.Team,
                       Position=excluded.Position,
                       Minutes=excluded.Minutes,
                       TD_m=excluded.TD_m,
                       HSR_m=excluded.HSR_m,
                       Sprint_m=excluded.Sprint_m,
                       ACC=excluded.ACC,
                       DECEL=excluded.DECEL,
                       PlayerIntensityIndex=excluded.PlayerIntensityIndex
                    """
                    params = {
                        "Name": name2, "Team": team2, "Position": pos_str2,
                        "DateStart": ds2, "DateEnd": de2,
                        "Minutes": int(minutes2),
                        "TD_m": int(td_m),
                        "HSR_m": int(hsr_m),
                        "Sprint_m": int(sprint_m),
                        "ACC": int(acc),
                        "DECEL": int(decel),
                        "PII": pii
                    }
                    try:
                        upsert(sql, params)
                        st.success("Zapisano / zaktualizowano rekord (MOTORYKA).")
                    except PermissionError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Błąd zapisu: {e}")

# ===================== PODGLĄD DANYCH =====================
elif page == "Podgląd danych":
    st.subheader("Podgląd ostatnich wpisów")
    opt = st.selectbox("Tabela", ["fantasypasy_stats", "motoryka_stats", "all_stats", "players", "teams", "measurement_periods"],
                       key="preview_table_pick")
    try:
        st.dataframe(fetch_df(f"SELECT * FROM `{opt}` ORDER BY 1 DESC LIMIT 500;"),
                     use_container_width=True)
    except Exception as e:
        st.error(f"Nie mogę wczytać: {e}")
# ===================== PORÓWNANIA (Młodzież vs C1) =====================
def load_motoryka_for_compare(date_start=None, date_end=None):
    # Używamy all_stats, aby zapewnić, że PlayerIntensityIndex jest obliczony
    sql = """
        SELECT Name, Team, Position, DateStart, DateEnd,
               Minutes, HSR_m, Sprint_m, ACC, DECEL, PlayerIntensityIndex
        FROM all_stats
        WHERE PlayerIntensityIndex IS NOT NULL
    """
    params = {}
    if date_start:
        sql += " AND DateStart >= :ds"; params["ds"] = date_start
    if date_end:
        sql += " AND DateEnd <= :de"; params["de"] = date_end
    sql += " ORDER BY DateStart DESC"
    return fetch_df(sql, params)

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
                 .agg({"HSR_m":"mean","Sprint_m":"mean","ACC":"mean","DECEL":"mean","PlayerIntensityIndex":"mean"})
                 .rename(columns={"HSR_m":"HSR_m_C1","Sprint_m":"Sprint_m_C1","ACC":"ACC_C1","DECEL":"DECEL_C1","PlayerIntensityIndex":"PII_C1"})
          ).reset_index()
    return grp

def add_diffs(df, ref, by_position=True):
    import numpy as np
    df = df.copy()

    # 1) Normalizacja ewentualnych sufiksów po merge
    for base in ["TD_m","HSR_m","Sprint_m","ACC","DECEL","PlayerIntensityIndex"]:
        for suf in ["_x","_y"]:
            col = base + suf
            if col in df.columns and base not in df.columns:
                df.rename(columns={col: base}, inplace=True)

    # 2) Metryki, które realnie mamy
    metrics_all = ["HSR_m","Sprint_m","ACC","DECEL","PlayerIntensityIndex"]
    metrics = [m for m in metrics_all if m in df.columns]

    if not metrics:
        return df

    # 3) Jeśli ref to DataFrame (tak działa Twój kod w "Porównania")
    if isinstance(ref, pd.DataFrame):
        if by_position:
            # ref_c1_by_position: kolumny jak HSR_m_C1, Sprint_m_C1, ACC_C1, DECEL_C1, PII_C1
            # scal po pozycji
            df = df.merge(ref, on="Position", how="left")
            for m in metrics:
                ref_col = "PII_C1" if m == "PlayerIntensityIndex" and "PII_C1" in df.columns else f"{m}_C1"
                if ref_col in df.columns:
                    df[m + "_diff"] = pd.to_numeric(df[m], errors="coerce") - pd.to_numeric(df[ref_col], errors="coerce")
                    denom = pd.to_numeric(df[ref_col], errors="coerce").replace(0, np.nan)
                    df[m + "_pct"] = pd.to_numeric(df[m], errors="coerce") / denom
        else:
            # ref_c1_global: to 1-wierszowy DF z kolumnami HSR_m, Sprint_m, ACC, DECEL, PlayerIntensityIndex oraz __key_global__=1
            df["__key_global__"] = 1
            df = df.merge(ref, on="__key_global__", how="left", suffixes=("", "_C1"))
            for m in metrics:
                ref_col = f"{m}_C1" if f"{m}_C1" in df.columns else m  # zabezpieczenie
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


if page == "Porównania":
    st.subheader("Porównanie młodzieży do pierwszego zespołu C1")

    mode_cmp = st.radio("Wybór zakresu", ["Okres/Test z rejestru", "Z istniejących par dat", "Ręcznie"],
                        horizontal=True, key="cmp_mode")
    ds_f, de_f = None, None
    if mode_cmp == "Okres/Test z rejestru":
        periods = fetch_df("SELECT PeriodID, Label, DateStart, DateEnd FROM measurement_periods ORDER BY DateStart DESC;")
        if periods.empty:
            st.info("Brak zapisanych okresów – wybierz inny tryb.")
        else:
            labels = [f"{r.Label} [{r.DateStart}→{r.DateEnd}]" for _, r in periods.iterrows()]
            pick = st.selectbox("Okres/Test", labels, index=0, key="cmp_pick_period")
            sel = periods.iloc[labels.index(pick)]
            ds_f, de_f = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_f} → {de_f}")
    elif mode_cmp == "Z istniejących par dat":
        pairs = fetch_df("SELECT DISTINCT DateStart, DateEnd FROM motoryka_stats ORDER BY DateStart DESC, DateEnd DESC")
        if pairs.empty:
            st.info("Brak danych – wybierz „Ręcznie”.")
        else:
            opts = [f"{r.DateStart} → {r.DateEnd}" for _, r in pairs.iterrows()]
            pick = st.selectbox("Para dat", opts, index=0, key="cmp_pick_pair")
            sel = pairs.iloc[opts.index(pick)]
            ds_f, de_f = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_f} → {de_f}")
    else:
        c1, c2 = st.columns(2)
        ds_f = c1.date_input("Od (DateStart)", key="cmp_ds")
        de_f = c2.date_input("Do (DateEnd)", key="cmp_de")

    scope = st.radio("Zakres referencji C1", ["Globalna średnia C1", "Średnie C1 per pozycja"],
                     horizontal=True, key="cmp_scope")

    df_m = load_motoryka_for_compare(ds_f, de_f)
    if df_m.empty:
        st.info("Brak danych w wybranym zakresie.")
    else:
        show_only_non_c1 = st.toggle("Pokaż tylko zawodników spoza C1", value=True, key="cmp_non_c1_only")
        df_view = df_m[df_m["Team"] != "C1"].copy() if show_only_non_c1 else df_m.copy()

        all_positions = sorted(set(_explode_positions(df_m)["Position"].unique().tolist()))
        pos_pick_multi = st.multiselect("Filtr pozycji (hybrydy wliczają się automatycznie):",
                                        options=all_positions, default=[], key="cmp_pos_multi")
        if pos_pick_multi:
            df_view = _explode_positions(df_view)
            df_view = df_view[df_view["Position"].isin(pos_pick_multi)]

        players = sorted(df_view["Name"].dropna().unique().tolist())
        pick_players = st.multiselect("Zawodnicy do tabeli", players, default=players[:10], key="cmp_players_multi")
        if pick_players:
            df_view = df_view[df_view["Name"].isin(pick_players)]

        ref = ref_c1_global(df_m) if scope == "Globalna średnia C1" else ref_c1_by_position(df_m)
    if ref is None:
        st.info("Brak referencji C1 dla wybranego zakresu.")
    else:
        df_comp = add_diffs(df_view, ref, by_position=(scope != "Globalna średnia C1"))
        metryki = ["HSR_m", "Sprint_m", "ACC", "DECEL", "PlayerIntensityIndex"]
        kolumny = ["Name", "Team", "Position", "DateStart", "DateEnd"] + metryki \
                  + [m+"_diff" for m in metryki] + [m+"_pct" for m in metryki]
        st.dataframe(
            df_comp[kolumny].sort_values(["Position","Name","DateStart"], ascending=[True,True,False]),
            use_container_width=True
        )

        # --- LEGENDA POD TABELĄ (TYLKO DLA "Porównania") ---
        st.markdown("""
<div style='margin-top: 1rem; font-size: 0.9rem; line-height: 1.5;'>
<b>Legenda kolumn:</b><br>
• <b>HSR_m</b>, <b>Sprint_m</b>, <b>ACC</b>, <b>DECEL</b>, <b>PlayerIntensityIndex</b> – wartości surowe (średnie lub sumy z motoryki).<br>
• <b>_diff</b> – różnica między zawodnikiem a średnią zespołu C1 (dla wybranego zakresu i trybu porównania).<br>
&nbsp;&nbsp;&nbsp;&nbsp;Pozytywna wartość = zawodnik powyżej średniej C1, ujemna = poniżej.<br>
• <b>_pct</b> – stosunek wartości zawodnika do średniej C1 (np. 1.10 = 110%).<br>
• <b>Position</b> – pozycja z której pochodzi wpis (uwzględnia hybrydy).<br>
• <b>DateStart / DateEnd</b> – zakres okresu/testu, z którego pochodzi pomiar.<br>
</div>
""", unsafe_allow_html=True)


# ===================== ANALIZA (pozycje & zespoły) =====================
@st.cache_data(show_spinner=False)
def load_fantasy(date_start=None, date_end=None, teams=None):
    # POPRAWKA: Ręczne obliczanie PktOff i PktDef w load_fantasy
    sql = """
        SELECT Name, Team, Position, DateStart, DateEnd,
               NumberOfGames, Minutes,
               Goal, Assist, ChanceAssist, KeyPass,
               KeyLoss, DuelLossInBox, DuelLossOutBox, MissBlockShot,
               Finalization, KeyIndividualAction, KeyRecover, DuelWinInBox, BlockShot,
               RescueAction 
        FROM fantasypasy_stats
        WHERE 1=1
    """
    params = {}
    if date_start and date_end:
        sql += " AND NOT (DateEnd < :ds OR DateStart > :de)"
        params["ds"] = date_start
        params["de"] = date_end
    elif date_start:
        sql += " AND DateEnd >= :ds"     # cokolwiek kończy się po ds
        params["ds"] = date_start
    elif date_end:
        sql += " AND DateStart <= :de"   # cokolwiek zaczyna się przed de
        params["de"] = date_end

    if teams:
        sql += " AND Team IN :teams"
        params["teams"] = tuple(teams)

    sql += " ORDER BY DateStart DESC"
    
    df = fetch_df(sql, params)

    if not df.empty:
        # Dodaj obliczenia kolumn PktOff i PktDef (zastępują GENERATED ALWAYS AS)
        df["PktOff"] = (pd.to_numeric(df["Goal"], errors='coerce') + pd.to_numeric(df["Assist"], errors='coerce') + 
                        pd.to_numeric(df["ChanceAssist"], errors='coerce') + pd.to_numeric(df["KeyPass"], errors='coerce') + 
                        pd.to_numeric(df["KeyLoss"], errors='coerce') + pd.to_numeric(df["Finalization"], errors='coerce') + 
                        pd.to_numeric(df["KeyIndividualAction"], errors='coerce'))
        df["PktDef"] = (pd.to_numeric(df["KeyRecover"], errors='coerce') + pd.to_numeric(df["DuelWinInBox"], errors='coerce') + 
                        pd.to_numeric(df["DuelLossInBox"], errors='coerce') + pd.to_numeric(df["DuelLossOutBox"], errors='coerce') + 
                        pd.to_numeric(df["BlockShot"], errors='coerce') + pd.to_numeric(df["MissBlockShot"], errors='coerce') + 
                        pd.to_numeric(df["RescueAction"], errors='coerce'))

    return df


@st.cache_data(show_spinner=False)
def load_motoryka_all(date_start=None, date_end=None, teams=None):
    # POPRAWKA: Ręczne obliczanie PlayerIntensityIndex w load_motoryka_all
    sql = """
        SELECT Name, Team, Position, DateStart, DateEnd,
               Minutes, TD_m, HSR_m, Sprint_m, ACC, DECEL,
               PlayerIntensityIndexComparingToTeamAverage 
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
    
    df = fetch_df(sql, params)

    if not df.empty:
        # Dodaj obliczenie PlayerIntensityIndex (zastępuje GENERATED ALWAYS AS)
        df["PlayerIntensityIndex"] = (pd.to_numeric(df["HSR_m"], errors='coerce') * 1.0 + 
                                      pd.to_numeric(df["Sprint_m"], errors='coerce') * 1.5 + 
                                      pd.to_numeric(df["ACC"], errors='coerce') * 2.0 + 
                                      pd.to_numeric(df["DECEL"], errors='coerce') * 2.0)

    return df


def _q25(x: pd.Series): return x.quantile(0.25)
def _q75(x: pd.Series): return x.quantile(0.75)

def flat_agg(df: pd.DataFrame, group_cols: list, num_cols: list) -> pd.DataFrame:
    df2 = df.copy()
    for c in num_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")
    agg = df2.groupby(group_cols, dropna=False)[num_cols].agg(
        ['count','sum','mean','median','std','min','max',_q25,_q75]
    )
    try:
        agg.rename(columns={'_q25':'q25','_q75':'q75'}, level=1, inplace=True)
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

if page == "Analiza (pozycje & zespoły)":
    st.subheader("Analiza statystyk – per pozycja i per team (obie tabele)")

    teams_pick = st.multiselect("Zespoły (puste = wszystkie)", get_team_list(), default=[],
                                key="an_team_multi")
    src = st.radio("Tabela", ["FANTASYPASY", "MOTORYKA"], horizontal=True, key="an_src")

    mode_a = st.radio("Wybór zakresu", ["Okres/Test z rejestru", "Z istniejących par dat", "Ręcznie"],
                      horizontal=True, key="an_mode")
    ds_a, de_a = None, None
    if mode_a == "Okres/Test z rejestru":
        periods = fetch_df("SELECT PeriodID, Label, DateStart, DateEnd FROM measurement_periods ORDER BY DateStart DESC;")
        if periods.empty:
            st.info("Brak zapisanych okresów – wybierz inną opcję.")
        else:
            labels = [f"{r.Label} [{r.DateStart}→{r.DateEnd}]" for _, r in periods.iterrows()]
            pick = st.selectbox("Okres/Test", labels, index=0, key="an_pick_period")
            sel = periods.iloc[labels.index(pick)]
            ds_a, de_a = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_a} → {de_a}")
    elif mode_a == "Z istniejących par dat":
        table = "motoryka_stats" if src == "MOTORYKA" else "fantasypasy_stats"
        pairs = fetch_df(f"SELECT DISTINCT DateStart, DateEnd FROM {table} ORDER BY DateStart DESC, DateEnd DESC")
        if pairs.empty:
            st.info(f"Brak danych w {table} – wybierz „Ręcznie”.")
        else:
            opts = [f"{r.DateStart} → {r.DateEnd}" for _, r in pairs.iterrows()]
            pick = st.selectbox("Para dat", opts, index=0, key="an_pick_pair")
            sel = pairs.iloc[opts.index(pick)]
            ds_a, de_a = sel["DateStart"], sel["DateEnd"]
            st.caption(f"Zakres: {ds_a} → {de_a}")
    else:
        c1, c2 = st.columns(2)
        ds_a = c1.date_input("Od (DateStart)", value=None, key="an_ds")
        de_a = c2.date_input("Do (DateEnd)", value=None, key="an_de")

    if src == "FANTASYPASY":
        df = load_fantasy(ds_a, de_a, teams_pick or None)
        if df.empty:
            st.info("Brak danych w wybranym zakresie.")
        else:
            df_pos = _explode_positions(df)
            all_pos = sorted(df_pos["Position"].unique().tolist())
            pos_pick = st.multiselect("Filtr pozycji (hybrydy uwzględnione)", all_pos, default=[], key="an_pos_fant")
            if pos_pick:
                df_pos = df_pos[df_pos["Position"].isin(pos_pick)]

            num_cols = ["NumberOfGames","Minutes","Goal","Assist","ChanceAssist","KeyPass",
                        "KeyLoss","DuelLossInBox","MissBlockShot","Finalization","KeyIndividualAction",
                        "KeyRecover","DuelWinInBox","BlockShot","PktOff","PktDef"]

            st.markdown("Pozycja")
            agg_pos = add_per90_from_sums(
                flat_agg(df_pos, ["Position"], num_cols), "Minutes",
                ["Goal","Assist","ChanceAssist","KeyPass","KeyLoss","DuelLossInBox","MissBlockShot",
                 "Finalization","KeyIndividualAction","KeyRecover","DuelWinInBox","BlockShot","PktOff","PktDef"]
            )
            st.dataframe(agg_pos, use_container_width=True)
            download_button_for_df(agg_pos, " CSV (pozycja)", "fantasypasy_per_position.csv")

            st.markdown("Team")
            agg_team = add_per90_from_sums(
                flat_agg(df, ["Team"], num_cols), "Minutes",
                ["Goal","Assist","ChanceAssist","KeyPass","KeyLoss","DuelLossInBox","MissBlockShot",
                 "Finalization","KeyIndividualAction","KeyRecover","DuelWinInBox","BlockShot","PktOff","PktDef"]
            )
            st.dataframe(agg_team, use_container_width=True)
            download_button_for_df(agg_team, " CSV (team)", "fantasypasy_per_team.csv")

            st.markdown("Team × Pozycja")
            agg_pos_team = add_per90_from_sums(
                flat_agg(df_pos, ["Team","Position"], num_cols), "Minutes",
                ["Goal","Assist","ChanceAssist","KeyPass","KeyLoss","DuelLossInBox","MissBlockShot",
                 "Finalization","KeyIndividualAction","KeyRecover","DuelWinInBox","BlockShot","PktOff","PktDef"]
            )
            st.dataframe(agg_pos_team, use_container_width=True)
            download_button_for_df(agg_pos_team, " CSV (team×pozycja)", "fantasypasy_position_team.csv")

            # === LEGENDA (FANTASYPASY) ===
            st.markdown("""
<div style='margin-top: 1rem; font-size: 0.9rem; line-height: 1.5;'>
<b>Legenda kolumn (Analiza):</b><br>
• <b>__count</b> – liczba wpisów w grupie.<br>
• <b>__sum</b> – suma wartości w grupie.<br>
• <b>__mean</b> – średnia; <b>__median</b> – mediana; <b>__std</b> – odchylenie std.<br>
• <b>__min</b> / <b>__max</b> – min / max; <b>__q25</b> / <b>__q75</b> – kwartyle 25/75%.<br>
• <b>__per90</b> – przeliczenie na 90 minut (gdy dotyczy).<br>
</div>
""", unsafe_allow_html=True)

    else:
        df = load_motoryka_all(ds_a, de_a, teams_pick or None)
        if df.empty:
            st.info("Brak danych w wybranym zakresie.")
        else:
            df_pos = _explode_positions(df)
            all_pos = sorted(df_pos["Position"].unique().tolist())
            pos_pick = st.multiselect("Filtr pozycji (hybrydy uwzględnione)", all_pos, default=[], key="an_pos_moto")
            if pos_pick:
                df_pos = df_pos[df_pos["Position"].isin(pos_pick)]

            num_cols = ["Minutes","TD_m","HSR_m","Sprint_m","ACC","DECEL","PlayerIntensityIndex"]

            st.markdown("Pozycja")
            agg_pos = flat_agg(df_pos, ["Position"], num_cols)
            st.dataframe(agg_pos, use_container_width=True)
            download_button_for_df(agg_pos, " CSV (pozycja)", "motoryka_per_position.csv")

            st.markdown("Team")
            agg_team = flat_agg(df, ["Team"], num_cols)
            st.dataframe(agg_team, use_container_width=True)
            download_button_for_df(agg_team, " CSV (team)", "motoryka_per_team.csv")

            st.markdown("Team × Pozycja")
            agg_pos_team = flat_agg(df_pos, ["Team","Position"], num_cols)
            st.dataframe(agg_pos_team, use_container_width=True)
            download_button_for_df(agg_pos_team, " CSV (team×pozycja)", "motoryka_position_team.csv")

            # === LEGENDA (MOTORYKA) ===
            st.markdown("""
<div style='margin-top: 1rem; font-size: 0.9rem; line-height: 1.5;'>
<b>Legenda kolumn (Analiza):</b><br>
• <b>__count</b> – liczba wpisów w grupie.<br>
• <b>__sum</b> – suma wartości w grupie.<br>
• <b>__mean</b> – średnia; <b>__median</b> – mediana; <b>__std</b> – odchylenie std.<br>
• <b>__min</b> / <b>__max</b> – min / max; <b>__q25</b> / <b>__q75</b> – kwartyle 25/75%.<br>
• (MOTORYKA nie ma przeliczeń __per90 w tej tabeli).<br>
</div>
""", unsafe_allow_html=True)

# ===================== WYKRESY ZMIAN =====================
elif page == "Wykresy zmian":
    st.subheader("Wykresy zmian po dacie")
    st.caption("MOTORYKA – metryki zliczeniowe w przeliczeniu na minutę; bez indeksów.")

    # wybór zakresu dat (jak wcześniej)
    c1, c2 = st.columns(2)
    ds_w = c1.date_input("Od (DateStart)", value=None, key="plot_ds")
    de_w = c2.date_input("Do (DateEnd)", value=None, key="plot_de")

    # dane źródłowe
    dfw = load_motoryka_all(ds_w, de_w, None)
    if dfw.empty:
        st.info("Brak danych w wybranym zakresie.")
        st.stop()

    # dostępne metryki (bez indeksów)
    per_minute_base = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

    # wybór trybu: cała drużyna / pozycja / wybrani gracze
    mode_players = st.radio(
        "Zakres zawodników",
        ["Cała drużyna", "Pozycja", "Wybrani gracze"],
        horizontal=True,
        key="plot_players_mode"
    )

    # przygotowanie list pomocniczych
    teams_list = sorted(dfw["Team"].dropna().unique().tolist())
    positions_all = sorted(extract_positions(dfw["Position"]))

    selected_names = []
    if mode_players == "Cała drużyna":
        t = st.selectbox("Drużyna", teams_list if teams_list else [""], key="plot_pick_team")
        dff = dfw[dfw["Team"] == t].copy() if t else dfw.copy()
        selected_names = sorted(dff["Name"].dropna().unique().tolist())
    elif mode_players == "Pozycja":
        pos_pick = st.multiselect("Pozycja", positions_all, default=[], key="plot_pick_positions")
        if pos_pick:
            # rozbij hybrydy pozycji i filtruj
            dff_rows = []
            for _, r in dfw.iterrows():
                parts = str(r["Position"] or "").replace("\\", "/").split("/")
                if any(p.strip().upper() in pos_pick for p in parts):
                    dff_rows.append(r)
            dff = pd.DataFrame(dff_rows) if dff_rows else dfw.iloc[0:0].copy()
        else:
            dff = dfw.iloc[0:0].copy()
        selected_names = sorted(dff["Name"].dropna().unique().tolist())
    else:  // Wybrani gracze
        all_names = sorted(dfw["Name"].dropna().unique().tolist())
        selected_names = st.multiselect("Zawodnicy", all_names, default=all_names[:3], key="plot_pick_names")
        dff = dfw[dfw["Name"].isin(selected_names)].copy() if selected_names else dfw.iloc[0:0].copy()

    if dff.empty or not selected_names:
        st.info("Brak danych po zastosowaniu wyboru zawodników.")
        st.stop()

    // przeliczenie na minutę
    dff = dff.copy()
    dff["Minutes"] = pd.to_numeric(dff["Minutes"], errors="coerce").replace(0, np.nan)
    for m in per_minute_base:
        if m in dff.columns:
            dff[m + "_per_min"] = pd.to_numeric(dff[m], errors="coerce") / dff["Minutes"]
        else:
            dff[m + "_per_min"] = np.nan

    // oś czasu: środek zakresu
    mid = pd.to_datetime(dff["DateStart"]) + (pd.to_datetime(dff["DateEnd"]) - pd.to_datetime(dff["DateStart"])) / 2
    dff["DateMid"] = mid.dt.date

    // wybór metryki (tylko per_min)
    metric = st.selectbox(
        "Metryka (na minutę)",
        [m + "_per_min" for m in per_minute_base],
        key="plot_metric_per_min"
    )

    // przygotowanie danych do wykresu
    plot = dff[["Name", "DateMid", metric]].rename(columns={metric: "Value"}).dropna()
    if plot.empty:
        st.info("Brak wartości do wykresu dla wybranej metryki.")
        st.stop()

    // wykres: wiele osób, kolor = Name
    chart = (
        alt.Chart(plot)
        .mark_line(point=True)
        .encode(
            x=alt.X("DateMid:T", title="Data"),
            y=alt.Y("Value:Q", title=metric),
            color=alt.Color("Name:N", title="Zawodnik", legend=alt.Legend(columns=2)),
            tooltip=["DateMid:T", "Name:N", "Value:Q"]
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        plot.sort_values(["Name", "DateMid"]),
        use_container_width=True
    )

// ===================== FANTASY – PRZEGLĄD GRAFICZNY =====================
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

    if mode_fx == "Z istniejących par dat (FANTASYPASY)":
        pairs = fetch_df("""
            SELECT DISTINCT DateStart, DateEnd
            FROM fantasypasy_stats
            ORDER BY DateStart DESC, DateEnd DESC
        """)
        if pairs.empty:
            st.info("Brak zapisanych par dat w FANTASYPASY – wybierz inną opcję.")
        else:
            opts = [f"{r.DateStart} → {r.DateEnd}" for _, r in pairs.iterrows()]
            label_to_tuple = {f"{r.DateStart} → {r.DateEnd}": (r.DateStart, r.DateEnd) for _, r in pairs.iterrows()}

            if "fx_pick_pairs" not in st.session_state:
                st.session_state["fx_pick_pairs"] = []

            c1, c2 = st.columns([1, 3])
            if c1.button("Wybierz wszystkie", key="fx_btn_all_pairs"):
                st.session_state["fx_pick_pairs"] = opts

            picked = c2.multiselect("Pary dat", opts, key="fx_pick_pairs")
            selected_pairs = {label_to_tuple[x] for x in picked} if picked else None

    elif mode_fx == "Okres/Test z rejestru":
        periods = fetch_df("""
            SELECT PeriodID, Label, DateStart, DateEnd
            FROM measurement_periods
            ORDER BY DateStart DESC
        """)
        if periods.empty:
            st.info("Brak okresów w rejestrze – wybierz inną opcję.")
        else:
            labels = [f"{r.Label} [{r.DateStart}→{r.DateEnd}]" for _, r in periods.iterrows()]
            label_to_tuple = {f"{r.Label} [{r.DateStart}→{r.DateEnd}]": (r.DateStart, r.DateEnd) for _, r in periods.iterrows()}

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
        pick_df = pd.DataFrame(list(selected_pairs), columns=["DateStart","DateEnd"])
        // Konwersja dat na stringi dla poprawnego merge
        df["DateStart"] = df["DateStart"].astype(str)
        df["DateEnd"] = df["DateEnd"].astype(str)
        pick_df["DateStart"] = pick_df["DateStart"].astype(str)
        pick_df["DateEnd"] = pick_df["DateEnd"].astype(str)
        
        df = df.merge(pick_df, on=["DateStart","DateEnd"], how="inner")
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
            ["PktOff","PktDef","Goal","Assist","ChanceAssist","KeyPass",
             "KeyLoss","DuelLossInBox","MissBlockShot","Finalization",
             "KeyIndividualAction","KeyRecover","DuelWinInBox","BlockShot"],
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
                tooltip=["Name","Team","DateMid","PktOff","PktDef"]
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
                df.groupby(["Name","Team"], as_index=False)[metric]
                  .mean()
                  .sort_values(metric, ascending=False)
            )
            st.dataframe(rank_df.head(20), use_container_width=True)









elif page == "Indeks – porównania":
    st.subheader("Indeks – porównania i rankingi")

    c1, c2 = st.columns(2)
    ds_i = c1.date_input("Od (DateStart)", value=None, key="idx_ds_fix")
    de_i = c2.date_input("Do (DateEnd)", value=None, key="idx_de_fix")

    df = load_motoryka_all(ds_i, de_i, None)
    if df.empty:
        st.info("Brak danych w wybranym zakresie.")
        st.stop()

    df = df.copy()
    df["PlayerIntensityIndex"] = pd.to_numeric(df.get("PlayerIntensityIndex"), errors="coerce")
    df = df.dropna(subset=["PlayerIntensityIndex"])

    mid_i = pd.to_datetime(df["DateStart"]) + (pd.to_datetime(df["DateEnd"]) - pd.to_datetime(df["DateStart"])) / 2
    df["DateMid"] = mid_i.dt.date

    // PII_vs_team_avg: z bazy lub fallback względem 1. drużyny (np. C1)
    if "PlayerIntensityIndexComparingToTeamAverage" in df.columns and df["PlayerIntensityIndexComparingToTeamAverage"].notna().any():
        df["PII_vs_team_avg"] = pd.to_numeric(df["PlayerIntensityIndexComparingToTeamAverage"], errors="coerce")
    else:
        base_team = "C1"
        base_mean = df.loc[df["Team"] == base_team, "PlayerIntensityIndex"].mean()
        df["PII_vs_team_avg"] = df["PlayerIntensityIndex"] - base_mean

    mode = st.radio(
        "Tryb",
        ["Ranking ogólny", "Ranking per data", "Ranking per zespół", "Porównanie graczy"],
        horizontal=True,
        key="idx_mode_fix"
    )

    if mode == "Ranking ogólny":
        top_n = st.slider("Top N", 5, 50, 20, key="idx_top_all_fix")
        view = (df[["Name","Team","DateMid","PlayerIntensityIndex","PII_vs_team_avg"]]
                  .sort_values("PlayerIntensityIndex", ascending=False)
                  .head(top_n))
        st.dataframe(view, use_container_width=True)

    elif mode == "Ranking per data":
        dates = sorted(df["DateMid"].unique().tolist(), reverse=True)
        pick = st.multiselect("Daty", dates, default=dates[:3], key="idx_dates_fix")
        top_n = st.slider("Top N na datę", 3, 20, 10, key="idx_top_date_fix")
        if not pick:
            st.info("Wybierz co najmniej jedną datę.")
        else:
            out = []
            for d in pick:
                tmp = (df[df["DateMid"] == d]
                       .sort_values("PlayerIntensityIndex", ascending=False)
                       .head(top_n))
                tmp = tmp.assign(_Date=d)
                out.append(tmp)
            view = pd.concat(out, ignore_index=True)[["DateMid","Name","Team","PlayerIntensityIndex","PII_vs_team_avg"]]
            st.dataframe(view, use_container_width=True)
            chart = (
                alt.Chart(view.rename(columns={"PlayerIntensityIndex": "Value"}))
                .mark_bar()
                .encode(
                    y=alt.Y("Name:N", sort="-x", title="Zawodnik"),
                    x=alt.X("Value:Q", title="Indeks"),
                    color=alt.Color("Team:N", title="Zespół"),
                    column=alt.Column("DateMid:T", header=alt.Header(labelAngle=0))
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
                tmp = (df[df["Team"] == t]
                       .sort_values("PlayerIntensityIndex", ascending=False)
                       .head(top_n))
                tmp = tmp.assign(_Team=t)
                out.append(tmp)
            view = pd.concat(out, ignore_index=True)[["Team","Name","DateMid","PlayerIntensityIndex","PII_vs_team_avg"]]
            st.dataframe(view, use_container_width=True)
            chart = (
                alt.Chart(view.rename(columns={"PlayerIntensityIndex": "Value"}))
                .mark_bar()
                .encode(
                    y=alt.Y("Name:N", sort="-x", title="Zawodnik"),
                    x=alt.X("Value:Q", title="Indeks"),
                    color=alt.Color("Team:N", title="Zespół"),
                    column=alt.Column("Team:N", header=alt.Header(labelAngle=0))
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

        metric = st.radio("Metryka", ["PlayerIntensityIndex", "PII_vs_team_avg"], horizontal=True, key="idx_cmp_metric_fix")
        src = subset[["Name", "Team", "DateMid", metric]].copy()
        plot = src.rename(columns={metric: "Value"}).dropna(subset=["Value"])

        line = (
            alt.Chart(plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("DateMid:T", title="Data"),
                y=alt.Y("Value:Q", title=metric),
                color=alt.Color("Name:N", title="Zawodnik"),
                tooltip=["DateMid:T","Name:N","Team:N","Value:Q"]
            )
            .properties(height=420)
        )
        st.altair_chart(line, use_container_width=True)

        table = (subset.groupby("Name", as_index=False)[["PlayerIntensityIndex","PII_vs_team_avg"]]
                        .mean()
                        .sort_values(metric, ascending=False))
        st.dataframe(table, use_container_width=True)








elif page == "Profil zawodnika":
    st.subheader("Profil zawodnika – pełny przegląd")

    // --- wybór zawodnika ---
    players_df = fetch_df("SELECT Name, Team, Position FROM players ORDER BY Name;")
    players_list = players_df["Name"].tolist()
    p = st.selectbox("Zawodnik", players_list if players_list else [], key="prof_all_player")
    if not p:
        st.info("Dodaj zawodnika w rejestrze, aby zacząć.")
        st.stop()

    prow = players_df[players_df["Name"] == p].iloc[0] if not players_df.empty else None
    team_label = str(prow["Team"]) if prow is not None else "—"
    pos_str = str(prow["Position"] or "—")
    st.caption(f"**Zespół:** {team_label}  •  **Domyślne pozycje:** {pos_str}")

    // --- dane bazowe z bazy ---
    // Używamy load_motoryka_all i load_fantasy, aby mieć obliczone kolumny
    moto = load_motoryka_all(teams=None)
    moto = moto[moto["Name"] == p].sort_values("DateStart", ascending=True)

    fant = load_fantasy(teams=None)
    fant = fant[fant["Name"] == p].sort_values("DateStart", ascending=True)
    
    // --- bezpieczniki ---
    if moto is None or moto.empty:
        moto = pd.DataFrame()
    if fant is None or fant.empty:
        fant = pd.DataFrame()
    per_min_cols = ["TD_m", "HSR_m", "Sprint_m", "ACC", "DECEL"]

    // --- przygotowanie motoryki ---
    if not moto.empty:
        moto = moto.copy()
        moto["Minutes"] = pd.to_numeric(moto.get("Minutes"), errors="coerce").replace(0, np.nan)

        // metryki na minutę
        for m in per_min_cols:
            if m in moto.columns:
                moto[m + "_per_min"] = pd.to_numeric(moto[m], errors="coerce") / moto["Minutes"]
            else:
                moto[m + "_per_min"] = np.nan

        // środkowa data
        mid_m = pd.to_datetime(moto["DateStart"]) + (
            pd.to_datetime(moto["DateEnd"]) - pd.to_datetime(moto["DateStart"])
        ) / 2
        moto["DateMid"] = mid_m.dt.date

        // PlayerIntensityIndex jest już obliczony w load_motoryka_all
        moto["PlayerIntensityIndex"] = pd.to_numeric(moto.get("PlayerIntensityIndex"), errors="coerce")

        // PII_vs_team_avg:
        // 1) jeśli jest PlayerIntensityIndexComparingToTeamAverage – użyj jej
        // 2) inaczej policz względem średniej PII C1
        if (
            "PlayerIntensityIndexComparingToTeamAverage" in moto.columns
            and moto["PlayerIntensityIndexComparingToTeamAverage"].notna().any()
        ):
            moto["PII_vs_team_avg"] = pd.to_numeric(
                moto["PlayerIntensityIndexComparingToTeamAverage"],
                errors="coerce"
            )
        else:
            _all_moto = load_motoryka_all(None, None, None).copy()
            if not _all_moto.empty and "PlayerIntensityIndex" in _all_moto.columns:
                c1_mean_idx = pd.to_numeric(
                    _all_moto.loc[_all_moto["Team"] == "C1", "PlayerIntensityIndex"],
                    errors="coerce"
                ).mean()
            else:
                c1_mean_idx = np.nan
            moto["PII_vs_team_avg"] = moto["PlayerIntensityIndex"] - c1_mean_idx

    // --- przygotowanie FANTASYPASY ---
    if not fant.empty:
        fant = fant.copy()
        fant["Minutes"] = pd.to_numeric(fant.get("Minutes"), errors="coerce")
        mid_f = pd.to_datetime(fant["DateStart"]) + (
            pd.to_datetime(fant["DateEnd"]) - pd.to_datetime(fant["DateStart"])
        ) / 2
        fant["DateMid"] = mid_f.dt.date

    // szybkie info o liczbie meczów / minut
    if not moto.empty:
        mecze_m = pd.to_numeric(moto.get("NumberOfGames"), errors="coerce").sum()
        min_m   = pd.to_numeric(moto.get("Minutes"),       errors="coerce").sum()
        st.caption(f"MOTORYKA: mecze = {int(mecze_m) if pd.notna(mecze_m) else 0}, minuty = {int(min_m) if pd.notna(min_m) else 0}")
    if not fant.empty:
        mecze_f = pd.to_numeric(fant.get("NumberOfGames"), errors="coerce").sum()
        min_f   = pd.to_numeric(fant.get("Minutes"),       errors="coerce").sum()
        st.caption(f"FANTASYPASY: mecze = {int(mecze_f) if pd.notna(mecze_f) else 0}, minuty = {int(min_f) if pd.notna(min_f) else 0}")

    // --- zakładki (po przygotowaniu danych!) ---
    tabs_prof = st.tabs(["Motoryka", "Indeks", "FANTASYPASY", "Tabele i eksport"])

    // ======================== 1) MOTORYKA ========================
    with tabs_prof[0]:
        if moto.empty:
            st.info("Brak danych motorycznych.")
        else:
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
                )
                chart_m = (
                    alt.Chart(plot_m)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("DateMid:T", title="Data"),
                        y=alt.Y("Value:Q", title="Wartość"),
                        color=alt.Color("Metric:N", title="Metryka"),
                        tooltip=["DateMid:T", "Metric:N", "Value:Q"]
                    )
                    .properties(height=420)
                )
                st.altair_chart(chart_m, use_container_width=True)

            st.markdown("**Tabela: bazowe wartości motoryczne (per minuta)**")
            show_cols = ["DateMid", "Minutes"] + [m + "_per_min" for m in per_min_cols]
            show_cols = [c for c in show_cols if c in moto.columns]
            st.dataframe(moto[show_cols].sort_values("DateMid", ascending=True), use_container_width=True)

            // TOP / BOTTOM 5
            st.markdown("---")
            st.subheader("Najlepsze i najsłabsze występy – per minuta")
            cL, cR = st.columns(2)
            best_metric = cL.selectbox(
                "Najlepsze/najsłabsze – metryka",
                [m + "_per_min" for m in per_min_cols],
                index=0,
                key="prof_all_best_moto"
            )
            if best_metric:
                view = moto[["DateMid", best_metric]].rename(columns={best_metric: "Value"}).dropna(subset=["Value"])
                cL.write("Top 5")
                cL.dataframe(view.sort_values("Value", ascending=False).head(5), use_container_width=True)
                cR.write("Bottom 5")
                cR.dataframe(view.sort_values("Value", ascending=True).head(5), use_container_width=True)

            // C1 referencja per minuta
            st.markdown("---")
            st.subheader("Zawodnik vs C1 – per minuta (mean/median, per data i globalnie)")

            df_all_for_ref = load_motoryka_all(None, None, None).copy()
            if df_all_for_ref.empty or df_all_for_ref[df_all_for_ref["Team"] == "C1"].empty:
                st.info("Brak danych referencyjnych C1.")
            else:
                df_all_for_ref["Minutes"] = pd.to_numeric(df_all_for_ref["Minutes"], errors="coerce").replace(0, np.nan)
                for m in per_min_cols:
                    if m in df_all_for_ref.columns:
                        df_all_for_ref[m + "_per_min"] = pd.to_numeric(df_all_for_ref[m], errors="coerce") / df_all_for_ref["Minutes"]
                    else:
                        df_all_for_ref[m + "_per_min"] = np.nan

                mid_ref = pd.to_datetime(df_all_for_ref["DateStart"]) + (
                    pd.to_datetime(df_all_for_ref["DateEnd"]) - pd.to_datetime(df_all_for_ref["DateStart"])
                ) / 2
                df_all_for_ref["DateMid"] = mid_ref.dt.date

                c1ref = df_all_for_ref[df_all_for_ref["Team"] == "C1"].copy()
                per_min_cols_pm = [c + "_per_min" for c in per_min_cols]

                c1_per_date = c1ref.groupby("DateMid")[per_min_cols_pm].agg(["mean", "median"])
                st.dataframe(c1_per_date, use_container_width=True)

    // ======================== 2) INDEKS ========================
    with tabs_prof[1]:
        if moto.empty or moto["PlayerIntensityIndex"].dropna().empty:
            st.info("Brak danych indeksu.")
        else:
            metric_idx = st.radio(
                "Metryka indeksowa",
                ["PlayerIntensityIndex", "PII_vs_team_avg"],
                horizontal=True,
                key="prof_all_idx_metric"
            )

            // wykres
            src_line = moto[["DateMid", metric_idx]].rename(columns={metric_idx: "Value"}).dropna(subset=["Value"])
            chart_i = (
                alt.Chart(src_line)
                .mark_line(point=True)
                .encode(
                    x=alt.X("DateMid:T", title="Data"),
                    y=alt.Y("Value:Q", title=metric_idx),
                    tooltip=["DateMid:T", "Value:Q"]
                )
                .properties(height=360)
            )
            st.altair_chart(chart_i, use_container_width=True)

            // tabela wartości po dacie
            st.markdown("**Tabela: Indeks – wartości (po dacie)**")
            tbl_by_date = (
                moto[["DateMid", metric_idx]]
                .rename(columns={metric_idx: "Value"})
                .sort_values("DateMid", ascending=True)
            )
            st.dataframe(tbl_by_date, use_container_width=True)

            // podsumowanie
            st.markdown("**Tabela: Podsumowanie indeksu**")
            summary_tbl = tbl_by_date["Value"].agg(["mean", "max", "min"]).to_frame(name="Value")
            st.dataframe(summary_tbl, use_container_width=True)

            // składowe: PII zawodnika + średnia zespołu + różnica
            all_m = load_motoryka_all(None, None, None).copy()
            if not all_m.empty:
                mid_all = pd.to_datetime(all_m["DateStart"]) + (
                    pd.to_datetime(all_m["DateEnd"]) - pd.to_datetime(all_m["DateStart"])
                ) / 2
                all_m["DateMid"] = mid_all.dt.date
                all_m["PlayerIntensityIndex"] = pd.to_numeric(all_m["PlayerIntensityIndex"], errors="coerce")

                team_mean = (
                    all_m.groupby(["Team", "DateMid"], as_index=False)["PlayerIntensityIndex"]
                    .mean()
                    .rename(columns={"PlayerIntensityIndex": "Team_PII_mean"})
                )
            else:
                team_mean = pd.DataFrame(columns=["Team", "DateMid", "Team_PII_mean"])

            comp = moto[["Team", "DateMid", "PlayerIntensityIndex", "PII_vs_team_avg"]].merge(
                team_mean, on=["Team", "DateMid"], how="left"
            )

            if metric_idx == "PII_vs_team_avg":
                comp["Diff"] = comp["PII_vs_team_avg"]
                diff_label = "Różnica (zawodnik – średnia zespołu)"
            else:
                comp["Diff"] = comp["PlayerIntensityIndex"] - comp["Team_PII_mean"]
                diff_label = "PII – średnia zespołu"

            comp_view = (
                comp[["DateMid", "Team", "PlayerIntensityIndex", "Team_PII_mean", "Diff"]]
                .sort_values("DateMid", ascending=True)
                .rename(columns={
                    "DateMid": "Data",
                    "Team": "Zespół (własny)",
                    "PlayerIntensityIndex": "PII zawodnika",
                    "Team_PII_mean": "PII średnia zespołu (data)",
                    "Diff": diff_label
                })
            )

            st.markdown("**Tabela: Składowe indeksu (wartość zawodnika + średnia zespołu + różnica)**")
            st.caption("Dla wybranej metryki pokazujemy PII zawodnika, średnią PII jego zespołu w danej dacie i różnicę.")
            st.dataframe(comp_view, use_container_width=True)

            // ======================== KLASTERYZACJA VS WSZYSTKIE ZESPOŁY ========================
            st.markdown("---")
            st.subheader("Klasteryzacja: do którego zespołu najbardziej pasuje zawodnik (PII)")

            if not all_m.empty:
                team_day_mean = (
                    all_m
                    .groupby(["Team", "DateMid"], as_index=False)["PlayerIntensityIndex"]
                    .mean()
                    .rename(columns={"PlayerIntensityIndex": "Team_PII_mean"})
                )

                team_global_mean = (
                    all_m
                    .groupby("Team", as_index=False)["PlayerIntensityIndex"]
                    .mean()
                    .rename(columns={"PlayerIntensityIndex": "Team_PII_global_mean"})
                )
            else:
                team_day_mean = pd.DataFrame(columns=["Team", "DateMid", "Team_PII_mean"])
                team_global_mean = pd.DataFrame(columns=["Team", "Team_PII_global_mean"])

            // PII zawodnika per data
            player_idx = (
                moto[["DateMid", "PlayerIntensityIndex"]]
                .rename(columns={"PlayerIntensityIndex": "Player_PII"})
                .dropna(subset=["Player_PII"])
            )

            if player_idx.empty or team_day_mean.empty:
                st.info("Brak danych do klasteryzacji (zawodnik lub zespoły bez PII).")
            else:
                // każda data zawodnika × wszystkie zespoły, które mają wtedy PII
                merged = player_idx.merge(team_day_mean, on="DateMid", how="left")

                merged["Diff_abs"] = (merged["Player_PII"] - merged["Team_PII_mean"]).abs()

                merged_valid = merged.dropna(subset=["Team_PII_mean"])

                if merged_valid.empty:
                    st.info("Zawodnik ma wpisy, ale w tych dniach brak danych motorycznych zespołów (nie było meczu).")
                else:
                    best = (
                        merged_valid
                        .sort_values(["DateMid", "Diff_abs"])
                        .groupby("DateMid", as_index=False)
                        .first()
                    )

                    best = best.merge(team_global_mean, on="Team", how="left")

                    final = player_idx.merge(best, on="DateMid", how="left", suffixes=("", "_cluster"))

                    final["Diff_vs_global"] = final["Player_PII"] - final["Team_PII_global_mean"]

                    comments = []
                    for _, r in final.iterrows():
                        if pd.isna(r.get("Team")):
                            comments.append("Brak danych zespołowych w tym dniu – prawdopodobnie nie było meczu.")
                        else:
                            try:
                                comments.append(
                                    f"Najbliżej {r['Team']} (dzień: {r['Player_PII']:.1f} vs {r['Team_PII_mean']:.1f}; "
                                    f"globalnie: {r['Team_PII_global_mean']:.1f})."
                                )
                            except Exception:
                                comments.append(f"Najbliżej {r['Team']} (brak pełnych danych do opisu).")
                    final["Komentarz"] = comments

                    cluster_view = final[[ 
                        "DateMid",
                        "Team",
                        "Player_PII",
                        "Team_PII_mean",
                        "Team_PII_global_mean",
                        "Diff_abs",
                        "Diff_vs_global",
                        "Komentarz"
                    ]].rename(columns={
                        "DateMid": "Data",
                        "Team": "Zespół (klaster)",
                        "Player_PII": "PII zawodnika",
                        "Team_PII_mean": "PII zespołu (dany dzień)",
                        "Team_PII_global_mean": "PII zespołu (średnia globalna)",
                        "Diff_abs": "|różnica| (dzień)",
                        "Diff_vs_global": "różnica vs średnia globalna zespołu"
                    })

                    num_cols = [
                        "PII zawodnika",
                        "PII zespołu (dany dzień)",
                        "PII zespołu (średnia globalna)",
                        "|różnica| (dzień)",
                        "różnica vs średnia globalna zespołu"
                    ]
                    for c in num_cols:
                        cluster_view[c] = pd.to_numeric(cluster_view[c], errors="coerce").round(2)

                    st.markdown("**Tabela: Klasteryzacja zawodnika po PII względem wszystkich zespołów**")
                    st.caption(
                        "Dla każdej daty indeksu zawodnika przypisujemy go do tego zespołu, "
                        "którego średni PII w tym dniu jest najbliższy. "
                        "Jeśli w danym dniu żaden zespół nie ma danych, pojawia się informacja, że nie było meczu."
                    )
                    st.dataframe(cluster_view.sort_values("Data", ascending=True), use_container_width=True)

    // ======================== 3) FANTASYPASY ========================
    with tabs_prof[2]:
        fant_metrics = [
            "PktOff", "PktDef", "Goal", "Assist", "ChanceAssist", "KeyPass",
            "KeyLoss", "DuelLossInBox", "MissBlockShot", "Finalization",
            "KeyIndividualAction", "KeyRecover",
            "DuelWinInBox", "BlockShot"
        ]
        if fant.empty:
            st.info("Brak danych FANTASYPASY.")
        else:
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
                        tooltip=["DateMid:T", "Metric:N", alt.Tooltip("Value:Q", format=".2f")]
                    )
                    .properties(height=360)
                )
                st.altair_chart(chart_f, use_container_width=True)

            cL2, cR2 = st.columns(2)
            best_f = cL2.selectbox(
                "Najlepsze/najsłabsze – metryka",
                fant_metrics,
                index=0,
                key="prof_all_best_fant"
            )
            if best_f:
                vf = fant[["DateMid", best_f]].rename(columns={best_f: "Value"}).dropna(subset=["Value"])
                cL2.write("Top 5")
                cL2.dataframe(vf.sort_values("Value", ascending=False).head(5), use_container_width=True)
                cR2.write("Bottom 5")
                cR2.dataframe(vf.sort_values("Value", ascending=True).head(5), use_container_width=True)

    // ======================== 4) Tabele i eksport ========================
    with tabs_prof[3]:
        st.markdown("### Surowe dane – motoryka")
        if moto.empty:
            st.info("Brak danych motorycznych.")
        else:
            st.dataframe(moto.sort_values("DateMid", ascending=True), use_container_width=True)

        st.markdown("### Surowe dane – FANTASYPASY")
        if fant.empty:
            st.info("Brak danych FANTASYPASY.")
        else:
            st.dataframe(fant.sort_values("DateMid", ascending=True), use_container_width=True)

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

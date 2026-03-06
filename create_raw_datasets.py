import pybaseball
from pybaseball import statcast
from pybaseball import chadwick_register
from pybaseball import pitching_stats
import duckdb
import pandas as pd

#############################################
# 1. POPULATE BASEBALL REFERENCE DATA
#############################################
def populate_raw_baseballref():

    START_YEAR = 2021
    END_YEAR = 2025

    con = duckdb.connect("pitch_design.db")

    print("Pulling FanGraphs pitching data...")

    pitcher_season = pitching_stats(
        START_YEAR,
        END_YEAR,
        qual=0
    )

    #############################################
    # SELECT COLUMNS
    #############################################

    pitcher_season = pitcher_season.rename(columns={
        "Name": "player_name",
        "Season": "game_year",
        "Team": "team",
        "IP": "innings_pitched",
        "K%": "K_percent",
        "BB%": "BB_percent",
        "HR": "home_runs",
        "SO": "strikeouts",
        "BB": "walks",
        "HBP": "hbp"
    })

    numeric_cols = [
        "innings_pitched","ERA","FIP","xFIP","WAR",
        "K_percent","BB_percent",
        "home_runs","strikeouts","walks","hbp"
    ]

    for col in numeric_cols:
        if col in pitcher_season.columns:
            pitcher_season[col] = pd.to_numeric(
                pitcher_season[col],
                errors="coerce"
            )

    #############################################
    # CREATE ERA+ AND FIP+
    #############################################

    # League ERA by season (IP-weighted)
    league_era = (
        pitcher_season
        .groupby("game_year")
        .apply(lambda x:
            (x["ERA"] * x["innings_pitched"]).sum()
            / x["innings_pitched"].sum()
        )
        .reset_index(name="league_ERA")
    )

    pitcher_season = pitcher_season.merge(
        league_era,
        on="game_year",
        how="left"
    )

    pitcher_season["ERA_plus"] = (
        100 * pitcher_season["league_ERA"]
        / pitcher_season["ERA"]
    )

    # League FIP by season (IP-weighted)
    league_fip = (
        pitcher_season
        .groupby("game_year")
        .apply(lambda x:
            (x["FIP"] * x["innings_pitched"]).sum()
            / x["innings_pitched"].sum()
        )
        .reset_index(name="league_FIP")
    )

    pitcher_season = pitcher_season.merge(
        league_fip,
        on="game_year",
        how="left"
    )

    pitcher_season["FIP_plus"] = (
        100 * pitcher_season["league_FIP"]
        / pitcher_season["FIP"]
    )

    #############################################
    # SAVE TO DUCKDB
    #############################################

    con.register("season_df", pitcher_season)

    con.execute("""
    CREATE OR REPLACE TABLE pitcher_season_stats AS
    SELECT *
    FROM season_df
    """)

    con.close()

#############################################
# 2. POPULATE STATCAST DATA
#############################################
def populate_raw_statcast():
    con = duckdb.connect("pitch_design.db")
    data = statcast(start_dt="2021-03-01", end_dt="2025-12-31")
    pybaseball.cache.enable()

    #Create Raw Table
    con.execute("CREATE TABLE raw_statcast AS SELECT * FROM data")

    #Create Whiff Flag
    con.execute("""ALTER TABLE raw_statcast ADD COLUMN whiff_flag INTEGER;
    UPDATE raw_statcast
    SET whiff_flag = CASE
        WHEN description IN ('swinging_strike', 'swinging_strike_blocked')
        THEN 1
        ELSE 0
    END;""")

    #Create Swing Flag
    con.execute("""ALTER TABLE raw_statcast ADD COLUMN swing_flag INTEGER;
    UPDATE raw_statcast
    SET swing_flag = CASE
        WHEN description IN (
            'swinging_strike',
            'swinging_strike_blocked',
            'foul',
            'foul_tip',
            'hit_into_play',
            'hit_into_play_score',
            'hit_into_play_no_out'
        )
        THEN 1
        ELSE 0
    END;""")

    #Create Strike Flag
    con.execute("""ALTER TABLE raw_statcast ADD COLUMN called_strike_flag INTEGER;
    UPDATE raw_statcast
    SET called_strike_flag = CASE
        WHEN description = 'called_strike'
        THEN 1
        ELSE 0
    END;""")

    #Create Hard Hit Flag
    con.execute("""ALTER TABLE raw_statcast ADD COLUMN hard_hit_flag INTEGER;
    UPDATE raw_statcast
    SET hard_hit_flag = CASE
        WHEN launch_speed >= 95 THEN 1
        ELSE 0
    END;""")

    #Create Barrel Flag
    con.execute("""ALTER TABLE raw_statcast ADD COLUMN barrel_flag INTEGER;
    UPDATE raw_statcast
    SET barrel_flag = CASE
        WHEN launch_speed >= 98
            AND launch_angle BETWEEN 26 AND 30
        THEN 1
        ELSE 0
    END;""")

    #Create Primary Key
    con.execute("""ALTER TABLE raw_statcast ADD COLUMN pitch_pk VARCHAR;
    UPDATE raw_statcast
    SET pitch_pk =
        CAST(pitcher AS VARCHAR) || '_' ||
        CAST(game_pk AS VARCHAR) || '_' ||
        CAST(at_bat_number AS VARCHAR) || '_' ||
        CAST(pitch_number AS VARCHAR);""")

    con.close()

#############################################
# 3. POPULATE PLAYER LOOKUP TABLE
#############################################
def populate_player_lookup():

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 1. PULL CHADWICK REGISTER
    #############################################

    print("Pulling Chadwick player register...")

    id_map = chadwick_register()

    print(f"Total records pulled: {len(id_map)}")

    #############################################
    # 2. SELECT COLUMNS
    #############################################

    id_map = id_map[[
        "key_mlbam",
        "key_fangraphs",
        "key_bbref",
        "key_retro",
        "name_first",
        "name_last",
        "mlb_played_first",
        "mlb_played_last"
    ]].copy()

    # Create Full Name column (first + last)
    id_map["full_name"] = (
        id_map["name_first"].fillna("").str.strip() +
        " " +
        id_map["name_last"].fillna("").str.strip()
    ).str.strip()


    # Drop duplicates on MLBAM ID (keep most recent MLB season)
    id_map = (
        id_map
        .sort_values("mlb_played_last", ascending=False)
        .drop_duplicates(subset=["key_mlbam"])
    )

    #############################################
    # 3. SAVE TO DUCKDB
    #############################################

    con.register("id_map_df", id_map)

    con.execute("""
    CREATE OR REPLACE TABLE player_id_map AS
    SELECT *
    FROM id_map_df
    """)

    con.close()

#############################################
# MAIN FUNCTION
#############################################
if __name__ == '__main__':
    populate_raw_baseballref()
    populate_raw_statcast()
    populate_player_lookup()
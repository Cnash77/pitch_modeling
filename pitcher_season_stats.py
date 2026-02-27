import duckdb
import pandas as pd
import numpy as np

#############################################
# 1. CONNECT
#############################################

con = duckdb.connect("pitch_design.db")

#############################################
# 2. LOAD RAW DATA
#############################################

pitch_data = con.execute("""
SELECT *
FROM raw_statcast
""").df()

#############################################
# 3. BUILD PLATE APPEARANCE TABLE
#############################################

# Keep only rows that end a PA
pa_data = pitch_data[pitch_data["events"].notna()].copy()

# Create unique PA ID
pa_data["pa_id"] = (
    pa_data["game_pk"].astype(str) + "_" +
    pa_data["at_bat_number"].astype(str)
)

#############################################
# 4. DEFINE STRIKEOUT EVENTS
#############################################

strikeout_events = [
    "strikeout",
    "strikeout_double_play"
]

pa_data["is_strikeout"] = (
    pa_data["events"].isin(strikeout_events).astype(int)
)

#############################################
# 5. AGGREGATE BY PITCHER + YEAR
#############################################

pitcher_season = (
    pa_data
    .groupby(["pitcher","game_year"])
    .agg(
        batters_faced=("pa_id","nunique"),
        strikeouts=("is_strikeout","sum")
    )
    .reset_index()
)

#############################################
# 6. CALCULATE K%
#############################################

pitcher_season["K_percent"] = (
    pitcher_season["strikeouts"] /
    pitcher_season["batters_faced"]
)

#############################################
# 7. FILTER SMALL SAMPLES (STABILITY)
#############################################

pitcher_season = pitcher_season[
    pitcher_season["batters_faced"] >= 100
]

#############################################
# 8. SAVE TO DUCKDB
#############################################

con.register("season_df", pitcher_season)

con.execute("""
CREATE OR REPLACE TABLE pitcher_season_stats AS
SELECT p.player_name,
       c.*
FROM season_df c
INNER JOIN pitchers p
ON c.pitcher = p.pitcher
""")

con.execute("""
COPY (
    SELECT *
    FROM pitcher_season_stats
)
TO 'pitcher_season_stats.csv' (HEADER, DELIMITER ',');
""")

print("\nCreated pitcher_season_stats (robust PA method)\n")
print(pitcher_season.sort_values("K_percent", ascending=False).head())

con.close()
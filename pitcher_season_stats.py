import duckdb
import pandas as pd
from pybaseball import pitching_stats

#############################################
# SETTINGS
#############################################

START_YEAR = 2021
END_YEAR = 2025

#############################################
# CONNECT
#############################################

con = duckdb.connect("pitch_design.db")

#############################################
# PULL OFFICIAL DATA
#############################################

print("Pulling FanGraphs pitching data...")

pitcher_season = pitching_stats(
    START_YEAR,
    END_YEAR,
    qual=0
)

#############################################
# CLEAN / STANDARDIZE
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

# Ensure numeric
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
# OPTIONAL SAMPLE FILTER
#############################################

pitcher_season = pitcher_season[
    pitcher_season["innings_pitched"] >= 30
]

#############################################
# SAVE TO DUCKDB
#############################################

con.register("season_df", pitcher_season)

con.execute("""
CREATE OR REPLACE TABLE pitcher_season_stats AS
SELECT *
FROM season_df
""")

pitcher_season.to_csv("pitcher_season_stats.csv", index=False)

print("\nOfficial pitcher_season_stats created\n")
print(pitcher_season.head())

con.close()
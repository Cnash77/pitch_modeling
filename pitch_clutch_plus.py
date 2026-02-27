import pandas as pd
import numpy as np
import duckdb
from sklearn.preprocessing import StandardScaler

#############################################
# 1. LOAD DATA
#############################################

con = duckdb.connect("pitch_design.db")
data = con.sql("SELECT * FROM raw_statcast").df()

#############################################
# 2. BUILD GAME STATE
#############################################

# Score differential from fielding team perspective
data["score_diff"] = data["fld_score"] - data["bat_score"]

# Base state encoding (0–7)
data["base_state"] = (
    data["on_1b"].notna().astype(int) * 1 +
    data["on_2b"].notna().astype(int) * 2 +
    data["on_3b"].notna().astype(int) * 4
)

# Half inning flag
data["is_top"] = (data["inning_topbot"] == "Top").astype(int)

#############################################
# 3. DETERMINE GAME WINNER
#############################################

# Final score per game
final_scores = (
    data.groupby("game_pk")
    .agg(
        final_home=("home_score","max"),
        final_away=("away_score","max")
    )
    .reset_index()
)

data = data.merge(final_scores, on="game_pk", how="left")

# Did fielding team win?
data["fld_win"] = np.where(
    (data["is_top"] == 1) & (data["final_home"] > data["final_away"]),
    1,
    np.where(
        (data["is_top"] == 0) & (data["final_away"] > data["final_home"]),
        1,
        0
    )
)

#############################################
# 4. BUILD HISTORICAL WIN EXPECTANCY TABLE
#############################################

we_table = (
    data.groupby([
        "inning",
        "is_top",
        "score_diff",
        "outs_when_up",
        "base_state"
    ])
    .agg(
        win_prob=("fld_win","mean"),
        samples=("fld_win","count")
    )
    .reset_index()
)

# Optional: remove low-sample states
we_table = we_table[we_table["samples"] >= 50]

#############################################
# 5. MERGE WIN PROBABILITY BEFORE PITCH
#############################################

data = data.merge(
    we_table,
    on=["inning","is_top","score_diff","outs_when_up","base_state"],
    how="left"
)

data.rename(columns={"win_prob":"wp_before"}, inplace=True)

#############################################
# 6. COMPUTE POST-PITCH STATE
#############################################

# Update score diff after pitch
data["runs_scored"] = data["bat_score"].diff().fillna(0)
data["score_diff_after"] = data["score_diff"] - data["runs_scored"]

# For simplicity we keep same outs/base state after pitch
# (You can expand this with true post-pitch state logic)

data = data.merge(
    we_table,
    left_on=["inning","is_top","score_diff_after","outs_when_up","base_state"],
    right_on=["inning","is_top","score_diff","outs_when_up","base_state"],
    how="left",
    suffixes=("","_after")
)

data.rename(columns={"win_prob":"wp_after"}, inplace=True)

#############################################
# 7. CALCULATE WPA
#############################################

data["actual_wpa"] = data["wp_after"] - data["wp_before"]

#############################################
# 8. CALCULATE RUN VALUE RESIDUAL
#############################################

RUNS_PER_WOBA = 1.15

data["actual_rv"] = (
    data["estimated_woba_using_speedangle"] * RUNS_PER_WOBA
)

league_avg_woba = data["estimated_woba_using_speedangle"].mean()
data["expected_rv"] = league_avg_woba * RUNS_PER_WOBA

data["rv_residual"] = data["expected_rv"] - data["actual_rv"]

#############################################
# 9. DEFINE HIGH LEVERAGE
#############################################

data["leverage_index"] = data["actual_wpa"].abs()
cutoff = data["leverage_index"].quantile(0.75)

data["high_leverage"] = data["leverage_index"] >= cutoff

#############################################
# 10. AGGREGATE TO PITCHER-YEAR
#############################################

hl = data[data["high_leverage"] == True]

clutch = (
    hl.groupby(["pitcher","game_year"])
    .agg(
        hl_rv_residual=("rv_residual","sum"),
        hl_wpa=("actual_wpa","sum"),
        hl_events=("pitcher","count")
    )
    .reset_index()
)

#############################################
# 11. STANDARDIZE
#############################################

scaler = StandardScaler()

clutch[["rv_z","wpa_z"]] = scaler.fit_transform(
    clutch[["hl_rv_residual","hl_wpa"]]
)

clutch["Clutch_raw"] = clutch["rv_z"] + clutch["wpa_z"]

mean = clutch["Clutch_raw"].mean()
std  = clutch["Clutch_raw"].std()

clutch["Clutch_Score"] = 100 + 10 * (
    (clutch["Clutch_raw"] - mean) / std
)

#############################################
# 12. SAVE
#############################################

con.register("clutch_df", clutch)

con.execute("""
CREATE OR REPLACE TABLE pitcher_clutch AS
SELECT p.player_name,
       c.*    
FROM clutch_df c
INNER JOIN pitchers p 
ON c.pitcher = p.pitcher
""")

con.execute("""
    COPY (
        SELECT *
        FROM pitcher_clutch
    )
    TO 'pitcher_clutch.csv' (HEADER, DELIMITER ',');
""")

print("\nHistorical Clutch Model Complete\n")
print(clutch.sort_values("Clutch_Score", ascending=False).head())

con.close()
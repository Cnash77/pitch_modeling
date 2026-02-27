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
# 2. BUILD GAME STATE FEATURES (FULL DATA)
#############################################

# Base state (0–7 encoding)
data["base_state"] = (
    data["on_1b"].notna().astype(int) * 1 +
    data["on_2b"].notna().astype(int) * 2 +
    data["on_3b"].notna().astype(int) * 4
)

# Runs scored on play
data["runs_scored"] = (
    data.groupby("game_pk")["bat_score"]
    .diff()
    .fillna(0)
)

# Inning total runs
inning_totals = (
    data.groupby(["game_pk","inning"])
    .agg(total_runs=("runs_scored","sum"))
    .reset_index()
)

data = data.merge(inning_totals, on=["game_pk","inning"], how="left")

# Runs remaining in inning
data["runs_remaining"] = (
    data["total_runs"] -
    data.groupby(["game_pk","inning"])["runs_scored"].cumsum()
)

#############################################
# 3. BUILD RUN EXPECTANCY MATRIX
#############################################

re_matrix = (
    data.groupby(["outs_when_up","base_state"])
    .agg(
        run_expectancy=("runs_remaining","mean"),
        samples=("runs_remaining","count")
    )
    .reset_index()
)

re_matrix = re_matrix[re_matrix["samples"] >= 50]

#############################################
# 4. BUILD WIN EXPECTANCY TABLE
#############################################

data["score_diff"] = data["fld_score"] - data["bat_score"]
data["is_top"] = (data["inning_topbot"] == "Top").astype(int)

final_scores = (
    data.groupby("game_pk")
    .agg(
        final_home=("home_score","max"),
        final_away=("away_score","max")
    )
    .reset_index()
)

data = data.merge(final_scores, on="game_pk", how="left")

data["fld_win"] = np.where(
    (data["is_top"] == 1) & (data["final_home"] > data["final_away"]),
    1,
    np.where(
        (data["is_top"] == 0) & (data["final_away"] > data["final_home"]),
        1,
        0
    )
)

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

we_table = we_table[we_table["samples"] >= 50]

#############################################
# 5. IDENTIFY RELIEVERS (<20% STARTS)
#############################################

# Create appearance_id first
data["appearance_id"] = (
    data["game_pk"].astype(str) + "_" +
    data["pitcher"].astype(str)
)

#A. IDENTIFY STARTS (1ST INNING APPEARANCE)
#A start = pitcher throws in inning 1
first_inning_apps = (
    data[data["inning"] == 1]
    .groupby(["pitcher","game_year","game_pk"])
    .size()
    .reset_index(name="first_inning_flag")
)

# Count starts per pitcher-season
starts_per_year = (
    first_inning_apps
    .groupby(["pitcher","game_year"])
    .agg(games_started=("first_inning_flag","count"))
    .reset_index()
)

# B. COUNT TOTAL APPEARANCES
appearances_per_year = (
    data.groupby(["pitcher","game_year"])
    .agg(total_appearances=("game_pk","nunique"))
    .reset_index()
)

#C. MERGE + CALCULATE START %
usage = appearances_per_year.merge(
    starts_per_year,
    on=["pitcher","game_year"],
    how="left"
)

usage["games_started"] = usage["games_started"].fillna(0)

usage["start_pct"] = (
    usage["games_started"] /
    usage["total_appearances"]
)

#D. DEFINE RELIEVERS
relievers = usage[
    usage["start_pct"] < 0.20
][["pitcher","game_year"]]

#E.FILTER DATASET
relief = data.merge(
    relievers,
    on=["pitcher","game_year"],
    how="inner"
)

relief["appearance_id"] = (
    relief["game_pk"].astype(str) + "_" +
    relief["pitcher"].astype(str)
)

#############################################
# 6. ENTRY STATE PER APPEARANCE
#############################################

entry = (
    relief.sort_values("pitch_number")
    .groupby("appearance_id")
    .first()
    .reset_index()
)

entry = entry.merge(
    re_matrix,
    on=["outs_when_up","base_state"],
    how="left"
)

entry.rename(columns={"run_expectancy":"re_at_entry"}, inplace=True)

entry = entry.merge(
    we_table,
    on=["inning","is_top","score_diff","outs_when_up","base_state"],
    how="left"
)

entry.rename(columns={"win_prob":"wp_at_entry"}, inplace=True)

#############################################
# 7. LEVERAGE INDEX
#############################################

entry["leverage_index"] = abs(entry["wp_at_entry"] - 0.5)
entry["leverage_index"] /= entry["leverage_index"].mean()

#############################################
# 8. ACTUAL RUNS ALLOWED
#############################################

runs_allowed = (
    relief.groupby("appearance_id")
    .agg(actual_runs=("runs_scored","sum"))
    .reset_index()
)

entry = entry.merge(runs_allowed, on="appearance_id", how="left")

#############################################
# 9. INHERITED VALUE
#############################################

entry["inherited_value"] = (
    entry["re_at_entry"] -
    entry["actual_runs"]
)

entry["leveraged_inherited_value"] = (
    entry["inherited_value"] *
    entry["leverage_index"]
)

#############################################
# 10. PITCH-LEVEL RUN VALUE RESIDUAL
#############################################

RUNS_PER_WOBA = 1.15

relief["actual_rv"] = (
    relief["estimated_woba_using_speedangle"] * RUNS_PER_WOBA
)

league_avg_woba = relief["estimated_woba_using_speedangle"].mean()

relief["expected_rv"] = league_avg_woba * RUNS_PER_WOBA

relief["rv_residual"] = (
    relief["expected_rv"] -
    relief["actual_rv"]
)

pitch_level_summary = (
    relief.groupby(["pitcher","game_year"])
    .agg(
        pitch_rv_value=("rv_residual","sum"),
        relief_pitches=("pitcher","count")
    )
    .reset_index()
)

#############################################
# 11. AGGREGATE INHERITED VALUE
#############################################

inherited_summary = (
    entry.groupby(["pitcher","game_year"])
    .agg(
        inherited_value=("leveraged_inherited_value","sum"),
        relief_apps=("appearance_id","count")
    )
    .reset_index()
)

#############################################
# 12. MERGE COMPONENTS
#############################################

rrp = pitch_level_summary.merge(
    inherited_summary,
    on=["pitcher","game_year"],
    how="left"
)

rrp["total_relief_value"] = (
    rrp["pitch_rv_value"] +
    rrp["inherited_value"]
)

#############################################
# 13. STANDARDIZE TO RRP+
#############################################

scaler = StandardScaler()

rrp["z_score"] = scaler.fit_transform(
    rrp[["total_relief_value"]]
)

rrp["RRP_plus"] = 100 + 10 * rrp["z_score"]

#############################################
# 14. MINIMUM SAMPLE FILTER
#############################################

rrp = rrp[rrp["relief_pitches"] >= 300]

#############################################
# 15. SAVE OUTPUT
#############################################

con.register("rrp_df", rrp)

con.execute("""
CREATE OR REPLACE TABLE relief_run_prevention_plus AS
SELECT p.player_name,
       c.*
FROM rrp_df c
INNER JOIN pitchers p
ON c.pitcher = p.pitcher
""")

con.execute("""
COPY (
    SELECT *
    FROM relief_run_prevention_plus
)
TO 'relief_run_prevention_plus.csv'
(HEADER, DELIMITER ',');
""")

print("\nRelief Run Prevention+ (Final Model) Complete\n")
print(rrp.sort_values("RRP_plus", ascending=False).head())

con.close()
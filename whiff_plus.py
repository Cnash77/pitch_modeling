import duckdb
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# -----------------------------------
# Connect + Load Data
# -----------------------------------
con = duckdb.connect("pitch_design.db")

data = con.execute("""
SELECT
    pitcher,
    game_year,
    pitch_type,
    stand,
    p_throws,
    balls,
    strikes,
    plate_x,
    plate_z,
    sz_top,
    sz_bot,
    release_speed,
    effective_speed,
    release_spin_rate,
    pfx_x,
    pfx_z,
    release_pos_x,
    release_pos_y,
    release_pos_z,
    arm_angle,
    release_extension,
    api_break_z_with_gravity,
    api_break_x_arm,
    api_break_x_batter_in,
    description
FROM raw_statcast
""").df()

# -----------------------------------
# Basic Cleaning
# -----------------------------------
data = data.dropna()

# -----------------------------------
# Create Whiff + Swing Flags
# -----------------------------------
data["is_whiff"] = data["description"].isin([
    "swinging_strike",
    "swinging_strike_blocked"
]).astype(int)

data["is_swing"] = data["description"].isin([
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play"
]).astype(int)

# Only model whiff conditional on swing
data = data[data["is_swing"] == 1].copy()

# -----------------------------------
# Feature Engineering
# -----------------------------------

# Handedness flags
data["is_RHP"] = (data["p_throws"] == "R").astype(int)
data["is_RHB"] = (data["stand"] == "R").astype(int)

# Normalize vertical location
data["zone_height_norm"] = (
    (data["plate_z"] - data["sz_bot"]) /
    (data["sz_top"] - data["sz_bot"])
)

features = [
    # Shape
    "release_speed",
    "effective_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "plate_z",
    "release_extension",
    "api_break_z_with_gravity",
    "api_break_x_arm",
    "api_break_x_batter_in",
    
    # Location
    "plate_x",
    "zone_height_norm",
    
    # Count context
    "balls",
    "strikes",
    
    # Handedness
    "is_RHP",
    "is_RHB"
]

# -----------------------------------
# Initialize Prediction Column
# -----------------------------------
data["raw_whiff"] = np.nan

# -----------------------------------
# Train Per Pitch Type (Time Split)
# -----------------------------------
pitch_types = data["pitch_type"].unique()

for pitch in pitch_types:

    pitch_df = data[data["pitch_type"] == pitch].copy()

    if len(pitch_df) < 100:
        print(f"Skipping {pitch} — insufficient swings")
        continue

    # Time-based split
    train = pitch_df[pitch_df["game_year"] < pitch_df["game_year"].max()]
    test  = pitch_df[pitch_df["game_year"] == pitch_df["game_year"].max()]

    if len(test) < 50:
        train = pitch_df.sample(frac=0.8, random_state=42)
        test  = pitch_df.drop(train.index)

    X_train = train[features]
    y_train = train["is_whiff"]

    X_test  = test[features]
    y_test  = test["is_whiff"]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"{pitch} AUC: {auc:.3f}")

    # Predict for ALL swings of this pitch type
    full_pred = model.predict_proba(pitch_df[features])[:, 1]

    data.loc[pitch_df.index, "raw_whiff"] = full_pred

# -----------------------------------
# Global Scaling (Critical Fix)
# -----------------------------------
league_mean = data["raw_whiff"].mean()
league_std  = data["raw_whiff"].std()

data["Whiff_plus"] = (
    100 + 10 * ((data["raw_whiff"] - league_mean) / league_std)
)

# -----------------------------------
# Aggregate to Pitcher + Pitch Type + Season
# -----------------------------------
pitcher_whiff = (
    data.groupby(["pitcher", "game_year", "pitch_type"])
        .agg(
            swings=("is_swing", "count"),
            avg_whiff_plus=("Whiff_plus", "mean"),
            avg_raw_whiff=("raw_whiff", "mean")
        )
        .reset_index()
)

# Optional minimum swings filter
pitcher_whiff = pitcher_whiff[pitcher_whiff["swings"] >= 50]

# -----------------------------------
# Save
# -----------------------------------
con.register("pitcher_whiff_df", pitcher_whiff)

con.execute("""
CREATE OR REPLACE TABLE pitcher_whiff AS
SELECT * FROM pitcher_whiff_df
""")

pitcher_whiff.to_csv("pitcher_whiff.csv", index=False)

print(pitcher_whiff.head())

con.close()
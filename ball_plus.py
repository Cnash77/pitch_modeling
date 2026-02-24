# ball_plus.py
import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#############################################
# 1. LOAD DATA
#############################################

con = duckdb.connect("pitch_design.db")
data = con.sql("SELECT * FROM raw_statcast").df()

#############################################
# 2. BUILD SWING + TAKE FLAGS
#############################################

swing_events = [
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_score",
    "hit_into_play_no_out"
]

data["is_swing"] = data["description"].isin(swing_events).astype(int)
data["is_take"] = 1 - data["is_swing"]

#############################################
# 3. BUILD BALL TARGET (TAKEN PITCHES ONLY)
#############################################

ball_events = [
    "ball",
    "blocked_ball",
    "intent_ball",
    "pitchout"
]

data["is_ball"] = data["description"].isin(ball_events).astype(int)

# Restrict to taken pitches only
data_model = data[data["is_take"] == 1].copy()

#############################################
# 4. FEATURE ENGINEERING
#############################################

# Normalize vertical zone location
data_model["zone_height_norm"] = (
    (data_model["plate_z"] - data_model["sz_bot"]) /
    (data_model["sz_top"] - data_model["sz_bot"])
)

# Handedness flags
data_model["is_RHP"] = (data_model["p_throws"] == "R").astype(int)
data_model["is_RHB"] = (data_model["stand"] == "R").astype(int)

# Categorical pitch type
data_model["pitch_type"] = data_model["pitch_type"].astype("category")

#############################################
# 5. FEATURE SET
#############################################

ball_features = [
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

data_model = data_model.dropna(subset=ball_features + ["is_ball"])

X = data_model[ball_features]
y = data_model["is_ball"]

#############################################
# 6. TRAIN / TEST SPLIT
#############################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#############################################
# 7. TRAIN MODEL
#############################################

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

#############################################
# 8. EVALUATE
#############################################

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)

print(f"Ball Model AUC: {auc:.4f}")

#############################################
# 9. GENERATE PITCH-LEVEL xBall
#############################################

data_model["xBall"] = model.predict_proba(X)[:, 1]

#############################################
# 10. SCALE TO BALL+
#############################################

league_mean = data_model["xBall"].mean()
league_std = data_model["xBall"].std()

# Inverted so higher = better (fewer balls)
data_model["Ball_plus"] = 100 - 10 * (
    (data_model["xBall"] - league_mean) / league_std
)

#############################################
# 11. AGGREGATE TO PITCHER / PITCH TYPE / SEASON
#############################################

grouped = (
    data_model
    .groupby(["pitcher", "game_year", "pitch_type"])
    .agg(
        pitches=("is_ball", "count"),
        avg_xBall=("xBall", "mean"),
        Ball_plus=("Ball_plus", "mean")
    )
    .reset_index()
)

#############################################
# 12. SAVE OUTPUT
#############################################

con.register("grouped_df", grouped)

con.execute("""
CREATE OR REPLACE TABLE ball_plus AS
SELECT * FROM grouped_df
""")

grouped.to_csv("ball_plus.csv", index=False)

print("Ball+ model complete. Output saved to ball_plus.csv")
con.close()
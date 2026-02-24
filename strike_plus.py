# strike_plus.py
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
# 2. BUILD STRIKE TARGET
#############################################

strike_events = [
    "called_strike",
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "hit_into_play",
    "hit_into_play_score",
    "hit_into_play_no_out"
]

data["is_strike"] = data["description"].isin(strike_events).astype(int)

#############################################
# 3. FEATURE ENGINEERING
#############################################

# Normalize vertical location relative to zone
data["zone_height_norm"] = (
    (data["plate_z"] - data["sz_bot"]) /
    (data["sz_top"] - data["sz_bot"])
)

# Handedness flags
data["is_RHP"] = (data["p_throws"] == "R").astype(int)
data["is_RHB"] = (data["stand"] == "R").astype(int)

# Convert pitch_type to categorical
data["pitch_type"] = data["pitch_type"].astype("category")

#############################################
# 4. FEATURE SET
#############################################

strike_features = [
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

data_model = data.dropna(subset=strike_features + ["is_strike"]).copy()

X = data_model[strike_features]
y = data_model["is_strike"]

#############################################
# 5. TRAIN / TEST SPLIT
#############################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#############################################
# 6. TRAIN MODEL
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
# 7. EVALUATE MODEL
#############################################

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)

print(f"Strike Model AUC: {auc:.4f}")

#############################################
# 8. GENERATE PITCH-LEVEL EXPECTED STRIKE
#############################################

data_model["xStrike"] = model.predict_proba(X)[:, 1]

#############################################
# 9. SCALE TO STRIKE+
#############################################

league_mean = data_model["xStrike"].mean()
league_std = data_model["xStrike"].std()

data_model["Strike_plus"] = 100 + 10 * (
    (data_model["xStrike"] - league_mean) / league_std
)

#############################################
# 10. AGGREGATE TO PITCHER / PITCH TYPE / SEASON
#############################################

grouped = (
    data_model
    .groupby(["pitcher", "game_year", "pitch_type"])
    .agg(
        pitches=("is_strike", "count"),
        avg_xStrike=("xStrike", "mean"),
        Strike_plus=("Strike_plus", "mean")
    )
    .reset_index()
)

#############################################
# 11. SAVE OUTPUT
#############################################

con.register("grouped_df", grouped)

con.execute("""
CREATE OR REPLACE TABLE strike_plus AS
SELECT * FROM grouped_df
""")

grouped.to_csv("strike_plus.csv", index=False)

print("Strike+ model complete. Output saved to strike_plus.csv")
con.close()
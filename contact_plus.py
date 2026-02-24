# contact_plus.py

import duckdb
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#############################################
# 1. LOAD DATA
#############################################

con = duckdb.connect("pitch_design.db")
data = con.sql("SELECT * FROM raw_statcast").df()

#############################################
# 2. DEFINE BALLS IN PLAY
#############################################

bip_events = [
    "hit_into_play",
    "hit_into_play_score",
    "hit_into_play_no_out"
]

data["is_bip"] = data["description"].isin(bip_events).astype(int)

# Restrict to contact events only
data_model = data[data["is_bip"] == 1].copy()

#############################################
# 3. TARGET VARIABLE
#############################################

# Must exist in dataset
TARGET = "estimated_woba_using_speedangle"

data_model = data_model.dropna(subset=[TARGET])

#############################################
# 4. FEATURE ENGINEERING
#############################################

# Normalize vertical location relative to strike zone
data_model["zone_height_norm"] = (
    (data_model["plate_z"] - data_model["sz_bot"]) /
    (data_model["sz_top"] - data_model["sz_bot"])
)

# Handedness flags
data_model["is_RHP"] = (data_model["p_throws"] == "R").astype(int)
data_model["is_RHB"] = (data_model["stand"] == "R").astype(int)

# Convert pitch_type to categorical
data_model["pitch_type"] = data_model["pitch_type"].astype("category")

#############################################
# 5. FEATURE SET
#############################################

contact_features = [
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

data_model = data_model.dropna(subset=contact_features)

X = data_model[contact_features]
y = data_model[TARGET]

#############################################
# 6. TRAIN / TEST SPLIT
#############################################

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#############################################
# 7. TRAIN MODEL
#############################################

model = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    enable_categorical=True,
    random_state=42
)

model.fit(X_train, y_train)

#############################################
# 8. EVALUATE MODEL
#############################################

preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"Contact Model RMSE: {rmse:.4f}")

#############################################
# 9. GENERATE PITCH-LEVEL xContact
#############################################

data_model["xContact"] = model.predict(X)

#############################################
# 10. SCALE TO CONTACT+
#############################################

league_mean = data_model["xContact"].mean()
league_std = data_model["xContact"].std()

# Invert so higher = better (lower damage allowed)
data_model["Contact_plus"] = 100 - 10 * (
    (data_model["xContact"] - league_mean) / league_std
)

#############################################
# 11. AGGREGATE TO PITCHER / PITCH TYPE / SEASON
#############################################

grouped = (
    data_model
    .groupby(["pitcher", "game_year", "pitch_type"])
    .agg(
        bip=("is_bip", "count"),
        avg_xContact=("xContact", "mean"),
        Contact_plus=("Contact_plus", "mean")
    )
    .reset_index()
)

#############################################
# 12. SAVE OUTPUT
#############################################

con.register("grouped_df", grouped)

con.execute("""
CREATE OR REPLACE TABLE contact_plus AS
SELECT * FROM grouped_df
""")

grouped.to_csv("contact_plus.csv", index=False)

print("Contact+ model complete. Output saved to contact_plus.csv")
con.close()
import duckdb
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

def calculate_whiff_plus():
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

    print("Whiff+ model complete.")

    con.close()

def calculate_contact_plus():
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

    print("Contact+ model complete.")
    con.close()

def calculate_strike_plus():
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

    print("Strike+ model complete.")
    con.close()

def calculate_ball_plus():
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

    print("Ball+ model complete.")
    con.close()

def calculate_swing_plus():
    #############################################
    # 1. LOAD DATA
    #############################################

    con = duckdb.connect("pitch_design.db")
    data = con.sql("SELECT * FROM raw_statcast").df()

    #############################################
    # 2. BUILD SWING TARGET
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

    #############################################
    # 3. FEATURE ENGINEERING
    #############################################

    # Normalize vertical location inside strike zone
    data["zone_height_norm"] = (
        (data["plate_z"] - data["sz_bot"]) /
        (data["sz_top"] - data["sz_bot"])
    )

    # Handedness flags
    data["is_RHP"] = (data["p_throws"] == "R").astype(int)
    data["is_RHB"] = (data["stand"] == "R").astype(int)

    # Keep original pitch_type but convert to category
    data["pitch_type"] = data["pitch_type"].astype("category")

    #############################################
    # 4. FEATURE SET
    #############################################

    swing_features = [
        "release_speed",
        "release_spin_rate",
        "pfx_x",
        "pfx_z",
        "plate_x",
        "zone_height_norm",
        "balls",
        "strikes",
        "is_RHP",
        "is_RHB",
        "pitch_type"
    ]

    data_model = data.dropna(subset=swing_features + ["is_swing"]).copy()

    X = data_model[swing_features]
    y = data_model["is_swing"]

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
    # 7. EVALUATE
    #############################################

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    print(f"Swing Model AUC: {auc:.4f}")

    #############################################
    # 8. GENERATE PITCH-LEVEL EXPECTED SWING
    #############################################

    data_model["xSwing"] = model.predict_proba(X)[:, 1]

    #############################################
    # 9. SCALE TO SWING+
    #############################################

    league_mean = data_model["xSwing"].mean()
    league_std = data_model["xSwing"].std()

    data_model["Swing_plus"] = 100 + 10 * (
        (data_model["xSwing"] - league_mean) / league_std
    )

    #############################################
    # 10. AGGREGATE TO PITCHER / PITCH TYPE / SEASON
    #############################################

    grouped = (
        data_model
        .groupby(["pitcher", "game_year", "pitch_type"])
        .agg(
            pitches=("is_swing", "count"),
            avg_xSwing=("xSwing", "mean"),
            Swing_plus=("Swing_plus", "mean")
        )
        .reset_index()
    )

    #############################################
    # 11. SAVE OUTPUT
    #############################################

    con.register("grouped_df", grouped)

    con.execute("""
    CREATE OR REPLACE TABLE swing_plus AS
    SELECT * FROM grouped_df
    """)

    print("Swing+ model complete. Output saved to swing_plus.csv")
    con.close()

def calculate_pitch_composite_score(): 
    # -----------------------------------
    # Connect
    # -----------------------------------
    con = duckdb.connect("pitch_design.db")

    # -----------------------------------
    # Load Whiff+
    # -----------------------------------
    whiff = con.execute("""
    SELECT
        pitcher,
        game_year,
        pitch_type,
        avg_whiff_plus AS Whiff_plus,
        swings
    FROM pitcher_whiff
    """).df()

    # -----------------------------------
    # Load Contact+
    # -----------------------------------
    contact = con.execute("""
    SELECT
        pitcher,
        game_year,
        pitch_type,
        Contact_plus,
        bip,
        avg_xContact
    FROM contact_plus
    """).df()

    # -----------------------------------
    # Load Swing+
    # -----------------------------------
    swing = con.execute("""
    SELECT
        pitcher,
        game_year,
        pitch_type,
        Swing_plus,
        pitches AS swing_pitches
    FROM swing_plus
    """).df()

    # -----------------------------------
    # Load Strike+
    # -----------------------------------
    strike = con.execute("""
    SELECT
        pitcher,
        game_year,
        pitch_type,
        Strike_plus,
        pitches AS strike_pitches
    FROM strike_plus
    """).df()

    # -----------------------------------
    # Load Ball+
    # -----------------------------------
    ball = con.execute("""
    SELECT
        pitcher,
        game_year,
        pitch_type,
        Ball_plus,
        pitches AS ball_pitches
    FROM ball_plus
    """).df()

    # -----------------------------------
    # Merge All Components
    # -----------------------------------
    df = (
        whiff
        .merge(contact, on=["pitcher","game_year","pitch_type"], how="inner")
        .merge(swing,   on=["pitcher","game_year","pitch_type"], how="inner")
        .merge(strike,  on=["pitcher","game_year","pitch_type"], how="inner")
        .merge(ball,    on=["pitcher","game_year","pitch_type"], how="inner")
    )

    # -----------------------------------
    # Stability Filters
    # -----------------------------------
    df = df[
        (df["swings"] >= 50) &
        (df["bip"] >= 25) &
        (df["swing_pitches"] >= 100) &
        (df["strike_pitches"] >= 100) &
        (df["ball_pitches"] >= 100)
    ]

    # -----------------------------------
    # DEFINE TARGET
    # -----------------------------------
    TARGET = "avg_xContact"

    df = df.dropna(subset=[TARGET]).copy()

    # -----------------------------------
    # STANDARDIZE FEATURES
    # -----------------------------------
    feature_cols = [
        "Whiff_plus",
        "Contact_plus",
        "Swing_plus",
        "Strike_plus",
        "Ball_plus"
    ]

    X = df[feature_cols]
    y = df[TARGET]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------------
    # FIT LINEAR REGRESSION
    # -----------------------------------
    reg = LinearRegression()
    reg.fit(X_scaled, y)

    betas = dict(zip(feature_cols, reg.coef_))

    # -----------------------------------
    # NORMALIZE WEIGHTS
    # -----------------------------------
    total = sum(abs(v) for v in betas.values())

    weights = {k: abs(v)/total for k, v in betas.items()}

    # -----------------------------------
    # BUILD WEIGHTED COMPOSITE
    # -----------------------------------
    df["Pitch_Grade_raw"] = sum(
        weights[col] * df[col] for col in feature_cols
    )

    # -----------------------------------
    # RESCALE TO 100 = LEAGUE AVG
    # -----------------------------------
    mean = df["Pitch_Grade_raw"].mean()
    std  = df["Pitch_Grade_raw"].std()

    df["Pitch_Grade"] = 100 + 10 * (
        (df["Pitch_Grade_raw"] - mean) / std
    )

    # -----------------------------------
    # Diagnostics Columns
    # -----------------------------------
    for col in feature_cols:
        df[f"{col}_Component"] = df[col]

    # -----------------------------------
    # Save to Database
    # -----------------------------------
    con.register("pitch_grade_df", df)

    con.execute("""
    CREATE OR REPLACE TABLE pitch_grade AS
    SELECT * FROM pitch_grade_df
    """)

    print("Pitch Grade model complete.")
    con.close()

def calculate_stuff_plus():
    #############################################
    # 1. LOAD DATA
    #############################################

    con = duckdb.connect("pitch_design.db")
    data = con.execute("SELECT * FROM raw_statcast").df()

    #############################################
    # 2. CREATE SWING DATASET (TRAINING ONLY)
    #############################################

    full_data = data.copy()

    swing_data = data[data["swing_flag"] == 1].copy()
    swing_data["is_whiff"] = swing_data["whiff_flag"]

    #############################################
    # 3. COMPUTE APPROACH ANGLES (FULL DATA)
    #############################################

    for df in [full_data, swing_data]:

        df["t_plate"] = -df["release_pos_y"] / df["vy0"]

        df["vz_plate"] = df["vz0"] + df["az"] * df["t_plate"]
        df["vy_plate"] = df["vy0"] + df["ay"] * df["t_plate"]
        df["vx_plate"] = df["vx0"] + df["ax"] * df["t_plate"]

        df["VAA"] = np.degrees(np.arctan(df["vz_plate"] / df["vy_plate"]))
        df["HAA"] = np.degrees(np.arctan(df["vx_plate"] / df["vy_plate"]))

        df = df[(df["VAA"] > -15) & (df["VAA"] < 5)]

    #############################################
    # 4. FEATURE SELECTION
    #############################################

    cols = [
        "pitcher","game_year","pitch_type",
        "stand","p_throws",
        "release_speed","release_spin_rate",
        "release_extension","release_pos_z",
        "arm_angle",
        "api_break_z_with_gravity",
        "api_break_x_batter_in",
        "plate_x","plate_z",
        "VAA","HAA"
    ]

    full_data = full_data[cols].dropna()
    swing_data = swing_data[cols + ["is_whiff"]].dropna()

    #############################################
    # 5. FEATURE ENGINEERING (BOTH DATASETS)
    #############################################

    def engineer_features(df):

        df["is_RHP"] = (df["p_throws"] == "R").astype(int)
        df["is_RHB"] = (df["stand"] == "R").astype(int)

        family_map = {
            "FF": "FB", "SI": "FB", "FC": "FB", "FA": "FB",
            "SL": "BB", "ST": "BB", "CU": "BB", "KC": "BB",
            "CH": "OS", "FS": "OS", "FO": "OS", "SV": "OS"
        }

        df["pitch_family"] = df["pitch_type"].map(family_map)
        df = df.dropna(subset=["pitch_family"])

        df["velo_diff"] = (
            df["release_speed"]
            - df.groupby(["pitcher","game_year"])["release_speed"].transform("mean")
        )

        df["ivb_diff"] = (
            df["api_break_z_with_gravity"]
            - df.groupby(["pitcher","game_year"])["api_break_z_with_gravity"].transform("mean")
        )

        df["hb_diff"] = (
            df["api_break_x_batter_in"]
            - df.groupby(["pitcher","game_year"])["api_break_x_batter_in"].transform("mean")
        )

        df["ivb_per_mph"] = df["api_break_z_with_gravity"] / df["release_speed"]
        df["hb_per_mph"] = df["api_break_x_batter_in"] / df["release_speed"]

        mean_release_height = df["release_pos_z"].mean()
        df["vaa_adj_height"] = (
            df["VAA"] - 0.6 * (df["release_pos_z"] - mean_release_height)
        )

        df["x_sweep_interaction"] = df["plate_x"] * df["api_break_x_batter_in"]
        df["z_vaa_interaction"] = df["plate_z"] * df["VAA"]

        return df

    full_data = engineer_features(full_data)
    swing_data = engineer_features(swing_data)

    #############################################
    # 6. MODEL FEATURES
    #############################################

    features = [
        "release_speed",
        "release_spin_rate",
        "release_extension",
        "arm_angle",
        "api_break_z_with_gravity",
        "api_break_x_batter_in",
        "VAA","HAA","vaa_adj_height",
        "velo_diff","ivb_diff","hb_diff",
        "ivb_per_mph","hb_per_mph",
        "plate_x","plate_z",
        "x_sweep_interaction","z_vaa_interaction",
        "is_RHP","is_RHB"
    ]

    #############################################
    # 7. TRAIN PER FAMILY (ON SWINGS ONLY)
    #############################################

    full_data["raw_stuff"] = np.nan
    full_data["Stuff_plus"] = np.nan

    family_models = {}

    for fam in swing_data["pitch_family"].unique():

        fam_train = swing_data[swing_data["pitch_family"] == fam].copy()

        if len(fam_train) < 200:
            continue

        X = fam_train[features]
        y = fam_train["is_whiff"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=3,
            reg_alpha=1,
            min_child_weight=5,
            eval_metric="logloss",
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, preds)

        # Score ALL pitches in that family
        fam_all = full_data[full_data["pitch_family"] == fam].copy()
        fam_all = fam_all.dropna(subset=features)

        raw_pred_all = model.predict_proba(fam_all[features])[:,1]

        mean = raw_pred_all.mean()
        std = raw_pred_all.std()

        full_data.loc[fam_all.index, "raw_stuff"] = raw_pred_all
        full_data.loc[fam_all.index, "Stuff_plus"] = (
            100 + 10 * ((raw_pred_all - mean) / std)
        )

        family_models[fam] = model

    #############################################
    # 8. AGGREGATE TO PITCHER-SEASON-PITCH TYPE
    #############################################

    pitcher_stuff = (
        full_data.groupby(["pitcher","game_year","pitch_type"])
        .agg(
            usage=("pitch_type","count"),
            avg_stuff=("Stuff_plus","mean")
        )
        .reset_index()
    )

    # Optional: minimum total pitch threshold
    pitcher_stuff = pitcher_stuff[pitcher_stuff["usage"] >= 25]

    #############################################
    # 9. SAVE OUTPUT
    #############################################

    con.register("pitcher_stuff_df", pitcher_stuff)

    con.execute("""
    CREATE OR REPLACE TABLE pitcher_stuff AS
    SELECT *
    FROM pitcher_stuff_df
    """)

    print("Physics-Based Stuff+ (Full Arsenal) Complete")
    con.close()

def compile_full_pitch_model():
    con = duckdb.connect("pitch_design.db")

    con.execute("""CREATE OR REPLACE TABLE pitcher_advanced_model_pitches AS
    SELECT rs.pitcher,
        p.full_name,
        rs.season,
        rs.pitch_type,
        rs.pitches_thrown,
        rs.usage_rate,        
        rs.release_pos_x,   
        rs.release_pos_y, 
        rs.arm_angle,
        rs.effective_speed,  
        rs.halfway_velo_x,  
        rs.halfway_velo_y,  
        rs.halfway_velo_z,  
        rs.halfway_accel_x,  
        rs.halfway_accel_y,  
        rs.halfway_accel_z,  
        rs.release_extension,
        rs.release_pos_y,
        rs.release_spin_rate,
        rs.horizontal_break,
        rs.vertical_break,  
        rs.spin_axis,
        rs.gravity_break,
        rs.arm_side_break,
        rs.inside_batter_break,
        rs.whiffs,
        rs.swings,   
        rs.csw_rate,    
        rs.called_strikes, 
        rs.hard_hits,
        rs.barrels,
        pg.whiff_plus,
        pg.Contact_plus,
        pg.Swing_plus,
        pg.Strike_plus,
        pg.Ball_plus,
        pg.Pitch_Grade_raw,
        pg.Pitch_Grade,
        ps.avg_stuff
    FROM(
        SELECT
            pitcher,
            game_year AS season,
            pitch_type,
            COUNT(*) AS pitches_thrown,
            COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (PARTITION BY pitcher, game_year) AS usage_rate,        
            AVG(release_pos_x) AS release_pos_x,   
            AVG(release_pos_y) AS release_pos_y, 
            AVG(arm_angle) AS arm_angle,
            AVG(effective_speed) AS effective_speed,  
            AVG(vx0) AS halfway_velo_x,  
            AVG(vy0) AS halfway_velo_y,  
            AVG(vz0) AS halfway_velo_z,  
            AVG(ax) AS halfway_accel_x,  
            AVG(ay) AS halfway_accel_y,  
            AVG(az) AS halfway_accel_z,  
            AVG(release_extension) AS release_extension,
            AVG(release_pos_y) AS release_pos_y,
            AVG(release_spin_rate) AS release_spin_rate,
            AVG(pfx_x) AS horizontal_break,
            AVG(pfx_z) AS vertical_break,  
            AVG(spin_axis) AS spin_axis,
            AVG(api_break_z_with_gravity) AS gravity_break,
            AVG(api_break_x_arm) AS arm_side_break,
            AVG(api_break_x_batter_in) AS inside_batter_break,
            SUM(whiff_flag) AS whiffs,
            SUM(swing_flag) AS swings,   
            (SUM(whiff_flag) + SUM(called_strike_flag)) * 1.0 / COUNT(*) AS csw_rate,    
            SUM(called_strike_flag) AS called_strikes, 
            SUM(hard_hit_flag) AS hard_hits,
            SUM(barrel_flag) AS barrels
        FROM raw_statcast 
        GROUP BY pitcher, game_year, pitch_type) AS rs
    INNER JOIN pitch_grade pg ON rs.pitcher = pg.pitcher AND rs.season = pg.game_year AND rs.pitch_type = pg.pitch_type
    INNER JOIN pitcher_stuff ps ON rs.pitcher = ps.pitcher AND rs.season = ps.game_year AND rs.pitch_type = ps.pitch_type 
    INNER JOIN player_id_map p ON rs.pitcher = p.key_mlbam
    ORDER BY rs.pitcher, rs.season DESC, usage_rate DESC;""")


    con.execute("COPY (SELECT * FROM pitcher_advanced_model_pitches) TO 'pitcher_summarized_data.csv' (HEADER, DELIMITER ',');")

    print("Full pitch model completed. Full dataset is in pitcher_summarized_data.csv")
    #Close db connection
    con.close()

#Run Main Function
if __name__ == '__main__':
    calculate_whiff_plus()
    calculate_contact_plus()
    calculate_strike_plus()
    calculate_ball_plus()
    calculate_swing_plus()
    calculate_pitch_composite_score()
    calculate_stuff_plus()
    compile_full_pitch_model()
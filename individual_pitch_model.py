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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#############################################
# 1. CALCULATE WHIFF+
#############################################
def calculate_whiff_plus():

    #############################################
    # 1 CONNECT TO DB
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2 LOAD DATA
    #############################################

    df = con.execute("""
        SELECT
            pitcher,
            pitch_type,
            game_date,
            swing_flag,
            whiff_flag
        FROM raw_statcast
        WHERE pitch_type IS NOT NULL
    """).df()

    #############################################
    # 3 EXTRACT SEASON
    #############################################

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_year"] = df["game_date"].dt.year

    #############################################
    # 4 FILTER TO SWINGS
    #############################################

    swings = df[df["swing_flag"] == 1].copy()

    #############################################
    # 5 CALCULATE WHIFF RATE
    #############################################

    pitch_whiff = (
        swings
        .groupby(["pitcher","game_year","pitch_type"])
        .agg(
            swings=("swing_flag","count"),
            whiffs=("whiff_flag","sum")
        )
        .reset_index()
    )

    pitch_whiff["whiff_rate"] = pitch_whiff["whiffs"] / pitch_whiff["swings"]

    #############################################
    # 6 REMOVE SMALL SAMPLE
    #############################################

    pitch_whiff = pitch_whiff[pitch_whiff["swings"] >= 20]

    #############################################
    # 7 LEAGUE NORMALIZATION
    #############################################

    pitch_whiff["Whiff_plus"] = (
        pitch_whiff
        .groupby(["game_year","pitch_type"])["whiff_rate"]
        .transform(lambda x: 100 + 10*((x-x.mean())/x.std()))
    )

    #############################################
    # 8 SAVE RESULTS
    #############################################

    con.register("whiff_plus_df", pitch_whiff)

    con.execute("""
        CREATE OR REPLACE TABLE pitcher_whiff_plus AS
        SELECT * FROM whiff_plus_df
    """)

    print("Whiff+ model complete.")

    con.close()

#############################################
# 2. CALCULATE CONTACT+
#############################################
def calculate_contact_plus():
    #############################################
    # 1 CONNECT TO DB
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2 LOAD DATA
    #############################################

    df = con.execute("""
        SELECT
            pitcher,
            pitch_type,
            game_date,
            description,
            estimated_woba_using_speedangle
        FROM raw_statcast
        WHERE pitch_type IS NOT NULL
    """).df()

    #############################################
    # 3 EXTRACT SEASON
    #############################################

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_year"] = df["game_date"].dt.year

    #############################################
    # 4 FILTER TO CONTACT EVENTS
    #############################################

    CONTACT_EVENTS = [
        "hit_into_play",
        "hit_into_play_score",
        "hit_into_play_no_out"
    ]

    df = df[df["description"].isin(CONTACT_EVENTS)].copy()

    #############################################
    # 5 CALCULATE CONTACT QUALITY
    #############################################

    contact_quality = (
        df.groupby(["pitcher","game_year","pitch_type"])
        .agg(
            contacts=("description","count"),
            contact_woba=("estimated_woba_using_speedangle","mean")
        )
        .reset_index()
    )

    #############################################
    # 6 REMOVE SMALL SAMPLES
    #############################################

    contact_quality = contact_quality[contact_quality["contacts"] >= 10]

    #############################################
    # 7 NORMALIZE INTO CONTACT+
    #############################################

    contact_quality["Contact_plus"] = (
        contact_quality
        .groupby(["game_year","pitch_type"])["contact_woba"]
        .transform(lambda x: 100 - 10*((x-x.mean())/x.std()))
    )

    #############################################
    # 8 SAVE RESULTS
    #############################################

    con.register("contact_plus_df", contact_quality)

    con.execute("""
        CREATE OR REPLACE TABLE pitcher_contact_plus AS
        SELECT * FROM contact_plus_df
    """)

    print("Contact+ model complete.")

    con.close()

#############################################
# 3. CALCULATE STRIKE+
#############################################
def calculate_strike_plus():
    #############################################
    # 1. CONNECT TO DB
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2. LOAD DATA
    #############################################

    df = con.execute("""
        SELECT
            pitcher,
            pitch_type,
            game_date,
            description
        FROM raw_statcast
        WHERE pitch_type IS NOT NULL
    """).df()

    #############################################
    # 3. EXTRACT SEASON
    #############################################

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_year"] = df["game_date"].dt.year

    #############################################
    # 4. DEFINE STRIKE EVENTS
    #############################################

    STRIKE_EVENTS = [
        "called_strike",
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip"
    ]

    df["is_strike"] = df["description"].isin(STRIKE_EVENTS).astype(int)

    #############################################
    # 5. CALCULATE STRIKE RATE
    #############################################

    strike_rate = (
        df.groupby(["pitcher","game_year","pitch_type"])
        .agg(
            pitches=("description","count"),
            strikes=("is_strike","sum")
        )
        .reset_index()
    )

    strike_rate["strike_rate"] = strike_rate["strikes"] / strike_rate["pitches"]

    #############################################
    # 6. REMOVE SMALL SAMPLES
    #############################################

    strike_rate = strike_rate[strike_rate["pitches"] >= 10]

    #############################################
    # 7. NORMALIZE INTO STRIKE+
    #############################################

    strike_rate["Strike_plus"] = (
        strike_rate
        .groupby(["game_year","pitch_type"])["strike_rate"]
        .transform(lambda x: 100 + 10*((x-x.mean())/x.std()))
    )

    #############################################
    # 8. SAVE RESULTS
    #############################################

    con.register("strike_plus_df", strike_rate)

    con.execute("""
        CREATE OR REPLACE TABLE pitcher_strike_plus AS
        SELECT * FROM strike_plus_df
    """)

    print("Strike+ model complete.")

    con.close()
#############################################
# 4. CALCULATE BALL+
#############################################
def calculate_ball_plus():
    #############################################
    # 1 CONNECT TO DATABASE
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2 LOAD DATA
    #############################################

    df = con.execute("""
        SELECT
            pitcher,
            pitch_type,
            game_date,
            description
        FROM raw_statcast
        WHERE pitch_type IS NOT NULL
    """).df()

    #############################################
    # 3 EXTRACT SEASON
    #############################################

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_year"] = df["game_date"].dt.year

    #############################################
    # 4 DEFINE BALL EVENTS
    #############################################

    BALL_EVENTS = [
        "ball",
        "blocked_ball",
        "ball_in_dirt"
    ]

    df["is_ball"] = df["description"].isin(BALL_EVENTS).astype(int)

    #############################################
    # 5 CALCULATE BALL RATE
    #############################################

    ball_rate = (
        df.groupby(["pitcher","game_year","pitch_type"])
        .agg(
            pitches=("description","count"),
            balls=("is_ball","sum")
        )
        .reset_index()
    )

    ball_rate["ball_rate"] = ball_rate["balls"] / ball_rate["pitches"]

    #############################################
    # 6 REMOVE SMALL SAMPLES
    #############################################

    ball_rate = ball_rate[ball_rate["pitches"] >= 10]

    #############################################
    # 7 NORMALIZE INTO BALL+
    #############################################

    ball_rate["Ball_plus"] = (
        ball_rate
        .groupby(["game_year","pitch_type"])["ball_rate"]
        .transform(lambda x: 100 - 10*((x-x.mean())/x.std()))
    )

    #############################################
    # 8 SAVE RESULTS
    #############################################

    con.register("ball_plus_df", ball_rate)

    con.execute("""
        CREATE OR REPLACE TABLE pitcher_ball_plus AS
        SELECT * FROM ball_plus_df
    """)

    print("Ball+ model complete.")

    con.close()

#############################################
# 5. CALCULATE CHASE+
#############################################
def calculate_chase_plus():
    #############################################
    # 1 CONNECT TO DB
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2 LOAD DATA
    #############################################

    df = con.execute("""
        SELECT
            pitcher,
            pitch_type,
            game_date,
            zone,
            swing_flag
        FROM raw_statcast
        WHERE pitch_type IS NOT NULL
    """).df()

    #############################################
    # 3 EXTRACT SEASON
    #############################################

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_year"] = df["game_date"].dt.year

    #############################################
    # 4 IDENTIFY OUT OF ZONE PITCHES
    #############################################

    df["out_of_zone"] = df["zone"] > 9

    #############################################
    # 5 FILTER TO PITCHES OUTSIDE THE ZONE
    #############################################

    oz = df[df["out_of_zone"] == True].copy()

    #############################################
    # 6 CALCULATE CHASE RATE
    #############################################

    chase = (
        oz.groupby(["pitcher","game_year","pitch_type"])
        .agg(
            pitches_outside=("out_of_zone","count"),
            chases=("swing_flag","sum")
        )
        .reset_index()
    )

    chase["chase_rate"] = chase["chases"] / chase["pitches_outside"]

    #############################################
    # 7 REMOVE SMALL SAMPLES
    #############################################

    chase = chase[chase["pitches_outside"] >= 10]

    #############################################
    # 8 NORMALIZE INTO CHASE+
    #############################################

    chase["Chase_plus"] = (
        chase
        .groupby(["game_year","pitch_type"])["chase_rate"]
        .transform(lambda x: 100 + 10*((x-x.mean())/x.std()))
    )

    #############################################
    # 9 SAVE RESULTS
    #############################################

    con.register("chase_plus_df", chase)

    con.execute("""
        CREATE OR REPLACE TABLE pitcher_chase_plus AS
        SELECT * FROM chase_plus_df
    """)

    print("Chase+ model complete.")

    con.close()

#############################################
# 6. CALCULATE PITCHGRADE+
#############################################
def calculate_pitch_composite_score():

        #############################################
    # 1 CONNECT TO DATABASE
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2 LOAD METRIC TABLES
    #############################################

    whiff = con.execute("SELECT pitcher, game_year, pitch_type, Whiff_plus FROM pitcher_whiff_plus").df()
    contact = con.execute("SELECT pitcher, game_year, pitch_type, Contact_plus FROM pitcher_contact_plus").df()
    chase = con.execute("SELECT pitcher, game_year, pitch_type, Chase_plus FROM pitcher_chase_plus").df()
    strike = con.execute("SELECT pitcher, game_year, pitch_type, Strike_plus FROM pitcher_strike_plus").df()
    ball = con.execute("SELECT pitcher, game_year, pitch_type, Ball_plus FROM pitcher_ball_plus").df()

    #############################################
    # 3 MERGE ALL METRICS
    #############################################

    df = whiff.merge(contact, on=["pitcher","game_year","pitch_type"], how="outer")

    df = df.merge(chase, on=["pitcher","game_year","pitch_type"], how="outer")
    df = df.merge(strike, on=["pitcher","game_year","pitch_type"], how="outer")
    df = df.merge(ball, on=["pitcher","game_year","pitch_type"], how="outer")

    #############################################
    # 4 DROP MISSING VALUES
    #############################################

    df = df.dropna()

    #############################################
    # 5 BUILD WEIGHTED PITCH SCORE
    #############################################

    df["pitch_score"] = (

        0.35 * df["Whiff_plus"]
        + 0.30 * df["Contact_plus"]
        + 0.15 * df["Chase_plus"]
        + 0.15 * df["Strike_plus"]
        + 0.05 * df["Ball_plus"]

    )

    #############################################
    # 6 NORMALIZE INTO PITCHGRADE+
    #############################################

    df["PitchGrade_plus"] = (
        df
        .groupby(["game_year","pitch_type"])["pitch_score"]
        .transform(lambda x: 100 + 10*((x-x.mean())/x.std()))
    )

    #############################################
    # 7 SAVE RESULTS
    #############################################

    con.register("pitchgrade_plus_df", df)

    con.execute("""
    CREATE OR REPLACE TABLE pitch_grade AS
    SELECT * FROM pitchgrade_plus_df
    """)

    print("Pitch Grade model complete.")

    con.close()

#############################################
# 7. CALCULATE STUFF+
#############################################
def calculate_stuff_plus():
    #############################################
    # 1. CONNECT TO DUCKDB
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2. LOAD PITCH LEVEL DATA
    #############################################

    df = con.execute("""
        SELECT
            pitcher,
            game_year,
            pitch_type,
            release_speed,
            release_spin_rate,
            pfx_x,
            pfx_z,
            release_pos_x,
            release_pos_z,
            release_extension,
            stand,
            p_throws,
            description,
            estimated_woba_using_speedangle
        FROM raw_statcast
        WHERE pitch_type IS NOT NULL
    """).df()

    NASTY_EVENTS = [
    "swinging_strike",
    "swinging_strike_blocked",
    "called_strike",
    "foul",
    "hit_into_play"
    ]

    df = df[
        (df["release_speed"] >= 70) &
        (df["description"].isin(NASTY_EVENTS))
    ]

    print("Rows after filtering:", len(df))

    #############################################
    # 3. FEATURE ENGINEERING
    #############################################

    df["velo"] = df["release_speed"]
    df["spin"] = df["release_spin_rate"]
    df["velo_sq"] = df["velo"] ** 2
    df["spin_sq"] = df["spin"] ** 2

    # movement in inches
    df["ivb"] = df["pfx_z"] * 12
    df["hb"] = df["pfx_x"] * 12
    df["movement_total"] = np.sqrt(df["ivb"]**2 + df["hb"]**2)
    df["movement_sq"] = df["movement_total"]**2
    df["spin_eff_proxy"] = df["movement_total"] / df["spin"]

    # release metrics
    df["release_height"] = df["release_pos_z"]
    df["release_side"] = df["release_pos_x"]

    # drop missing
    df = df.dropna(subset=[
        "velo",
        "spin",
        "ivb",
        "hb",
        "release_extension"
    ])

    # relative spin
    df["spin_diff"] = df["spin"] - df.groupby("pitcher")["spin"].transform("mean")

    # pitch mirroring
    df["release_diff"] = np.sqrt(
    (df["release_height"] - df.groupby("pitcher")["release_height"].transform("mean"))**2 +
    (df["release_side"] - df.groupby("pitcher")["release_side"].transform("mean"))**2
    )

    # Seam shifted wake prox
    df["expected_movement"] = df["spin"] * 0.0005
    df["ssw_proxy"] = df["movement_total"] - df["expected_movement"]

    # Movement Shape
    df["movement_ratio"] = df["hb"] / (abs(df["ivb"]) + 1)

    df["break_angle"] = np.degrees(
        np.arctan2(df["ivb"], df["hb"])
    )

    # Add whiff indicator
    df["is_whiff"] = df["description"].isin([
    "swinging_strike",
    "swinging_strike_blocked"
    ]).astype(int)

    # Weak contact indicator
    df["weak_contact"] = (
    (df["estimated_woba_using_speedangle"] < 0.2)
    ).astype(int)

    # Foul ball indicator
    df["is_foul"] = (df["description"] == "foul").astype(int)

    #############################################
    # 4. IDENTIFY PRIMARY FASTBALL AND SPEED DIFFERENTIALS
    #############################################

    FASTBALL_TYPES = ["FF","SI","FC"]

    fastball_df = df[df["pitch_type"].isin(FASTBALL_TYPES)]

    fb_stats = (
        fastball_df
        .groupby("pitcher")
        .agg(
            fb_velo=("velo","mean"),
            fb_ivb=("ivb","mean"),
            fb_hb=("hb","mean")
        )
        .reset_index()
    )

    df = df.merge(fb_stats, on="pitcher", how="left")

    df["velo_diff"] = df["velo"] - df["fb_velo"]
    df["ivb_diff"] = df["ivb"] - df["fb_ivb"]
    df["hb_diff"] = df["hb"] - df["fb_hb"]

    # magnitude of movement difference
    df["movement_diff"] = np.sqrt(df["ivb_diff"]**2 + df["hb_diff"]**2)

    #############################################
    # 5. BUILD TARGET
    #############################################
    # Miss difficulty
    df["is_whiff"] = df["description"].isin([
        "swinging_strike",
        "swinging_strike_blocked"
    ]).astype(int)

    # Weak contact
    df["weak_contact"] = (
        df["estimated_woba_using_speedangle"] < 0.2
    ).astype(int)

    # Contact suppression
    df["contact_value"] = 1 - df["estimated_woba_using_speedangle"]

    # Fill missing xwoba
    event_weights = (
        1 - df.groupby("description")["estimated_woba_using_speedangle"].mean()
    )

    df["contact_value"] = df["contact_value"].fillna(
        df["description"].map(event_weights)
    )

    #############################################
    # 6. BUILD TARGET
    #############################################

    # Final target
    df["target"] = (
        0.65 * df["is_whiff"]       
        + 0.30 * df["weak_contact"]   
        + 0.15 * df["contact_value"] 
    )

    #############################################
    # 6. PREPARE MODEL DATA
    #############################################
    stuff_features = [

        # core physics
        "velo",
        "spin",
        "ivb",
        "hb",
        "movement_total",

        # efficiency / wake
        "spin_eff_proxy",
        "ssw_proxy",

        # release traits
        "release_extension",
        "release_height",
        "release_side",
        "release_diff",

        # fastball relative
        "velo_diff",
        "ivb_diff",
        "hb_diff",
        "movement_diff",
        "spin_diff",

        # movement shape
        "movement_ratio",
        "break_angle",

        # nonlinear
        "velo_sq",
        "spin_sq",
        "movement_sq"
    ]

    velo_features = [
        "velo",
        "velo_sq",
        "velo_diff"
    ]

    shape_features = [

        "spin",
        "spin_sq",
        "spin_eff_proxy",
        "ssw_proxy",

        "ivb",
        "hb",
        "movement_total",
        "movement_sq",

        "release_extension",
        "release_height",
        "release_side",
        "release_diff",

        "ivb_diff",
        "hb_diff",
        "movement_diff",

        "movement_ratio",
        "break_angle",

        "spin_diff"
    ]

    # automatically detect pitch dummy columns
    pitch_cols = [c for c in df.columns if c.startswith("pitch_") and c != "pitch_type"]

    stuff_features = stuff_features + pitch_cols

    # remove rows with missing values
    df = df.dropna(subset=stuff_features + ["target"])
    y = df["target"]

    #############################################
    # 7. TRAIN STUFF MODEL
    #############################################
    feature_means = df[stuff_features].mean()

    models = {}

    for pitch in df["pitch_type"].unique():

        pitch_df = df[df["pitch_type"] == pitch]

        X = pitch_df[stuff_features]
        y = pitch_df["target"]

        model = RandomForestRegressor(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=75,
            n_jobs=-1,
            random_state=42
        )

        model.fit(X, y)

        models[pitch] = model

        ################################
        # FULL STUFF PREDICTION
        ################################

        df.loc[pitch_df.index, "stuff_pred"] = model.predict(X)

        ################################
        # VELO CONTRIBUTION
        ################################

        X_velo = X.copy()

        # Neutralize shape features
        for col in stuff_features:
            if col not in velo_features:
                X_velo[col] = feature_means[col]

        df.loc[pitch_df.index, "velo_pred"] = model.predict(X_velo)

        ################################
        # SHAPE CONTRIBUTION
        ################################

        X_shape = X.copy()

        # Neutralize velocity features
        for col in stuff_features:
            if col not in shape_features:
                X_shape[col] = feature_means[col]

        df.loc[pitch_df.index, "shape_pred"] = model.predict(X_shape)

    #############################################
    # 8. AGGREGATE TO PITCH LEVEL
    #############################################

    stuff_pitch = (
        df.groupby(["pitcher","game_year","pitch_type"])
        .agg(
            stuff_value=("stuff_pred","mean"),
            shape_value=("shape_pred","mean"),
            velo_value=("velo_pred","mean"),
            pitches=("stuff_pred","count")
        )
        .reset_index()
    )

    #############################################
    # 9. REMOVE SMALL SAMPLES
    #############################################

    stuff_pitch = stuff_pitch[stuff_pitch["pitches"] >= 10]


    #############################################
    # 10. NORMALIZE INTO STUFF+
    #############################################

    stuff_pitch["Stuff_plus"] = (
        stuff_pitch
        .groupby("pitch_type")["stuff_value"]
        .transform(lambda x: 100 + 10*((x-x.mean())/x.std()))
    )

    stuff_pitch["Shape_plus"] = (
        stuff_pitch
        .groupby("pitch_type")["shape_value"]
        .transform(lambda x: 100 + 10*((x-x.mean())/x.std()))
    )

    stuff_pitch["Velo_plus"] = (
        stuff_pitch
        .groupby("pitch_type")["velo_value"]
        .transform(lambda x: 100 + 10*((x-x.mean())/x.std()))
    )

    #############################################
    # 11. SAVE RESULTS TO DUCKDB
    #############################################

    con.register("stuff_plus_df", stuff_pitch)

    con.execute("""
        CREATE OR REPLACE TABLE pitcher_stuff AS
        SELECT * FROM stuff_plus_df
    """)

    print("Stuff+ model complete.")

    con.close()

#############################################
# 9. CALCULATE GAP BETWEEN PITCHGRADE AND STUFF
#############################################
def calculate_pitch_gap():
    #############################################
    # 1 CONNECT TO DATABASE
    #############################################

    con = duckdb.connect("pitch_design.db")

    #############################################
    # 2 LOAD TABLES
    #############################################

    stuff = con.execute("""
        SELECT
            pitcher,
            game_year,
            pitch_type,
            Stuff_plus
        FROM pitcher_stuff
    """).df()

    grade = con.execute("""
        SELECT
            pitcher,
            game_year,
            pitch_type,
            PitchGrade_plus
        FROM pitch_grade
    """).df()

    #############################################
    # 3 MERGE DATA
    #############################################

    df = stuff.merge(
        grade,
        on=["pitcher","game_year","pitch_type"],
        how="inner"
    )

    #############################################
    # 4 CALCULATE GAP
    #############################################

    df["PitchGap"] = df["Stuff_plus"] - df["PitchGrade_plus"]

    #############################################
    # 5 SAVE RESULTS
    #############################################

    con.register("pitch_gap_df", df)

    con.execute("""
        CREATE OR REPLACE TABLE pitcher_pitch_gap AS
        SELECT * FROM pitch_gap_df
    """)

    print("PitchGap model complete.")

    con.close()

#############################################
# 9. BUILD FULL PITCHING MODEL
#############################################
def compile_full_pitch_model():
    con = duckdb.connect("pitch_design.db")

    con.execute("""CREATE OR REPLACE TABLE pitcher_advanced_model_pitches AS
    SELECT rs.pitcher,
        p.full_name,
        rs.season,
        rs.pitch_type,
        rs.pitches_thrown,
        rs.usage_rate,    
        rs.release_speed,     
        rs.release_pos_x,   
        rs.release_pos_y, 
        rs.release_extension,
        rs.arm_angle, 
        rs.release_spin_rate,
        rs.horizontal_break,
        rs.vertical_break,  
        rs.spin_axis,
        rs.whiffs,
        rs.swings,   
        rs.csw_rate,    
        rs.called_strikes, 
        rs.hard_hits,
        rs.barrels,
        pg.whiff_plus,
        pg.Contact_plus,
        pg.Chase_plus,
        pg.Strike_plus,
        pg.Ball_plus,
        pg.Pitch_score,
        pg.PitchGrade_plus,
        ps.velo_plus,
        ps.shape_plus,
        ps.stuff_plus,
        ppg.PitchGap
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
            AVG(release_speed) AS release_speed,  
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
    INNER JOIN pitcher_pitch_gap ppg ON rs.pitcher = ppg.pitcher AND rs.season = ppg.game_year AND rs.pitch_type = ppg.pitch_type  
    ORDER BY rs.pitcher, rs.season DESC, usage_rate DESC;""")


    con.execute("COPY (SELECT * FROM pitcher_advanced_model_pitches) TO 'pitch_level_model.csv' (HEADER, DELIMITER ',');")

    print("Full pitch model completed. Full dataset is in pitch_level_model.csv")
    #Close db connection
    con.close()

#############################################
# MAIN FUNCTION
#############################################
if __name__ == '__main__':
    calculate_whiff_plus()
    calculate_contact_plus()
    calculate_strike_plus()
    calculate_ball_plus()
    calculate_chase_plus()
    calculate_pitch_composite_score()
    calculate_stuff_plus()
    calculate_pitch_gap()
    compile_full_pitch_model()
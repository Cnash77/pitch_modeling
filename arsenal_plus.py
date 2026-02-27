import duckdb
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

#############################################
# 1. LOAD DATA
#############################################

con = duckdb.connect("pitch_design.db")

pitch_data = con.execute("""
SELECT *
FROM raw_statcast
""").df()

stuff = con.execute("""
SELECT pitcher, game_year, pitch_type, avg_stuff
FROM pitcher_stuff
""").df()

k_data = con.execute("""
SELECT pitcher, game_year, K_percent
FROM pitcher_season_stats
""").df()

#############################################
# 2. BUILD xwOBA TARGET (RUN PREVENTION)
#############################################

xwoba_data = pitch_data[
    pitch_data["estimated_woba_using_speedangle"].notna()
].copy()

xwoba_pa = (
    xwoba_data
    .groupby(["pitcher","game_year","game_pk","at_bat_number"])
    .agg(xwoba=("estimated_woba_using_speedangle","mean"))
    .reset_index()
)

xwoba_season = (
    xwoba_pa
    .groupby(["pitcher","game_year"])
    .agg(xwoba_allowed=("xwoba","mean"))
    .reset_index()
)

#############################################
# 3. CALCULATE VAA / HAA
#############################################

pitch_data["t_plate"] = -pitch_data["release_pos_y"] / pitch_data["vy0"]

pitch_data["vz_plate"] = pitch_data["vz0"] + pitch_data["az"] * pitch_data["t_plate"]
pitch_data["vy_plate"] = pitch_data["vy0"] + pitch_data["ay"] * pitch_data["t_plate"]
pitch_data["vx_plate"] = pitch_data["vx0"] + pitch_data["ax"] * pitch_data["t_plate"]

pitch_data["VAA"] = np.degrees(
    np.arctan(pitch_data["vz_plate"] / pitch_data["vy_plate"])
)

pitch_data["HAA"] = np.degrees(
    np.arctan(pitch_data["vx_plate"] / pitch_data["vy_plate"])
)

pitch_data = pitch_data[(pitch_data["VAA"] > -15) & (pitch_data["VAA"] < 5)]

#############################################
# 4. BUILD PITCH PROFILE
#############################################

pitch_profile = (
    pitch_data
    .groupby(["pitcher","game_year","pitch_type"])
    .agg(
        usage=("pitch_type","count"),
        avg_velo=("release_speed","mean"),
        avg_ivb=("api_break_z_with_gravity","mean"),
        avg_hb=("api_break_x_batter_in","mean"),
        avg_vaa=("VAA","mean")
    )
    .reset_index()
)

pitch_profile = pitch_profile[pitch_profile["usage"] >= 50]

pitch_profile = pitch_profile.merge(
    stuff,
    on=["pitcher","game_year","pitch_type"],
    how="left"
)

pitch_profile = pitch_profile.dropna(subset=["avg_stuff"])

#############################################
# 5. USAGE-WEIGHTED STUFF BASE
#############################################

pitch_profile["total_usage"] = (
    pitch_profile.groupby(["pitcher","game_year"])["usage"]
    .transform("sum")
)

pitch_profile["usage_weight"] = (
    pitch_profile["usage"] / pitch_profile["total_usage"]
)

pitch_profile["weighted_stuff"] = (
    pitch_profile["avg_stuff"] * pitch_profile["usage_weight"]
)

base_arsenal = (
    pitch_profile
    .groupby(["pitcher","game_year"])
    .agg(base_score=("weighted_stuff","sum"))
    .reset_index()
)

#############################################
# 6. GLOBAL SHAPE SCALER
#############################################

global_scaler = StandardScaler()
global_scaler.fit(
    pitch_profile[["avg_velo","avg_ivb","avg_hb"]]
)

#############################################
# 7. INTERACTION FEATURE ENGINEERING
#############################################

def compute_interactions(group):

    if len(group) < 2:
        return pd.Series({
            "velo_int": 0,
            "ivb_int": 0,
            "hb_int": 0,
            "vaa_int": 0,
            "stuff_int": 0,
            "min_shape_distance": 0
        })

    pitches = group.copy()

    scaled_shapes = global_scaler.transform(
        pitches[["avg_velo","avg_ivb","avg_hb"]]
    )

    velo_total = ivb_total = hb_total = vaa_total = stuff_total = 0
    distances = []

    for (i, row_i), (j, row_j) in combinations(pitches.iterrows(), 2):

        weight = row_i["usage_weight"] * row_j["usage_weight"]

        velo_total  += abs(row_i["avg_velo"] - row_j["avg_velo"]) * weight
        ivb_total   += abs(row_i["avg_ivb"]  - row_j["avg_ivb"])  * weight
        hb_total    += abs(row_i["avg_hb"]   - row_j["avg_hb"])   * weight
        vaa_total   += abs(row_i["avg_vaa"]  - row_j["avg_vaa"])  * weight
        stuff_total += abs(row_i["avg_stuff"] - row_j["avg_stuff"]) * weight

    for a, b in combinations(scaled_shapes, 2):
        distances.append(np.linalg.norm(a - b))

    return pd.Series({
        "velo_int": velo_total,
        "ivb_int": ivb_total,
        "hb_int": hb_total,
        "vaa_int": vaa_total,
        "stuff_int": stuff_total,
        "min_shape_distance": min(distances)
    })

interaction_df = (
    pitch_profile
    .groupby(["pitcher","game_year"])
    .apply(compute_interactions)
    .reset_index()
)

#############################################
# 8. MERGE CORE FEATURES
#############################################

arsenal = base_arsenal.merge(
    interaction_df,
    on=["pitcher","game_year"],
    how="left"
)

#############################################
# 9. LEARN INTERACTION WEIGHTS (RIDGE)
#############################################

interaction_features = [
    "velo_int",
    "ivb_int",
    "hb_int",
    "vaa_int",
    "stuff_int"
]

alphas = np.logspace(-3, 3, 50)

ridge_int = RidgeCV(alphas=alphas, cv=5)
ridge_int.fit(arsenal[interaction_features], arsenal["base_score"])

arsenal["interaction_score_raw"] = ridge_int.predict(
    arsenal[interaction_features]
)

print("\nOptimal Interaction Alpha:", ridge_int.alpha_)

#############################################
# 2️⃣ ORTHOGONALIZE INTERACTIONS
#############################################

# Regress interaction score on base_score
ridge_ortho = RidgeCV(alphas=alphas, cv=5)
ridge_ortho.fit(
    arsenal[["base_score"]],
    arsenal["interaction_score_raw"]
)

interaction_pred = ridge_ortho.predict(
    arsenal[["base_score"]]
)

# Residual = orthogonal component
arsenal["interaction_score"] = (
    arsenal["interaction_score_raw"] - interaction_pred
)

print("Orthogonalization Complete")

#############################################
# 3️⃣ BUILD DATASETS FOR TARGETS
#############################################

arsenal_k = arsenal.merge(
    k_data,
    on=["pitcher","game_year"],
    how="inner"
).dropna(subset=["K_percent"])

arsenal_d = arsenal.merge(
    xwoba_season,
    on=["pitcher","game_year"],
    how="inner"
).dropna(subset=["xwoba_allowed"])

#############################################
# 4️⃣ FEATURE MATRIX
#############################################

feature_cols = [
    "base_score",
    "interaction_score",
    "min_shape_distance"
]

#############################################
# 5️⃣ TRAIN K+ MODEL (RIDGE CV)
#############################################

ridge_k = RidgeCV(alphas=alphas, cv=5)
ridge_k.fit(
    arsenal_k[feature_cols],
    arsenal_k["K_percent"]
)

arsenal["K_raw"] = ridge_k.predict(
    arsenal[feature_cols]
)

print("\nOptimal K+ Alpha:", ridge_k.alpha_)
print("K+ Weights:")
for name, coef in zip(feature_cols, ridge_k.coef_):
    print(f"{name}: {coef:.4f}")

#############################################
# 6️⃣ TRAIN DESIGN+ MODEL (RIDGE CV)
#############################################

ridge_d = RidgeCV(alphas=alphas, cv=5)
ridge_d.fit(
    arsenal_d[feature_cols],
    arsenal_d["xwoba_allowed"]
)

arsenal["Design_raw"] = ridge_d.predict(
    arsenal[feature_cols]
)

print("\nOptimal Design+ Alpha:", ridge_d.alpha_)
print("Design+ Weights:")
for name, coef in zip(feature_cols, ridge_d.coef_):
    print(f"{name}: {coef:.4f}")

#############################################
# 7️⃣ SCALE TO PLUS METRICS
#############################################

def scale_to_plus(series, invert=False):
    z = (series - series.mean()) / series.std()
    if invert:
        z = -z
    return 100 + 10 * z

arsenal["ArsenalK_plus"] = scale_to_plus(
    arsenal["K_raw"],
    invert=False
)

# lower xwOBA is better → invert
arsenal["ArsenalDesign_plus"] = scale_to_plus(
    arsenal["Design_raw"],
    invert=True
)

#############################################
# 8️⃣ FINAL OUTPUT TABLE
#############################################

arsenal_final = arsenal[[
    "pitcher",
    "game_year",
    "ArsenalK_plus",
    "ArsenalDesign_plus"
]].copy()

con.register("arsenal_df", arsenal_final)

con.execute("""
CREATE OR REPLACE TABLE arsenal_plus AS
SELECT p.player_name,
       c.*
FROM arsenal_df c
INNER JOIN pitchers p
ON c.pitcher = p.pitcher
""")

con.execute("""
COPY (
    SELECT *
    FROM arsenal_plus
)
TO 'arsenal_plus.csv' (HEADER, DELIMITER ',');
""")

print("\nDual Arsenal+ System Complete\n")
print(arsenal_final.sort_values("ArsenalK_plus", ascending=False).head())

#############################################
# 🔎 DUAL ARSENAL+ EVALUATION BLOCK
#############################################

print("\n==============================")
print("DUAL ARSENAL+ EVALUATION REPORT")
print("==============================\n")

#############################################
# 1️⃣ DISTRIBUTION CHECK
#############################################

for col in ["ArsenalK_plus", "ArsenalDesign_plus"]:

    if col not in arsenal_final.columns:
        continue

    mean_val = arsenal_final[col].mean()
    std_val  = arsenal_final[col].std()

    print(f"{col} Distribution:")
    print(f"Mean: {mean_val:.2f}")
    print(f"Std Dev: {std_val:.2f}")

    if abs(mean_val - 100) < 0.5 and 9 <= std_val <= 11:
        print("✓ Scaling looks correct\n")
    else:
        print("⚠ Scaling may be off\n")

#############################################
# 2️⃣ CORRELATION CHECKS
#############################################

print("Correlation Checks:\n")

# --- ArsenalK+ vs K% ---

arsenal_k_eval = arsenal_final.merge(
    k_data,
    on=["pitcher","game_year"],
    how="inner"
)

if len(arsenal_k_eval) > 0:

    k_corr = arsenal_k_eval["ArsenalK_plus"].corr(
        arsenal_k_eval["K_percent"]
    )

    print(f"ArsenalK+ vs K% (n={len(arsenal_k_eval)}): r = {k_corr:.3f}")

    if k_corr >= 0.55:
        print("🔥 Very strong strikeout relationship\n")
    elif k_corr >= 0.45:
        print("✓ Strong strikeout relationship\n")
    elif k_corr >= 0.30:
        print("• Moderate strikeout relationship\n")
    else:
        print("⚠ Weak strikeout relationship\n")

# --- ArsenalDesign+ vs xwOBA ---

arsenal_d_eval = arsenal_final.merge(
    xwoba_season,
    on=["pitcher","game_year"],
    how="inner"
)

if len(arsenal_d_eval) > 0:

    design_corr = arsenal_d_eval["ArsenalDesign_plus"].corr(
        arsenal_d_eval["xwoba_allowed"]
    )

    print(f"ArsenalDesign+ vs xwOBA (n={len(arsenal_d_eval)}): r = {design_corr:.3f}")

    if design_corr <= -0.50:
        print("🔥 Very strong run prevention relationship\n")
    elif design_corr <= -0.35:
        print("✓ Strong run prevention relationship\n")
    elif design_corr <= -0.20:
        print("• Moderate run prevention relationship\n")
    else:
        print("⚠ Weak run prevention relationship\n")

#############################################
# 3️⃣ YEAR-TO-YEAR STABILITY
#############################################

print("\nYear-to-Year Stability:\n")

for col in ["ArsenalK_plus","ArsenalDesign_plus"]:

    df = arsenal_final[["pitcher","game_year",col]].dropna()

    shifted = df.copy()
    shifted["game_year"] += 1

    yoy = df.merge(
        shifted,
        on=["pitcher","game_year"],
        suffixes=("_curr","_prev")
    )

    if len(yoy) > 0:
        yoy_corr = yoy[f"{col}_curr"].corr(
            yoy[f"{col}_prev"]
        )
        print(f"{col} YoY r: {yoy_corr:.3f}")

        if yoy_corr > 0.60:
            print("✓ Good stability\n")
        else:
            print("⚠ Low stability\n")
    else:
        print(f"{col}: Not enough overlapping seasons\n")

#############################################
# 4️⃣ DEPENDENCE ON STUFF BASE
#############################################

print("\nDependence on Base Stuff:")

for col in ["ArsenalK_plus", "ArsenalDesign_plus"]:
    
    if col in arsenal.columns:
        corr = arsenal[col].corr(arsenal["base_score"])
        print(f"{col} vs weighted Stuff+: r = {corr:.3f}")

        if corr > 0.9:
            print("⚠ Extremely dependent on Stuff+")
        elif corr > 0.75:
            print("• Moderately dependent on Stuff+")
        else:
            print("✓ Meaningfully independent of Stuff+")
        print()
    else:
        print(f"⚠ {col} not found in dataframe\n")

#############################################
# 5️⃣ TOP / BOTTOM TABLES
#############################################

print("\nTop 5 ArsenalK+:")
print(
    arsenal_final
    .sort_values("ArsenalK_plus", ascending=False)
    .head()
)

print("\nTop 5 ArsenalDesign+:")
print(
    arsenal_final
    .sort_values("ArsenalDesign_plus", ascending=False)
    .head()
)

print("\nBottom 5 ArsenalK+:")
print(
    arsenal_final
    .sort_values("ArsenalK_plus")
    .head()
)

print("\nBottom 5 ArsenalDesign+:")
print(
    arsenal_final
    .sort_values("ArsenalDesign_plus")
    .head()
)

print("\nEvaluation Complete\n")

con.close()
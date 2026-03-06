import pandas as pd
import numpy as np
import duckdb
from pybaseball import pitching_stats
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.linear_model import RidgeCV

#############################################
# 1. CALCULATE PITCHER CLUTCH
#############################################
def pitcher_clutch():
    
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

    # Remove low-sample states
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
    # 12. SAVE TO DUCKDB
    #############################################

    con.register("clutch_df", clutch)

    con.execute("""
    CREATE OR REPLACE TABLE pitcher_clutch AS
    SELECT *    
    FROM clutch_df
    """)

    print("\nHistorical Clutch Model Complete\n")
    print(clutch.sort_values("Clutch_Score", ascending=False).head())

    con.close()

#############################################
# 2. CALCULATE ARSENAL PLUS
#############################################
def arsenal_plus():

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
    SELECT p.key_mlbam as pitcher, pss.game_year, pss.K_percent
    FROM pitcher_season_stats pss
    INNER JOIN player_id_map p ON pss.IDfg = p.key_fangraphs
    """).df()

    #############################################
    # 2. BUILD xwOBA TARGET
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
    # 9. LEARN INTERACTION WEIGHTS
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
    # 10. ORTHOGONALIZE INTERACTIONS
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
    # 11. BUILD DATASETS FOR TARGETS
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

    feature_cols = [
        "base_score",
        "interaction_score",
        "min_shape_distance"
    ]

    #############################################
    # 12. TRAIN K+ MODEL
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
    # 13. TRAIN DESIGN+ MODEL 
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
    # 14. SCALE TO PLUS METRICS
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
    # 15. FINAL OUTPUT TABLE
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
    SELECT *
    FROM arsenal_df 
    """)
                
    con.close()

#############################################
# 3. CALCULATE RELIEF RUN PREVENTION PLUS
#############################################
def relief_run_prevention_plus():
    #############################################
    # 1. LOAD DATA
    #############################################

    con = duckdb.connect("pitch_design.db")
    data = con.sql("SELECT * FROM raw_statcast").df()

    #############################################
    # 2. BUILD GAME STATE FEATURES
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
    # 5. IDENTIFY RELIEVERS (<20% 0F APPEARANCES ARE STARTS)
    #############################################

    # Create appearance_id first
    data["appearance_id"] = (
        data["game_pk"].astype(str) + "_" +
        data["pitcher"].astype(str)
    )

    # Start = pitcher throws in inning 1
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

    # COUNT TOTAL APPEARANCES
    appearances_per_year = (
        data.groupby(["pitcher","game_year"])
        .agg(total_appearances=("game_pk","nunique"))
        .reset_index()
    )

    # MERGE + CALCULATE START %
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

    # DEFINE RELIEVERS
    relievers = usage[
        usage["start_pct"] < 0.20
    ][["pitcher","game_year"]]

    # FILTER DATASET
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

    rrp = rrp[rrp["relief_pitches"] >= 300]

    #############################################
    # 14. SAVE OUTPUT
    #############################################

    con.register("rrp_df", rrp)

    con.execute("""
    CREATE OR REPLACE TABLE relief_run_prevention_plus AS
    SELECT *
    FROM rrp_df 
    """)

    print("\nRelief Run Prevention+ (Final Model) Complete\n")
    print(rrp.sort_values("RRP_plus", ascending=False).head())

    con.close()

#############################################
# 4. COMPILE FULL PITCHER MODEL
#############################################
def compile_full_pitcher_model():
    con = duckdb.connect("pitch_design.db")

    con.execute("""CREATE OR REPLACE TABLE pitcher_advanced_model_pitches AS
    SELECT rs.pitcher,
        p.full_name,
        rs.season,
        pss.WAR,
        pss.innings_pitched,
        rs.pitches_thrown,
        rs.whiffs,
        rs.swings,   
        rs.csw_rate,    
        rs.called_strikes, 
        pss.strikeouts,
        pss.K_percent,
        pss."K/9",
        pss.walks,
        pss.BB_percent,
        pss."BB/9",
        rs.hard_hits,
        rs.barrels,
        pss.home_runs,
        pss."H/9",
        pss."GB/FB",
        pss.AVG AS BA_allowed,
        pss.WHIP,
        pss.ERA,
        pss.ERA_plus,
        pss.FIP,
        pss.FIP_plus,
        pss.SIERA,
        a.ArsenalK_plus,
        a.ArsenalDesign_plus,
        rrpp.relief_apps,
        rrpp.total_relief_value,
        rrpp.RRP_plus
    FROM(
        SELECT
            pitcher,
            game_year AS season,
            COUNT(*) AS pitches_thrown,     
            SUM(whiff_flag) AS whiffs,
            SUM(swing_flag) AS swings,   
            (SUM(whiff_flag) + SUM(called_strike_flag)) * 1.0 / COUNT(*) AS csw_rate,    
            SUM(called_strike_flag) AS called_strikes, 
            SUM(hard_hit_flag) AS hard_hits,
            SUM(barrel_flag) AS barrels
        FROM raw_statcast 
        GROUP BY pitcher, game_year) AS rs
    INNER JOIN player_id_map p ON rs.pitcher = p.key_mlbam
    INNER JOIN pitcher_season_stats pss ON p.key_fangraphs = pss.IDfg AND rs.season = pss.game_year
    INNER JOIN arsenal_plus a ON p.key_mlbam = a.pitcher AND rs.season = a.game_year
    LEFT JOIN relief_run_prevention_plus rrpp ON p.key_mlbam = rrpp.pitcher AND rs.season = rrpp.game_year
    ORDER BY rs.pitcher, rs.season DESC, pss.innings_pitched DESC, rs.pitches_thrown DESC;""")


    con.execute("COPY (SELECT * FROM pitcher_advanced_model_pitches) TO 'pitcher_advanced_model.csv' (HEADER, DELIMITER ',');")
    con.close()

#############################################
# MAIN FUNCTION
#############################################
if __name__ == '__main__':
    pitcher_clutch()
    arsenal_plus()
    relief_run_prevention_plus()
    compile_full_pitcher_model()
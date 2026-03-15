import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#############################################
# 1. LOAD DATA
#############################################

con = duckdb.connect("pitch_design.db")

df = con.execute("""
SELECT
    pitch_pk,
    pitcher,
    player_name,
    pitch_type,
    pitch_name,
    game_year,
    description,
    game_date, 
    inning,
    inning_topbot,
    outs_when_up,
    balls,
    strikes,
    release_speed,
    release_spin_rate,
    release_pos_x,
    release_pos_z,
    release_extension,
    pfx_x,
    pfx_z
FROM raw_statcast
WHERE release_speed >= 70
AND pitch_type != 'FA'
""").df()

print(f"Loaded {len(df):,} pitches")

#############################################
# 2. FEATURE ENGINEERING
#############################################

# Raw movement stats
df["ivb"] = df["pfx_z"] * 12
df["hb"]  = df["pfx_x"] * 12
df["ivb_abs"] = abs(df["ivb"])
df["hb_abs"] = abs(df["hb"])
df["spin"] = df["release_spin_rate"]

# Calculate movement features
df["expected_movement"] = df["spin"] * 0.0005
df["movement_total"] = np.sqrt(df["ivb"]**2 + df["hb"]**2)
df["movement_over_expected"] = df["movement_total"] - df["expected_movement"]
df["movement_gap"] = abs(df["movement_total"] - df["spin"] * 0.0005)
df["movement_sep"] = abs(df["ivb_abs"] - df["hb_abs"]) * 1.2

# Movement direction
df["movement_angle"] = np.degrees(
    np.arctan2(df["ivb"], df["hb"])
)

# Target flag for nasty outcomes
df["is_good_outcome"] = df["description"].isin([
    "swinging_strike",
    "swinging_strike_blocked",
    "called_strike",
    "foul_tip"
]).astype(int)

#############################################
# 3. SHAPE NASTINESS VIA PCA
#############################################

shape_features = [
    "ivb_abs",
    "hb_abs",
    "movement_total",
    "movement_gap",
    "movement_sep",
    "movement_angle",
    "movement_over_expected"
]

# Drop bad data
df = df.dropna(subset=shape_features).copy()

scaler = StandardScaler()
Z = scaler.fit_transform(df[shape_features])

pca = PCA(n_components=3)
components = pca.fit_transform(Z)

df["nastiness_raw"] = np.sqrt(
    components[:,0]**2 +
    components[:,1]**2 +
    components[:,2]**2
)

df = df[df["is_good_outcome"] == 1].copy()

print(f"\nFiltered to nasty outcomes: {len(df):,} pitches testing")
#############################################
# 4. CALCULATE NASTINESS+
#############################################

df["Nastiness_plus"] = (
    df["nastiness_raw"]
    .transform(lambda x: 100 + 10 * ((x - x.mean()) / x.std()))
)

print("\nNastiness+ Distribution")
print("Mean:", round(df["Nastiness_plus"].mean(),2))
print("Std:", round(df["Nastiness_plus"].std(),2))

#############################################
# 6. SAVE FULL TABLE
#############################################

con.register("nastiness_df", df)

con.execute("""
    CREATE OR REPLACE TABLE pitch_nastiness AS
    SELECT *
    FROM nastiness_df
""")

#############################################
# 7. EXPORT 2025 RESULTS
#############################################

con.execute("COPY (SELECT * FROM pitch_nastiness WHERE game_year = 2025 ORDER BY Nastiness_plus DESC LIMIT 200) TO 'pitch_nastiness_model.csv' (HEADER, DELIMITER ',');")

print("\nDirectional Nastiness model complete.\n")

con.close()
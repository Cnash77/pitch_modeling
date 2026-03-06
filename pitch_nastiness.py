import duckdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#############################################
# 1. LOAD DATA
#############################################

con = duckdb.connect("pitch_design.db")

df = con.execute("""
SELECT
    pitch_pk,
    player_name,
    pitch_type,
    pitch_name,
    game_year,
    description,
    release_speed,
    effective_speed,
    release_spin_rate,
    release_extension,
    pfx_x,
    pfx_z
FROM raw_statcast
WHERE release_speed >= 70
AND description IN (
    'swinging_strike',
    'swinging_strike_blocked',
    'called_strike'
)
AND pitch_type != 'FA'
""").df()

print(f"Loaded {len(df):,} pitches")

#############################################
# 2. FEATURE ENGINEERING
#############################################

# Absolute movement
df["abs_vert_break"]  = df["pfx_z"].abs()
df["abs_horiz_break"] = df["pfx_x"].abs()

features = [
    "release_speed",
    "abs_vert_break",
    "abs_horiz_break",
    "release_spin_rate",
    "release_extension"
]

df = df.dropna(subset=features + ["pitch_type"])

#############################################
# 3. DEFINE PITCH FAMILIES
#############################################

fastballs = ["FF", "SI", "FC"]
breaking  = ["SL", "CU", "KC", "ST", "SV", "CS", "KN"]
offspeed  = ["CH", "FS", "SC", "FO"]

def pitch_family(pt):
    if pt in fastballs:
        return "fastball"
    elif pt in breaking:
        return "breaking"
    elif pt in offspeed:
        return "offspeed"
    else:
        return "other"

df["pitch_family"] = df["pitch_type"].apply(pitch_family)

#############################################
# 4. FAMILY-RELATIVE MAHALANOBIS
#############################################

df["nastiness_raw"] = np.nan

for fam in df["pitch_family"].unique():

    fam_df = df[df["pitch_family"] == fam].copy()

    if len(fam_df) < 1000:
        continue

    print(f"Processing {fam} ({len(fam_df):,} pitches)")

    X = fam_df[features]

    # Standardize within family
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    # Covariance matrix
    cov_matrix = np.cov(Z.T)

    # Stabilize inversion
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6

    inv_cov = np.linalg.inv(cov_matrix)

    # Mahalanobis distance
    distances = np.sqrt(np.sum(Z @ inv_cov * Z, axis=1))

    df.loc[fam_df.index, "nastiness_raw"] = distances

#############################################
# 5. LEAGUE-WIDE CALCULATING NASTINESS+
#############################################

mean = df["nastiness_raw"].mean()
std  = df["nastiness_raw"].std()

df["Nastiness_plus"] = 100 + 10 * (
    (df["nastiness_raw"] - mean) / std
)

print("\nNastiness+ Distribution")
print("Mean:", round(df["Nastiness_plus"].mean(),2))
print("Std:", round(df["Nastiness_plus"].std(),2))

#############################################
# 6. SAVE FULL NASTINESS+ TABLE
#############################################

con.register("nastiness_df", df)

con.execute("""
    CREATE OR REPLACE TABLE pitch_nastiness AS
    SELECT *
    FROM nastiness_df
""")

#############################################
# 7. EXPORT 2025 NASTINESS+
#############################################

df_2025 = df[df["game_year"] == 2025]
df_2025.to_csv("pitch_nastiness_2025.csv", index=False)

print("\nDirectional Nastiness model complete.\n")

con.close()
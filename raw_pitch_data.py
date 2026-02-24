import pybaseball
from pybaseball import statcast
import duckdb
import pandas as pd

con = duckdb.connect("pitch_design.db")
data = statcast(start_dt="2021-03-01", end_dt="2025-12-31")
pybaseball.cache.enable()

con.execute("CREATE TABLE raw_statcast AS SELECT * FROM data")

con.execute("""ALTER TABLE raw_statcast ADD COLUMN whiff_flag INTEGER;
UPDATE raw_statcast
SET whiff_flag = CASE
    WHEN description IN ('swinging_strike', 'swinging_strike_blocked')
    THEN 1
    ELSE 0
END;""")

con.execute("""ALTER TABLE raw_statcast ADD COLUMN swing_flag INTEGER;
UPDATE raw_statcast
SET swing_flag = CASE
    WHEN description IN (
        'swinging_strike',
        'swinging_strike_blocked',
        'foul',
        'foul_tip',
        'hit_into_play',
        'hit_into_play_score',
        'hit_into_play_no_out'
    )
    THEN 1
    ELSE 0
END;""")

con.execute("""ALTER TABLE raw_statcast ADD COLUMN called_strike_flag INTEGER;
UPDATE raw_statcast
SET called_strike_flag = CASE
    WHEN description = 'called_strike'
    THEN 1
    ELSE 0
END;""")

con.execute("""ALTER TABLE raw_statcast ADD COLUMN hard_hit_flag INTEGER;
UPDATE raw_statcast
SET hard_hit_flag = CASE
    WHEN launch_speed >= 95 THEN 1
    ELSE 0
END;""")

con.execute("""ALTER TABLE raw_statcast ADD COLUMN barrel_flag INTEGER;
UPDATE raw_statcast
SET barrel_flag = CASE
    WHEN launch_speed >= 98
         AND launch_angle BETWEEN 26 AND 30
    THEN 1
    ELSE 0
END;""")

con.execute("""ALTER TABLE raw_statcast ADD COLUMN pitch_pk VARCHAR;
UPDATE raw_statcast
SET pitch_pk =
    CAST(pitcher AS VARCHAR) || '_' ||
    CAST(game_pk AS VARCHAR) || '_' ||
    CAST(at_bat_number AS VARCHAR) || '_' ||
    CAST(pitch_number AS VARCHAR);""")

con.execute("COPY (SELECT * FROM raw_statcast LIMIT 100) TO 'raw_statcast.csv' (HEADER, DELIMITER ',');")

#Close db connection
con.close()
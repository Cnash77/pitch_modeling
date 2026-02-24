import pybaseball
from pybaseball import statcast
import duckdb
import pandas as pd

con = duckdb.connect("pitch_design.db")

con.execute("""CREATE OR REPLACE TABLE pitchers AS
SELECT DISTINCT
    pitcher,
    player_name
FROM raw_statcast 
ORDER BY pitcher DESC;""")
#Close db connection
con.close()
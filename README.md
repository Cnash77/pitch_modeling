# pitchEffectiveness
Charlie Nash's pitch effectiveness modeling repository

TABLE OF CONTENTS:
1. create_raw_datasets.py: A python script that creates three tables of raw data:
   -pitcher_season_stats, a table containing raw baseball reference data, including columns such as ERA+, FIP, and FIP+
   -raw_statcast, a table containing raw statcast data for pitching and hitting at the individual pitch level
   -player_id_map, a player dimension table that can join baseball reference and statcast data together

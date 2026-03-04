# pitchEffectiveness

Charlie Nash's pitch effectiveness modeling repository

TABLE OF CONTENTS:

1. create_raw_datasets.py: A python script that creates three tables of raw data:

      -pitcher_season_stats, a table containing raw baseball reference data, including columns such as ERA+, FIP, and FIP+

      -raw_statcast, a table containing raw statcast data for pitching and hitting at the individual pitch level

      -player_id_map, a player dimension table that can join baseball reference and statcast data together

The final product is the three raw data tables in a duckdb table called "pitch_design.db"

2. individual_pitch_model.py: A python script that models individual pitches in each pitcher's arsenal, by year and describes different metrics about the pitch. Metrics that I created in this model are: Whiff+, Contact+, Strike+, Ball+, Swing+, Pitch Grade, Stuff+. The final product is a csv titled "pitcher_summarized_data.csv" that models each pitcher's pitch arsenal and the grade of each pitch.

3. pitch_nastiness.py: A python script that models individual pitches and grades their "nastiness", aka their ability to fool hitters into a whiff or called strike based on the pitch's movement, velocity, and release. Our final product is a csv called "pitch_nastiness_2025.csv" that contains the pitch nastiness of every pitch in the 2025 season that ended in a whiff or called strike.

4. pitcher_model.py: A python script that models a pitcher's overall performance over each season since 2021. This is a combination of baseball reference data and advanced metrics like: clutch+, arsenal+ and relief_run_prevention+. The final product is a csv called "pitcher_advanced_model.csv" that contains basic statistics that you would see on baseball reference for each pitcher's season as well as the advanced statistics that we mentioned earlier. 

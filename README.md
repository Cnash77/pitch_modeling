# Pitch Effectiveness Modeling

Charlie Nash's pitch effectiveness modeling repository

TABLE OF CONTENTS:

1. create_raw_datasets.py: A python script that creates three tables of raw data:

      -pitcher_season_stats, a table containing raw baseball reference data, including columns such as ERA+, FIP, and FIP+

      -raw_statcast, a table containing raw statcast data for pitching and hitting at the individual pitch level

      -player_id_map, a player dimension table that can join baseball reference and statcast data together

      The final product is the three raw data tables in a duckdb table called "pitch_design.db"

2. individual_pitch_model.py: A python script that models individual pitches in each pitcher's arsenal, by year and describes different metrics about the pitch. Metrics that I created in this model are: Whiff+, Contact+, Strike+, Ball+, Chase+, Pitch Grade, Velo+, Shape+, and Stuff+. The final product is a csv titled "pitch_level_model.csv" that models each pitcher's pitch arsenal and the grade of each pitch.

3. pitch_nastiness.py: A python script that models individual pitches and grades their "nastiness", aka their ability to fool hitters into a whiff or called strike based on the pitch's movement, velocity, and release. Our final product is a csv called "pitch_nastiness_model.csv" that contains the pitch nastiness of every pitch in the 2025 season that ended in a whiff or called strike.

4. pitcher_model.py: A python script that models a pitcher's overall performance over each season since 2021. This is a combination of baseball reference data and advanced metrics like: clutch+, arsenal+ and relief_run_prevention+. The final product is a csv called "pitcher_level_model.csv" that contains basic statistics that you would see on baseball reference for each pitcher's season as well as the advanced statistics that we mentioned earlier. 

METRICS CREATED:

pitch_level_model.csv:

      1. Whiff+: A pitch's ability to generate whiffs. A higher score means that pitch that can generate more whiffs. 100 is league average.
      
      2. Contact+: A grade of the contact quality a pitch generates. A higher score means that pitch is able to generate less contact and weaker contact. 100 is league average.
      
      3. Strike+: A pitch's ability to generate strikes. A higher score means that pitch is able to generate more strikes. 100 is league average.
      
      4. Ball+: A pitch's ability to limit balls called. A higher score means that pitch is able to minimize balls. 100 is league average.
      
      5. Chase+: A pitch's ability to generate chases from a batter. A higher score means that pitch is chased by the batter more. 100 is league average.

      6. Velo+: A pitch's velocity profile, meaning its ability to generate velocity. A higher score means that the pitch moves faster than average and expection. 100 is league average.

      7. Shape+: A pitch's movement and spin profile, meaning its ability to generate movement. A higher score means that the pitch moves more than the average and expectation for that pitch. 100 is league average. 
      
      8. Stuff+: A pitch's ability to move with velocity and movement. A higher score means that pitch is able move with more speed and movement. 100 is league average.
      
      9. PitchGrade+: A composite score of a pitch's overall effectiveness, based on the metrics outlined. A higher score means a more effective pitch. 100 is league average.

pitch_nastiness_model.csv:

      1. nastiness_raw: A singular pitch's overall nastiness based on its velocty and movement, making it possible to generate a whiff or called strike.

      2. Nastiness+: A singular pitch's nastiness_raw normalized across the entire season's dataset. 100 is league average nastiness and a higher nastiness+ means a higher liklihood of a whiff or called strike. 

pitcher_level_model.csv:

      1. clutch+: A pitcher's "clutch" ability, defined as their ability to prevent runs in cases where win probability could shift the other way. A higher score means a greater ability to prevent runs in high leverage situations. 100 is leagure average. 
      
      2. arsenal+: A pitcher's arsenal's grade based on each pitch's effectiveness, how each pitch interatcts with each other, and how the pitcher utilizes their pitches. A higher score means a greater pitching arsenal and greater effectiveness. 100 is league average. 
      
      3. relief_run_prevention+: A normalized score of run prevention ability that a pitcher has in relief appearances. A higher score means that a pitcher is more effective at run prevention in relief appearances. 100 is league average. 

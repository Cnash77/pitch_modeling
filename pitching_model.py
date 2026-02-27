import pybaseball
from pybaseball import statcast
import duckdb
import pandas as pd

con = duckdb.connect("pitch_design.db")

con.execute("""CREATE OR REPLACE TABLE pitcher_advanced_model_pitches AS
SELECT rs.pitcher,
    p.player_name,
    rs.season,
    rs.pitch_type,
    rs.pitches_thrown,
    rs.usage_rate,        
    rs.release_pos_x,   
    rs.release_pos_y, 
    rs.arm_angle,
    rs.effective_speed,  
    rs.halfway_velo_x,  
    rs.halfway_velo_y,  
    rs.halfway_velo_z,  
    rs.halfway_accel_x,  
    rs.halfway_accel_y,  
    rs.halfway_accel_z,  
    rs.release_extension,
    rs.release_pos_y,
    rs.release_spin_rate,
    rs.horizontal_break,
    rs.vertical_break,  
    rs.spin_axis,
    rs.gravity_break,
    rs.arm_side_break,
    rs.inside_batter_break,
    rs.whiffs,
    rs.swings,   
    rs.csw_rate,    
    rs.called_strikes, 
    rs.hard_hits,
    rs.barrels,
    pg.whiff_plus,
    pg.Contact_plus,
    pg.Swing_plus,
    pg.Strike_plus,
    pg.Ball_plus,
    pg.Pitch_Grade_raw,
    pg.Pitch_Grade,
    ps.avg_stuff
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
        AVG(effective_speed) AS effective_speed,  
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
INNER JOIN pitchers p ON rs.pitcher = p.pitcher
ORDER BY rs.pitcher, rs.season DESC, usage_rate DESC;""")


con.execute("COPY (SELECT * FROM pitcher_advanced_model_pitches) TO 'pitcher_summarized_data.csv' (HEADER, DELIMITER ',');")

#Close db connection
con.close()
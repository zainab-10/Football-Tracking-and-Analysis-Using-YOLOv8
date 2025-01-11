import cv2
from trackers import Tracker  
from team_assigner import TeamAssigner  
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from player_touch import PlayerTouchDetector
import time

def display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    tracker = Tracker('models/new_best.pt') 
    
    frame_num = 0  
    i=0
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    player_touch_detector = PlayerTouchDetector()
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    view_transformer = ViewTransformer(frame_width, frame_height)
    team_assigner = TeamAssigner()  
    start_time = time.time()  
    previous_time = start_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tracks = tracker.get_object_tracks(
            frame, read_from_stub=False, stub_path="stubs/track_stubs.pkl"
        )
        tracker.add_position_to_tracks(tracks)
        camera_movement_estimator = CameraMovementEstimator(frame)
        camera_movement = camera_movement_estimator.get_camera_movement(frame)
        camera_movement_estimator.add_adjust_positions_to_tracks(
            tracks, camera_movement
        )
        view_transformer.add_transformed_position_to_tracks(tracks)
        
        if i==0:
            team_assigner.assign_team_color(frame, tracks["players"][0])
        i=1
        if tracks["ball"]!=[{}]:
            tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        current_time = time.time()
        time_elapsed = current_time - previous_time
        if time_elapsed>0:
            print(time_elapsed)
            speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks,time_elapsed)
            frame=speed_and_distance_estimator.draw_speed_and_distance(frame, tracks)
        previous_time = current_time

        for player_id, track in tracks["players"][0].items():
            bbox = track["bbox"]
            team = team_assigner.get_player_team(frame, bbox, player_id)
            tracks["players"][0][player_id]["team"] = team
            tracks["players"][0][player_id]["team_color"] = team_assigner.team_colors[team]
            color=team_assigner.team_colors[team]
            frame = tracker.draw_ellipse(frame, bbox, color, track_id=player_id)

        for object_name in ['ball', 'goal keeper', 'referee']:
            if object_name in tracks:
                for object_id, track in tracks[object_name][0].items():
                    bbox = track["bbox"]
                    if object_name == 'ball':
                        frame = tracker.draw_triangle(frame, bbox, (255, 255, 0))  
                    elif object_name == 'goal keeper':
                        frame = tracker.draw_ellipse(frame, bbox, (0, 255, 0))  
                    elif object_name == 'referee':
                        frame = tracker.draw_ellipse(frame, bbox, (0, 0, 255))  
        if tracks["ball"]!=[{}] and 'players' in tracks:
            ball_tracks = tracks['ball']
            player_tracks = tracks['players'][0]
            distances, lines = tracker.calculate_distances_and_lines(player_tracks, ball_tracks[0][1])
            # Draw lines and display distances
            tracker.draw_lines_and_distances(frame, lines, distances)
        if tracks["ball"]!=[{}] and 'players' in tracks:
            ball_position = tracks["ball"][0][1]['bbox']
            closest_team,player_id = tracker.assign_ball_to_player(tracks["players"][0], ball_position)
            if player_id!=-1:
                print(player_id)
                print(tracks['players'][0])
                player_bbox = tracks['players'][0][player_id]['bbox']
                player_ball_color=(0, 100, 100)
                player_touch_detector.count_player_touches(player_id)
                frame = tracker.draw_triangle(frame, player_bbox, player_ball_color) 
            frame = tracker.draw_team_ball_control(frame, closest_team)
        frame = player_touch_detector.display_player_possession(frame, tracks["players"][0])
        screen_width = 1280
        screen_height = 720
        resized_frame = cv2.resize(frame, (screen_width, screen_height))
        # Display the frame
        cv2.imshow('Football Tracking', cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_num += 1  

    cap.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    video_path = 'input_videos/clip.mp4'
    display_video(video_path)

import cv2 
import os
import pandas as pd
import numpy as np


df = pd.read_csv('/Volumes/Ammar/Hallam/AI R&D Project/British_equistrian/Dataset/Raw_csv/Scott Brash - Hello Jefferson - Aachen - 1.60m - 03.07.2022 Jumpoff.csv')
df

def extract_video_chunk(video_path, start_duration, end_duration, output_folder, video_name):
    # Check if output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video is successfully loaded
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps * 1000  # Total duration in milliseconds

    # Convert start and end duration to frames
    start_frame = int(start_duration * fps / 1000)
    end_frame = int(end_duration * fps / 1000)

    # Check if the provided start and end durations are within the video duration
    if start_duration < 0 or start_duration > total_duration or end_duration < 0 or end_duration > total_duration:
        print("Error: Invalid start or end duration.")
        return

    # Set the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read and save the video chunk
    frames = []
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break

    # Save the chunk as a video
    if frames:
        output_video_path = os.path.join(output_folder, f"{video_name}.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
        for frame in frames:
            out.write(frame)
        out.release()

    # Release the video capture
    cap.release()


video_path = "/Volumes/Ammar/Hallam/AI R&D Project/British_equistrian/Dataset/Raw_videos/Scott Brash - Hello Jefferson - Aachen - 1.60m - 03.07.2022 Jumpoff.mp4"
start_duration = 0
end_duration = 0
output_folder = "/Volumes/Ammar/Hallam/AI R&D Project/British_equistrian/Dataset/Data"
clear_path = "/Volumes/Ammar/Hallam/AI R&D Project/British_equistrian/Dataset/Data/clear"
unclear_path = "/Volumes/Ammar/Hallam/AI R&D Project/British_equistrian/Dataset/Data/unclear"
parallel_path = ""
upright_path = ""
indoor_path = "/Volumes/Ammar/Hallam/AI R&D Project/British_equistrian/Dataset/Data/indoor"
outdoor_path = "/Volumes/Ammar/Hallam/AI R&D Project/British_equistrian/Dataset/Data/outdoor"

positions = df['Position']

# for i in range(1, len(positions)):
#     start_duration = positions[i - 1]
#     end_duration = positions[i]
#     print("chunk:",start_duration,"-",end_duration)
#     extract_video_chunk(video_path, start_duration, end_duration, output_folder)

# positions = df['Position']

# # Iterate over the positions list and calculate differences
# for i in range(len(positions)):
#     if i == 0:
#         start_duration = positions[i]
#         end_duration = positions[i] + (positions[i + 1] - positions[i]) + 1000  # Add 1 second (1000 milliseconds) after the first chunk
#     elif i == len(positions) - 1:
#         start_duration = positions[i - 1] - 1000  # Subtract 1 second (1000 milliseconds) before the last chunk
#         end_duration = positions[i]
#     else:
#         start_duration = positions[i - 1] - 1000  # Subtract 1 second (1000 milliseconds) for the start duration
#         end_duration = positions[i] + 1000  # Add 1 second (1000 milliseconds) for the end duration

#     print("Chunk:", start_duration, "-", end_duration)
#     extract_video_chunk(video_path, start_duration, end_duration, output_folder)

duration = df['Duration']

for dur in duration:
  print(dur)

for index, row in df.iterrows():
    location = ""
    if row['jump'] == 1:
        output_folder = clear_path
    else:
        output_folder = unclear_path

    start_duration = row['Position']
    duration = row['Duration']
    end_duration = start_duration + duration
    extract_video_chunk(video_path, start_duration+850, end_duration, output_folder, duration)
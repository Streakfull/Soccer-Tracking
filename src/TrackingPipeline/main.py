import random
import math
import yaml
from tqdm import tqdm
from collections import defaultdict
from blocks.team_classifier import TeamClassifier
import torch
import glob
from ultralytics import YOLO, ASSETS
from tqdm import tqdm
import cv2
import os
from termcolor import cprint
# from pytorchmodels.DeepIOUTracker.HMTracker import HMTracker
from pytorchmodels.DeepIOUTracker.HMReidTracker import HMTracker
from torchvision import transforms
from pytorchmodels.TeamClassifier import TeamClassifier as TeamSupervisedClassifier

from create_video_from_frames import CreateVideoFromFrames
import numpy as np

from pytorchmodels.PlayerClassification import PlayerClassification
from uglf.util import video
from PIL import Image


class TrackingPipeline:
    def __init__(self, video_path, output_path, img_size=1920, batch_size=8, max_frames=None, ball_conf=None, use_spatial=False):
        self.video_path = video_path
        self.output_path = output_path
        self.input_frames_path = f"{self.output_path}/input_frames"
        self.output_frames_path = f"{self.output_path}/output_frames"
        self.det_files = f"{self.output_path}/det_files"
        self.ball_conf = ball_conf
        self.use_spatial = use_spatial

        # Class variables for YOLO inference settings
        self.img_size = img_size
        self.batch_size = batch_size
        self.max_frames = max_frames

        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.input_frames_path, exist_ok=True)
        os.makedirs(self.output_frames_path, exist_ok=True)
        os.makedirs(self.det_files, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
        else:
            self.fps = None
            print(f"Warning: Could not read FPS from {self.video_path}")

        self.create_video = CreateVideoFromFrames(
            [self.output_path], fps=self.fps)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def run(self):
        cprint('Writing files', "blue")
        self.write_files()
        cprint("Plotting", "blue")
        self.plot_bounding_boxes()
        cprint("Writing video", "blue")
        self.create_video.process_folders()

    def write_files(self):
        cprint('Writing video frames', "yellow")
        self.write_video_to_frames()
        cprint('Running Player detector', "yellow")
        self.run_player_detector()
        torch.cuda.empty_cache()
        cprint('Running ball detector custom', "yellow")
        self.run_detect_ball_custom()
        torch.cuda.empty_cache()
        cprint('Running ball-player detector yolo', "yellow")
        torch.cuda.empty_cache()
        self.run_detect_ball_player()
        cprint('Merging player custom and yolo players', "yellow")
        self.merge_player_custom_yolo()
        cprint('Ball Files', "yellow")
        if (self.use_spatial):
            self.merge_ball_files_spatial()
        else:
            self.merge_ball_files()
        cprint("filtering extra ball detections", "yellow")
        self.filter_extra_ball_detections()
        cprint('Running tracking', "yellow")
        self.run_tracking()
        cprint('Running team clustering', "yellow")
        torch.cuda.empty_cache()
        self.run_team_clustering()
        # self.run_supervised_classif()
        self.merge_ball_player()
        cprint('Adjusting team IDs', "blue")
        self.adjust_team_ids()

    def plot_bounding_boxes(self):
        # Path to the bb.txt file and output path for annotated frames
        bb_file = os.path.join(self.det_files, "bb_teams.txt")
        output_path = self.output_frames_path

        # Read all bounding boxes from bb.txt and organize them by frame_id
        frame_detections = defaultdict(list)
        with open(bb_file, 'r') as f:
            detections = f.readlines()

        # Organize detections by frame_id for efficient access
        for detection in detections:
            parts = detection.strip().split(',')
            frame_id = int(parts[0])
            player_id = parts[1]
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            conf = float(parts[6])
            # cls = int(parts[-3])
            # team_clsid = int(parts[-2])
            team_id = parts[-1]  # Team ID (for players, either 0, 1, 2)

            # Store detection by frame_id
            frame_detections[frame_id].append({
                'player_id': player_id,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'conf': conf,
                # 'cls': cls,
                # 'team_cls': team_clsid,
                'team_id': team_id
            })

        # Loop through all the frames in input_frames_path
        frame_files = sorted(os.listdir(self.input_frames_path))

        # Initialize tqdm progress bar
        with tqdm(total=len(frame_files), desc="Annotating frames", unit="frame") as pbar:
            # Loop through each frame and annotate with bounding boxes

            for frame_file in frame_files:
                # Extract the frame index from the filename
                # Assuming file format like '000001.jpg'
                frame_id = int(frame_file.split('.')[0])
                if self.max_frames is not None and frame_id >= self.max_frames:
                    break

                # Load the frame
                frame_path = os.path.join(self.input_frames_path, frame_file)
                frame = cv2.imread(frame_path)

                # Check if the frame was loaded correctly
                if frame is None:
                    print(f"Error loading frame {frame_path}")
                    continue

                # Get all detections for the current frame
                detections_for_frame = frame_detections.get(frame_id, [])

                # Loop over the detections for the current frame
                for detection in detections_for_frame:
                    player_id = detection['player_id']
                    x = detection['x']
                    y = detection['y']
                    w = detection['w']
                    h = detection['h']
                    # cls = detection['cls']
                    # team_cls = detection['team_cls']
                    team_id = detection['team_id']

                    # if (cls not in [0, 1] and player_id != 'B'):
                    #     entry = class_id_map.get(cls, "u")
                    #     label, color = entry["label"], entry["color"]
                    # else:
                    # Set the color and label for the bounding box
                    if player_id == 'B':
                        # Ball: black color for bounding box
                        color = (0, 0, 0)  # Black
                        label = "B"
                    else:
                        # Player: Different colors for each team
                        if team_id == '2':
                            color = (0, 0, 255)  # blue
                            label = f"{player_id}"
                        elif team_id == '0':
                            color = (255, 0, 0)  # red
                            label = f"{player_id}"
                        elif team_id == '1':
                            color = (0, 255, 0)  # green for Team 2
                            label = f"{player_id}"
                        else:
                            continue  # Skip invalid team ID

                    # Draw the bounding box on the frame
                    top_left = (int(x), int(y))
                    bottom_right = (int(x + w), int(y + h))
                    frame = cv2.rectangle(
                        frame, top_left, bottom_right, color, 2)

                    # Put the label text on the frame
                    cv2.putText(frame, label, (int(x), int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Save the annotated frame to the output directory
                output_frame_path = os.path.join(output_path, frame_file)
                cv2.imwrite(output_frame_path, frame)

                # Update the progress bar after each frame
                pbar.update(1)

            print(
                f"Bounding boxes have been plotted and saved in {output_path}")

    def filter_extra_ball_detections(self):
        HEIGHT_MIN = 30
        WIDTH_MIN = 25
        B_AREA = HEIGHT_MIN * WIDTH_MIN
        player_merged_path = os.path.join(self.det_files, "player_merged.txt")
        ball_merged_path = os.path.join(self.det_files, "ball_merged.txt")

        # Read ball detections from ball_merged.txt
        ball_detections = {}
        large_bt = {}
        with open(ball_merged_path, "r") as f:
            ball_lines = f.readlines()

        for line in ball_lines:
            parts = line.strip().split(',')
            if parts[1] == "B":  # Ensure it's a ball detection
                frame_id = int(parts[0])
                x, y, w, h = map(float, parts[2:6])
                area = w * h
                if (area <= B_AREA):
                    # Store ball area per frame
                    ball_detections[frame_id] = area
                large_bt[frame_id] = True

        # Compute average ball area
        if ball_detections:
            avg_ball_area = np.mean(list(ball_detections.values()))
        else:
            print("No ball detections found, skipping filtering.")
            return

        # Read player detections
        updated_player_detections = []
        new_ball_detections = []

        with open(player_merged_path, "r") as f:
            player_lines = f.readlines()
        test = False
        for line in tqdm(player_lines, desc="Filtering ball-like detections", unit="detections"):
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            x, y, w, h = map(float, parts[2:6])
            area = w * h
            # If area is close to the average ball area, consider it a ball
            if abs(area - avg_ball_area) / avg_ball_area < 1:  # 20% tolerance
                pass
                # Only move it if this frame has NO existing ball detection
                # test = True
                # if frame_id not in ball_detections and frame_id not in large_bt:
                #     # new_ball_line = f"{frame_id},B,{x},{y},{w},{h},1.0,-1,-1"
                #     # new_ball_detections.append(new_ball_line)
                #     # Mark frame as containing a ball
                #     ball_detections[frame_id] = area
            else:
                updated_player_detections.append(line.strip())

        # Overwrite player_merged.txt with filtered player detections
        with open(player_merged_path, "w") as f:
            f.write("\n".join(updated_player_detections) + "\n")

        # Append new ball detections to ball_merged.txt
        if new_ball_detections:
            with open(ball_merged_path, "a") as f:
                f.write("\n".join(new_ball_detections) + "\n")

        print(
            f"Filtered player detections and updated ball file in {self.det_files}")

    def merge_ball_player(self):
        player_file = os.path.join(self.det_files, "tracking.txt")
        ball_file = os.path.join(self.det_files, "ball_merged.txt")
        merged_file = os.path.join(self.det_files, "bb.txt")

        # Read player detections from the tracking file
        with open(player_file, 'r') as f:
            player_lines = f.readlines()

        # Read ball detections from the ball_merged file
        with open(ball_file, 'r') as f:
            ball_lines = f.readlines()

        # Create a list to store all merged lines (player + ball detections)
        merged_lines = []

        # Add player detections to the merged lines
        for line in player_lines:
            merged_lines.append(line.strip())

        # Add ball detections to the merged lines
        for line in ball_lines:
            merged_lines.append(line.strip())

        # Write the merged detections to the new file (bb.txt)
        with open(merged_file, 'w') as f:
            for line in merged_lines:
                f.write(line + "\n")

        cprint(
            f"Merged player and ball detections saved to {merged_file}", color="green")

    def merge_ball_files(self):
        ball_custom = os.path.join(self.det_files, "ball_custom.txt")
        yolo_ball = os.path.join(self.det_files, "ball_player_yolo.txt")
        merged_file = os.path.join(self.det_files, "ball_merged.txt")

        # Read the ball_custom detections
        with open(ball_custom, 'r') as f:
            ball_custom_lines = f.readlines()

        # Read the yolo ball detections (filter for class 32 only)
        with open(yolo_ball, 'r') as f:
            yolo_lines = f.readlines()

        # Create dictionaries for frame detections
        ball_custom_dict = {}
        for line in ball_custom_lines:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            confidence = float(parts[6])
            if frame_id not in ball_custom_dict:
                ball_custom_dict[frame_id] = []
            ball_custom_dict[frame_id].append((line, confidence))

        yolo_dict = {}
        for line in yolo_lines:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            class_id = int(parts[1])
            confidence = float(parts[6])
            if class_id == 32:  # Only consider ball (class 32) detections
                if frame_id not in yolo_dict:
                    yolo_dict[frame_id] = []
                yolo_dict[frame_id].append((line, confidence))

        # Merge the detections
        merged_lines = []

        # Process each frame from ball_custom_dict and yolo_dict
        all_frame_ids = set(ball_custom_dict.keys()).union(
            set(yolo_dict.keys()))

        for frame_id in all_frame_ids:
            # Get detections from ball_custom
            ball_custom_det = ball_custom_dict.get(frame_id, None)
            # Get detections from yolo (class 32)
            yolo_det = yolo_dict.get(frame_id, None)

            # If both files have detections, select the one with the higher confidence
            if ball_custom_det and yolo_det:
                # Take the detection with the higher confidence
                custom_confidence = ball_custom_det[0][1]
                yolo_confidence = yolo_det[0][1]
                if custom_confidence >= yolo_confidence:
                    # Set id as 'B' before adding to the merged file
                    parts = ball_custom_det[0][0].strip().split(',')
                    # Add 'B' as the second element (index 1)
                    parts[1] = "B"
                    merged_lines.append(",".join(parts))  # Add updated line
                else:
                    # Set id as 'B' before adding to the merged file
                    parts = yolo_det[0][0].strip().split(',')
                    # Add 'B' as the second element (index 1)
                    parts[1] = "B"
                    merged_lines.append(",".join(parts))  # Add updated line
            elif ball_custom_det:
                # If only ball_custom has detections
                parts = ball_custom_det[0][0].strip().split(',')
                parts[1] = "B"  # Add 'B' as the second element (index 1)
                merged_lines.append(",".join(parts))  # Add updated line
            elif yolo_det:
                # If only yolo has detections
                parts = yolo_det[0][0].strip().split(',')
                parts[1] = "B"  # Add 'B' as the second element (index 1)
                merged_lines.append(",".join(parts))  # Add updated line

        # Write the merged detections to the new file
        with open(merged_file, 'w') as f:
            for line in merged_lines:
                f.write(line + "\n")

        cprint(f"Merged ball detections saved to {merged_file}", color="green")

    def merge_player_custom_yolo(self):
        player_custom = os.path.join(self.det_files, "player_custom.txt")
        yolo_players = os.path.join(self.det_files, "ball_player_yolo.txt")
        merged_file = os.path.join(self.det_files, "player_merged.txt")
        # Read the player_custom detections
        with open(player_custom, 'r') as f:
            player_custom_lines = f.readlines()

        # Read the yolo players detections (filter for class 0 only)
        with open(yolo_players, 'r') as f:
            yolo_lines = f.readlines()

        # Create a dictionary for frame detections from player_custom
        player_custom_dict = {}
        for line in player_custom_lines:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            if frame_id not in player_custom_dict:
                player_custom_dict[frame_id] = []
            player_custom_dict[frame_id].append(line)

        # Create a dictionary for yolo player detections (class 0)
        yolo_dict = {}
        for line in yolo_lines:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            class_id = int(parts[1])
            if class_id == 0:  # Only consider player (class 0) detections
                if frame_id not in yolo_dict:
                    yolo_dict[frame_id] = []
                yolo_dict[frame_id].append(line)

        # Merge detections
        merged_lines = []

        # Process player_custom detections first
        for frame_id in player_custom_dict:
            # Add all detections from player_custom
            merged_lines.extend(player_custom_dict[frame_id])

        # Add YOLO detections to frames that don't have any detections in player_custom
        for frame_id in yolo_dict:
            if frame_id not in player_custom_dict:
                # Add valid detections from YOLO
                merged_lines.extend(yolo_dict[frame_id])

        # Write merged detections to the new file
        with open(merged_file, 'w') as f:
            for line in merged_lines:
                f.write(line)

        cprint(f"Merged detections saved to {merged_file}", color="green")

    def write_video_to_frames(self):
        """Extract frames from the video and save them sequentially with a progress bar."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {self.video_path}")
            return

        # Get total frame count for tqdm progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_index = 1
        with tqdm(total=total_frames, desc="Extracting Frames", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret or (self.max_frames is not None and frame_index >= self.max_frames):
                    break  # Stop when the video ends

                # Format filename as '000001.jpg', '000002.jpg', etc.
                frame_filename = os.path.join(
                    self.input_frames_path, f"{frame_index:06d}.jpg")

                # Save the frame
                cv2.imwrite(frame_filename, frame)
                frame_index += 1
                pbar.update(1)  # Update progress bar

        cap.release()
        print(
            f"Extracted {frame_index - 1} frames to {self.input_frames_path}")

    def load_yolo_pt(self):
        weights = "./runs/detect/fintune-soccernet/weights/best.pt"
        model = YOLO(weights, verbose=False)
        self.configs_path = "./configs/global_configs.yaml"
        model.fuse()
        return model

    def load_yolo_ball_detectc(self):
        weights_path = "./runsBall/detect/train25/weights/best.pt"
        model = YOLO(weights_path)
        model.fuse()
        return model

    def load_player_classif(self):
        self.configs_path = "./configs/global_configs.yaml"
        with open(self.configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
            self.siamese_configs = self.global_configs["model"]["siamese"]
        model = PlayerClassification(
            self.siamese_configs, self.global_configs["training"])
        ckpt_path = "../logs/training/classification/playerClassif/t1/2025_04_05_16_20_05/checkpoints/epoch-best.ckpt"
        model.load_ckpt(ckpt_path)
        device = "cuda:0"
        model.to(device)
        return model

    def run_player_detector(self):
        model = self.load_yolo_pt()
        self.run_yolo_inference(model, "player_custom.txt", classes=[0])

    def run_detect_ball_custom(self):
        model = self.load_yolo_ball_detectc()
        self.run_yolo_inference(model, "ball_custom.txt")

    def load_yolo(self):
        model = YOLO("yolo11x.pt", verbose=False)
        model.fuse()
        return model

    def run_detect_ball_player(self):
        model = self.load_yolo()
        self.run_yolo_inference(
            model, "ball_player_yolo.txt", classes=[32], min_conf=self.ball_conf)

    def extract_player_crops(self, img, player_detections, with_transform=False):
        """
        Extracts crops of the players from the image based on player detections.
        The detections are expected to be in the form of [x_center, y_center, width, height, confidence, class_id].
        The crops will be in RGB format.
        """
        player_crops = []

        img_height, img_width, _ = img.shape  # Get image dimensions

        for detection in player_detections:
            _, _, x_center, y_center, width, height, _, _, _, _ = detection
            x_center, y_center, width, height = float(
                x_center), float(y_center), float(width), float(height)
            # Convert from xywh format to xyxy format (top-left, bottom-right corners)
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Ensure the bounding box coordinates are valid (non-negative, within image bounds)
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)

            # Check that the crop is valid (positive area)
            if x_min < x_max and y_min < y_max:
                crop = img[y_min:y_max, x_min:x_max]  # Extract the crop
                crop_rgb = cv2.cvtColor(
                    crop, cv2.COLOR_BGR2RGB)  # Convert to RGB
                if with_transform:
                    crop_pil = Image.fromarray(crop_rgb)
                    crop_transformed = self.transform(crop_pil)
                    player_crops.append(crop_transformed)
                else:
                    player_crops.append(crop_rgb)
            else:
                print(
                    f"Skipping invalid crop: ({x_min}, {y_min}), ({x_max}, {y_max})")

        return player_crops

    def extract_all_player_crops(self, file="player_custom.txt", transform=False):
        """
        Extract player crops from frames and return both the crops and detection lines from the 'player_custom.txt'.

        Returns:
            list: A flattened list of player crops (RGB format).
            list: A flattened list of detection lines (strings) from 'player_custom.txt'.
        """
        crops = []
        detection_lines = []  # List to store the corresponding detection lines

        # Read the player custom detection file
        detections_file = os.path.join(self.det_files, file)

        try:
            with open(detections_file, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Detection file 'player_custom.txt' not found.")
            return [], []

        # Iterate through all detections and extract crops based on the frame_idx
        for line in lines:
            # Split the detection line into parts: [frame_idx, -1, x_min, y_min, x_max, y_max, confidence, class_id]
            parts = line.strip().split(',')
            frame_idx = int(parts[0])  # Frame index is the first value

            # Read the image corresponding to this frame index
            frame_file = os.path.join(
                self.input_frames_path, f"{frame_idx:06d}.jpg")
            img = cv2.imread(frame_file)

            if img is None:
                print(f"Error reading frame {frame_file}")
                continue

            crop_rgb = self.extract_player_crops(
                img, [parts], with_transform=transform)  # Convert to RGB

            # Append the crop and the corresponding detection line
            crops.append(crop_rgb)
            # Store the original detection line
            detection_lines.append(line.strip())
        crops = [item for sublist in crops for item in sublist]
        return crops, detection_lines

    def run_team_clustering(self):
        # Initialize the classifier
        classifier = TeamClassifier(
            device="cuda", batch_size=72, cluster=3, umapcomp=512, save_path=f"{self.output_path}/team_classifier.pkl"
        )

        # Extract player crops and corresponding detection lines
        crops, lines = self.extract_all_player_crops()
        cprint(f"Found {len(crops)} crops - starting fitting", color="green")

        # Fit the classifier
        # random.shuffle(crops)
        classifier.fit(crops)

        # Now, batch the detections and predict using classifier.predict
        batch_size = classifier.batch_size  # Use the same batch size as in training

        # Calculate total number of batches
        num_batches = (len(crops) + batch_size - 1) // batch_size

        # Initialize an empty list to store predictions
        crops, lines = self.extract_all_player_crops("tracking.txt")
        new_detections_file = os.path.join(self.det_files, "tracking.txt")
        cprint(f"Found {len(crops)} crops - starting Training", color="green")
        all_predictions = []
        # Loop over the batches with tqdm for progress tracking
        with tqdm(total=num_batches, desc="Predicting teams", unit="batch") as pbar:
            for i in range(num_batches):
                # Get the current batch
                batch = crops[i * batch_size:(i + 1) * batch_size]
                # Get predictions for the batch
                predictions = classifier.predict(batch)
                # Add predictions to the list
                all_predictions.extend(predictions)
                pbar.update(1)  # Update the progress bar

        # At this point, all_predictions contains the team assignments for each crop
        cprint(
            f"Clustering completed. {len(all_predictions)} predictions made.", color="green")

        # Now, update the detection file with the predicted team IDs
        detections_path = os.path.join(self.det_files, "tracking.txt")

        try:
            with open(detections_path, "r") as f:
                detections = f.readlines()
        except FileNotFoundError:
            print(f"Detection file 'player_custom.txt' not found.")
            return

        updated_detection_lines = []

        # Update each detection line with the predicted team ID
        for idx, line in enumerate(lines):
            parts = line.split(',')
            # Replace the final -1 with the predicted team ID
            # Update the team ID in the bounding box
            parts[-1] = str(all_predictions[idx])
            updated_line = ",".join(parts)
            updated_detection_lines.append(updated_line)

        # Write the updated detection lines back to the detection file
        with open(detections_path, "w") as f:
            for updated_line in updated_detection_lines:
                f.write(updated_line + "\n")

        cprint(
            f"Detection file '{detections_path}' updated with team predictions.", color="green")

    def run_crops_classif(self):
        model = self.load_player_classif()
        crops, lines = self.extract_all_player_crops(
            "tracking.txt", transform=True)
        new_detections_file = os.path.join(self.det_files, "tracking.txt")
        cprint(f"Found {len(crops)} crops - starting Training", color="green")
        all_predictions = []
        batch_size = 64  # Use the same batch size as in training

        # Calculate total number of batches
        num_batches = (len(crops) + batch_size - 1) // batch_size
        # Loop over the batches with tqdm for progress tracking
        with tqdm(total=num_batches, desc="Predicting teams", unit="batch") as pbar:
            for i in range(num_batches):
                # Get the current batch
                batch = crops[i * batch_size:(i + 1) * batch_size]
                batch = torch.stack(batch).to("cuda:0")
                # Get predictions for the batch
                with torch.no_grad():
                    predictions = model.inference_with_cls(batch)
                # Add predictions to the list
                all_predictions.extend(predictions)
                pbar.update(1)  # Update the progress bar

        # At this point, all_predictions contains the team assignments for each crop
        cprint(
            f"Crops classification complete. {len(all_predictions)} predictions made.", color="green")

        # Now, update the detection file with the predicted team IDs
        detections_path = os.path.join(self.det_files, "tracking.txt")

        try:
            with open(detections_path, "r") as f:
                detections = f.readlines()
        except FileNotFoundError:
            print(f"Detection file 'player_custom.txt' not found.")
            return

        updated_detection_lines = []

        # Update each detection line with the predicted team ID
        for idx, line in enumerate(lines):
            parts = line.split(',')
            # Replace the final -1 with the predicted team ID
            # Update the team ID in the bounding box

            parts[-3] = str(all_predictions[idx].item())
            updated_line = ",".join(parts)
            updated_detection_lines.append(updated_line)

        # Write the updated detection lines back to the detection file
        with open(detections_path, "w") as f:
            for updated_line in updated_detection_lines:
                f.write(updated_line + "\n")

        cprint(
            f"Detection file '{detections_path}' updated with crops classification.", color="green")

    def run_tracking(self):
        torch.cuda.empty_cache()
        sequence_name = "tracking"
        tracker = HMTracker(self.det_files,
                            use_high_res=True, res=1920)
        tracker.process_frames_from_folder_fixed_bb2(
            frames_folder=self.input_frames_path,
            write_path="",
            global_text_file_path=self.det_files,
            sequence_name=sequence_name,
            write_directly=False,
            max_frames=self.max_frames,
            bb_file=os.path.join(self.det_files, "player_merged.txt")
        )

    def adjust_team_ids(self):
        bb_file = os.path.join(self.det_files, "bb.txt")
        output_file = os.path.join(self.det_files, "bb_teams.txt")

        player_team_counts = defaultdict(lambda: defaultdict(int))

        with open(bb_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(',')
            player_id = parts[1]
            team_id = parts[-1]
            player_team_counts[player_id][team_id] += 1

        player_majority_team = {
            player_id: max(team_counts, key=team_counts.get)
            for player_id, team_counts in player_team_counts.items()
        }

        with open(output_file, 'w') as f:
            for line in lines:
                parts = line.strip().split(',')
                player_id = parts[1]
                parts[-1] = player_majority_team[player_id]
                f.write(','.join(parts) + '\n')

        print(f"Adjusted team IDs saved to {output_file}")

    def run_yolo_inference(self, model, output_file="detections.txt", classes=None, min_conf=None):
        """
        Runs a YOLO model on input frames with batch processing.

        Args:
            model (YOLO): YOLO model to use for inference.
            output_file (str): Name of the output detection file.
            classes (list or None): List of class indices to filter detections.
        """
        frame_files = sorted(
            glob.glob(os.path.join(self.input_frames_path, "*.jpg")))

        if not frame_files:
            print("No frames found for detection!")
            return

        # Limit frames for debugging if max_frames is set
        if self.max_frames is not None:
            frame_files = frame_files[:self.max_frames]

        detections_path = os.path.join(self.det_files, output_file)

        with open(detections_path, "w") as f, tqdm(total=len(frame_files), desc="Running YOLO Detection", unit="frame") as pbar:
            for i in range(0, len(frame_files), self.batch_size):
                batch_files = frame_files[i: i + self.batch_size]
                batch_images = [cv2.imread(img_path)
                                for img_path in batch_files]

                # Run YOLO inference
                if (min_conf is not None):
                    results = model(
                        batch_images, imgsz=self.img_size, classes=classes, verbose=False, conf=min_conf)
                else:
                    results = model(
                        batch_images, imgsz=self.img_size, classes=classes, verbose=False)
                for frame_idx, (img_path, result) in enumerate(zip(batch_files, results), start=i + 1):
                    for t in result.boxes:
                        det_class = int(t.cls.item())
                        xyxy = t.xyxy[0].cpu().numpy()
                        tlwh = [xyxy[0], xyxy[1], xyxy[2] -
                                xyxy[0], xyxy[3] - xyxy[1]]
                        score = t.conf.item()

                        # Format: frame_idx,-1,tlwh[0],tlwh[1],tlwh[2],tlwh[3],score,-1,-1,-1
                        f.write(
                            f"{frame_idx},{det_class},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{score:.2f},-1,-1,-1\n")

                    pbar.update(1)

        print(f"Detections saved to {detections_path}")

    def load_supervised_team_classifier(self):
        ckpt_path = "../logs/training/week12/team-classif/full-train/2025_01_14_11_26_46/checkpoints/epoch-latest.ckpt"
        self.configs_path = "./configs/global_configs.yaml"
        with open(self.configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
            self.siamese_configs = self.global_configs["model"]["player_classif"]
        model = TeamSupervisedClassifier(self.siamese_configs)
        model.load_ckpt(ckpt_path=ckpt_path)
        device = "cuda:0"
        model.to(device)

        return model

    def run_supervised_classif(self):
        model = self.load_supervised_team_classifier()
        crops, lines = self.extract_all_player_crops(
            "tracking.txt", transform=False)
        new_detections_file = os.path.join(self.det_files, "tracking.txt")
        cprint(
            f"Found {len(crops)} crops - starting Training", color="green")
        all_predictions = []
        batch_size = 72
        num_batches = (len(crops) + batch_size - 1) // batch_size

        def _build_transform():
            """
            Builds the transformation pipeline.

            Returns:
                torchvision.transforms.Compose: A composition of image transformations.
            """

            transform_steps = [
                # Resize to the specified size
                transforms.Resize((128, 128)),
                transforms.ToTensor(),              # Convert to PyTorch tensor
            ]

            # transform_steps.append(
            #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
            #         0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            # )
            transform_steps.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]))

            return transforms.Compose(transform_steps)
        transform = _build_transform()
        # Loop over the batches with tqdm for progress tracking
        with tqdm(total=num_batches, desc="Predicting teams", unit="batch") as pbar:
            for i in range(num_batches):
                # Get the current batch
                batch = crops[i * batch_size:(i + 1) * batch_size]
                # Get predictions for the batch
                tensors = [transform(Image.fromarray(i)) for i in batch]
                tensors = torch.stack(tensors).to("cuda:0")
                predictions = model.inference(tensors)
                predictions = predictions.cpu().tolist()
                # predictions = classifier.predict(batch)
                # Add predictions to the list
                all_predictions.extend(predictions)
                pbar.update(1)  # Update the progress bar

        # At this point, all_predictions contains the team assignments for each crop
        cprint(
            f"Clustering completed. {len(all_predictions)} predictions made.", color="green")

        # Now, update the detection file with the predicted team IDs
        detections_path = os.path.join(self.det_files, "tracking.txt")

        try:
            with open(detections_path, "r") as f:
                detections = f.readlines()
        except FileNotFoundError:
            print(f"Detection file 'player_custom.txt' not found.")
            return

        updated_detection_lines = []

        # Update each detection line with the predicted team ID
        for idx, line in enumerate(lines):
            parts = line.split(',')
            # Replace the final -1 with the predicted team ID
            # Update the team ID in the bounding box
            parts[-1] = str(all_predictions[idx])
            updated_line = ",".join(parts)
            updated_detection_lines.append(updated_line)

        # Write the updated detection lines back to the detection file
        with open(detections_path, "w") as f:
            for updated_line in updated_detection_lines:
                f.write(updated_line + "\n")

        cprint(
            f"Detection file '{detections_path}' updated with team predictions.", color="green")

    def merge_ball_files_spatial(self, use_spatial_filter=True, max_frame_gap=100, max_distance=300):
        """
        Merges ball detections from custom and YOLO sources.

        Args:
            use_spatial_filter (bool): If True, filters YOLO detections that are too far from recent valid detection.
            max_frame_gap (int): Maximum number of frames to look back for valid comparison.
            max_distance (float): Maximum allowed distance between bounding box centers.
        """
        ball_custom = os.path.join(self.det_files, "ball_custom.txt")
        yolo_ball = os.path.join(self.det_files, "ball_player_yolo.txt")
        merged_file = os.path.join(self.det_files, "ball_merged.txt")

        # Read the ball_custom detections
        with open(ball_custom, 'r') as f:
            ball_custom_lines = f.readlines()

        # Read the yolo ball detections (filter for class 32 only)
        with open(yolo_ball, 'r') as f:
            yolo_lines = f.readlines()

        # Create dictionaries for frame detections
        ball_custom_dict = {}
        for line in ball_custom_lines:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            confidence = float(parts[6])
            ball_custom_dict.setdefault(
                frame_id, []).append((line, confidence))

        yolo_dict = {}
        for line in yolo_lines:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            class_id = int(parts[1])
            confidence = float(parts[6])
            if class_id == 32:  # Only consider ball (class 32) detections
                yolo_dict.setdefault(frame_id, []).append((line, confidence))

        merged_lines = []
        last_valid_detection = None  # (frame_id, x_center, y_center)

        def get_center(parts):
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            return x + w / 2, y + h / 2

        def too_far(current_parts):
            if not use_spatial_filter or last_valid_detection is None:
                return False
            last_frame, last_x, last_y = last_valid_detection
            curr_frame = int(current_parts[0])
            if abs(curr_frame - last_frame) > max_frame_gap:
                return False
            curr_x, curr_y = get_center(current_parts)
            distance = math.hypot(curr_x - last_x, curr_y - last_y)
            return distance > max_distance

        all_frame_ids = sorted(
            set(ball_custom_dict.keys()).union(set(yolo_dict.keys())))

        for frame_id in all_frame_ids:
            ball_custom_det = ball_custom_dict.get(frame_id, None)
            yolo_det = yolo_dict.get(frame_id, None)

            chosen_line = None

            if ball_custom_det and yolo_det:
                custom_line, custom_conf = ball_custom_det[0]
                yolo_line, yolo_conf = yolo_det[0]

                if custom_conf >= yolo_conf:
                    chosen_line = custom_line
                else:
                    parts = yolo_line.strip().split(',')
                    if not too_far(parts):
                        chosen_line = yolo_line
            elif ball_custom_det:
                chosen_line = ball_custom_det[0][0]
            elif yolo_det:
                parts = yolo_det[0][0].strip().split(',')
                if not too_far(parts):
                    chosen_line = yolo_det[0][0]

            if chosen_line:
                parts = chosen_line.strip().split(',')
                parts[1] = "B"  # Overwrite ID with 'B'
                merged_lines.append(",".join(parts))

                # Update last_valid_detection
                x_center, y_center = get_center(parts)
                last_valid_detection = (int(parts[0]), x_center, y_center)

        # Write merged detections to file
        with open(merged_file, 'w') as f:
            for line in merged_lines:
                f.write(line + "\n")


# video_path = "./raw_dataset/mancityVsLiverpool/clip_1.mp4"
# output_path = "../logs/trackingPipeline/debug"
# pipeline = TrackingPipeline(video_path, output_path, max_frames=32)
# pipeline.run()
video_paths = [
    # "./raw_dataset/clips/laliga-fcb-c1.mp4",
    # "./raw_dataset/clips/liverpool-mancity-clip1.mp4",
    # "./raw_dataset/clips/dfb-bayern-c1.mp4",
    "./raw_dataset/clips/ligue1-psg-c1.mp4",
    # "./raw_dataset/mancityVsLiverpool/clip_2.mp4",
    # "./raw_dataset/mancityVsLiverpool/clip_4.mp4",
]

video_paths = [
    "SNMOT-130.mp4",
    # "SNMOT-137.mp4",
    # "SNMOT-188.mp4",
    # "SNMOT-200.mp4"
]


max_frames = 12595
max_frames = None


output_folder = "trackingPipeline-ball-spatial"
# output_folder = "trackingPipelineFinalHMReid"
i = 0
for path in video_paths:
    output_path = path.split("/")[-1].split(".mp4")[0]
    output_path = f"../logs/{output_folder}/{output_path}"

    pipeline = TrackingPipeline(
        path, output_path, max_frames=max_frames, ball_conf=None, use_spatial=True,
    )

    pipeline.run()
    i += 1

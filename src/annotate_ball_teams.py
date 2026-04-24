from torch import nn, optim
from torchvision import transforms
import os
from pyexpat import model
import cv2
from tqdm import tqdm
from lutils.general import seed_all
from datasets.VisualSimMatches import VisualSimMatches
from datasets.TeamClusterDataset import TeamClusterDataset
from datasets.SimMatches import SimMatches
from training.ModelTrainer_reid_latest import ModelTrainer
import torch
import numpy as np
import cv2
import os
from blocks.team_classifier import TeamClassifier
from tqdm import tqdm
import itertools
import json
from PIL import Image


class AnnotateBallTeams:
    def __init__(self, raw_input_path="../logs/eval/week13-redemption2/v6-clip5/frames",
                 ball_labels_path="../logs/ball-detection/week13/clip5/labels_og",
                 max_frames=2000,
                 write_path="../logs/pipeline/week13/clip5",
                 # detection_labels="../logs/eval/week12-redemption/v4-with-kp/labels",
                 detection_labels="../logs/eval/week13-redemption2/v6-clip5/labels",
                 raw_frames="../logs/benchmarks/clip_5/raw_frames"
                 ):
        self.raw_input_path = raw_input_path
        self.ball_labels_path = ball_labels_path
        self.max_frames = max_frames
        self.write_path = write_path
        self.detection_labels = detection_labels
        trainer = ModelTrainer(dataset_type=TeamClusterDataset,
                               options={"tdm_notebook": False})
        self.image_size = (128, 128)  # Define your desired size (example)
        self.normalize = True  # You can modify this if needed
        self.model = trainer.model
        self.softmax = nn.Softmax(dim=1)
        self.raw_frames = raw_frames

        # Ensure write path exists
        os.makedirs(self.write_path, exist_ok=True)

    def plot_ball(self, detections, img):
        """
        Plots the detected ball bounding box (highest confidence) on the image.
        Each detection contains [class, x_min, y_min, x_max, y_max, confidence].
        \"\"\"
        if not detections:
            return img  # Return the original image if there are no detections

        # Find the detection with the highest confidence
        # Confidence is the last element
        best_detection = max(detections, key=lambda det: det[-1])
        _, x_min, y_min, x_max, y_max, conf = best_detection
        cv2.rectangle(img, (int(x_min), int(y_min)),
                      (int(x_max), int(y_max)), (0, 0, 0), 2)
        return img

    def plot_player_bounding_boxes(self, img, player_detections, team_predictions):
        """
        Plots bounding boxes around players and colors their IDs with a colored background
        based on team predictions. The bounding box color matches the team color.
        """
        # Updated team colors with dark green for Team 1
        team_colors = {
            1: (0, 0, 255),  # Red for Team 0
            2: (0, 128, 0),  # Dark Green for Team 1
            0: (255, 0, 0)   # Blue for Team 2
        }

        for detection, team_id in zip(player_detections, team_predictions):
            # Unpack player detection (x_min, y_min, x_max, y_max, _, _, id)
            x_min, y_min, x_max, y_max, _, _, player_id = detection

            # Ensure player_id and team_id are integers
            player_id = int(player_id)
            team_id = int(team_id)

            # Ensure the bounding box coordinates are within image bounds
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(img.shape[1], int(x_max))
            y_max = min(img.shape[0], int(y_max))

            # Get the color for the player's team (same as background color for text rect)
            # Default to white if team_id is not in team_colors
            team_color = team_colors.get(team_id, (255, 255, 255))

            # Draw the bounding box with the same color as the team background rectangle
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), team_color, 2)

            # Text properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(player_id)
            font_scale = 0.5
            font_thickness = 2
            text_size = cv2.getTextSize(
                text, font, font_scale, font_thickness)[0]
            text_w, text_h = text_size

            # Draw the colored background rectangle for the text
            rect_x1 = x_min
            rect_y1 = y_min - 25  # 25 is the height of the rectangle for text padding
            rect_x2 = x_min + text_w
            rect_y2 = y_min
            cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2),
                          team_color, -1)  # -1 fills the rectangle

            # Draw the player ID with white font on top of the rectangle
            cv2.putText(img, text, (x_min, y_min - 5), font, font_scale,
                        (255, 255, 255), font_thickness, cv2.LINE_AA)

        return img

    def _build_transform(self):
        """
        Builds the transformation pipeline.

        Returns:
            torchvision.transforms.Compose: A composition of image transformations.
        """
        transform_steps = [
            transforms.Resize(self.image_size),  # Resize to the specified size
            transforms.ToTensor(),              # Convert to PyTorch tensor
        ]

        if self.normalize:
            transform_steps.append(
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
                                     0.5, 0.5, 0.5])  # Normalize to [-1, 1]
            )

        return transforms.Compose(transform_steps)

    def read_ball_detections(self):
        """
        Reads ball detection labels from the specified directory and sorts them by frame number.
        Assumes file names are in the format 'frame_{x}.txt'.
        """
        detection_files = sorted(os.listdir(self.ball_labels_path),
                                 key=lambda x: int(x.split('_')[1].split('.')[0]))  # Extract frame number
        detections = {}
        for file in detection_files:
            frame_id = int(file.split('_')[1].split('.')[0])
            with open(os.path.join(self.ball_labels_path, file), 'r') as f:
                lines = f.readlines()
                # Parse each line as a detection
                # Format: [class, x_min, y_min, x_max, y_max, confidence]
                detections[frame_id] = [
                    list(map(float, line.strip().split())) for line in lines]
        return detections

    def read_player_detections(self):
        """
        Reads player detection labels from the specified detection label path.

        The detection label format is assumed to be:
        [x_min, y_min, x_max, y_max, confidence, class_id]

        Returns:
            dict: A dictionary with frame IDs as keys and a list of detections for each frame.
        """
        detection_files = sorted(os.listdir(self.detection_labels),
                                 key=lambda x: int(x.split('_')[1].split('.')[0]))  # Sort by frame number

        player_detections = {}
        for file in detection_files:
            frame_id = int(file.split('_')[1].split('.')[0])
            with open(os.path.join(self.detection_labels, file), 'r') as f:
                lines = f.readlines()
                # Parse each line in the file as [x_min, y_min, x_max, y_max, confidence, class_id]
                detections = [list(map(float, line.strip().split()))
                              for line in lines]
                player_detections[frame_id] = detections
        return player_detections

    def predict_teams(self, crops):
        with torch.no_grad():
            self.model.eval()
            crops = crops.to("cuda")
            pred = self.model.network(crops)
            pred = self.softmax(pred)
            pred = torch.max(pred, dim=1).indices
            return pred

    def process_video(self):
        """
        Processes frames from the input path, extracts player and ball crops based on detections,
        predicts teams for players, annotates the frame with player and ball detections,
        and saves them to the output path.
        """
        # Read player and ball detections
        player_detections = self.read_player_detections()
        ball_detections = self.read_ball_detections()

        # Sort frames by frame number
        frame_files = sorted(os.listdir(self.raw_frames),
                             key=lambda x: int(x.split('_')[1].split('.')[0]))  # Extract frame number

        frame_count = 0

        transform = self._build_transform()  # Build transformation pipeline

        for frame_file in tqdm(frame_files, desc="Processing frames"):
            if frame_count >= self.max_frames:
                break

            # Read the raw frame
            frame_id = int(frame_file.split('_')[1].split('.')[0])
            frame_path = os.path.join(self.raw_frames, frame_file)
            img = cv2.imread(frame_path)

            if img is None:
                print(f"Error reading frame {frame_path}, skipping...")
                continue

            # Get player and ball detections for this frame (if any)
            frame_player_detections = player_detections.get(frame_id, [])
            frame_ball_detections = ball_detections.get(frame_id, [])

            # Extract player crops from the frame based on the player detections
            player_crops = self.extract_player_crops(
                img, frame_player_detections)

            # Apply transformations to the player crops
            # Convert to PIL Images
            player_crops_pil = [Image.fromarray(crop) for crop in player_crops]
            player_crops_transformed = [
                transform(crop) for crop in player_crops_pil]
            # Stack them into a batch of tensors
            player_crops_tensor = torch.stack(player_crops_transformed)

            # Move player crops tensor to GPU if CUDA is available
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            player_crops_tensor = player_crops_tensor.to(
                device)  # Move to the appropriate device

            # Predict teams based on the cropped players
            team_predictions = self.predict_teams(player_crops_tensor)

            annotated_img = self.plot_player_bounding_boxes(
                img, frame_player_detections, team_predictions)

            # Annotate the frame with both player and ball detections
            annotated_img = self.plot_ball(
                frame_ball_detections, img)  # Annotate balls

            # Save the annotated frame
            save_path = os.path.join(self.write_path, f"{frame_file}")
            cv2.imwrite(save_path, annotated_img)

            frame_count += 1

    def extract_player_crops(self, img, player_detections):
        """
        Extracts crops of the players from the image based on player detections.
        The detections are expected to be in the form of[x_min, y_min, x_max, y_max, confidence, class_id].
        The crops will be in RGB format.
        """
        player_crops = []

        img_height, img_width, _ = img.shape  # Get image dimensions

        for detection in player_detections:
            x_min, y_min, x_max, y_max, _, _, id = detection

            # Ensure the bounding box coordinates are valid (non-negative, within image bounds)
            x_min = max(0, int(x_min))
            y_min = max(0, int(y_min))
            x_max = min(img_width, int(x_max))
            y_max = min(img_height, int(y_max))

            # Check that the crop is valid (positive area)
            if x_min < x_max and y_min < y_max:
                crop = img[y_min:y_max, x_min:x_max]  # Extract the crop
                crop_rgb = cv2.cvtColor(
                    crop, cv2.COLOR_BGR2RGB)  # Convert to RGB
                player_crops.append(crop_rgb)
            else:
                print(
                    f"Skipping invalid crop: ({x_min}, {y_min}), ({x_max}, {y_max})")

        return player_crops


# Example usage
if __name__ == "__main__":
    annotator = AnnotateBallTeams()
    annotator.process_video()

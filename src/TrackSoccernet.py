import os
import torch
from tqdm import tqdm
from pytorchmodels.DeepIOUTracker.HMReidTracker import HMTracker
# from pytorchmodels.SiameseTrackingNew import SiameseTracking
# from pytorchmodels.DeepIOUTracker.HMTracker import HMTracker


class TrackSoccernet:
    def __init__(self, dataset_root, write_path, res=None, debug=False, direct_path=None, use_crops_cls=False):
        """
        Initialize the tracker.

        Args:
            dataset_root (str): Root directory containing the 'det/' folder with sequences.
            write_path (str): Path to save tracking results.
        """
        self.dataset_root = dataset_root
        self.write_path = write_path
        self.res = res
        self.deubug = debug
        self.direct_path = direct_path
        self.use_crops_cls = use_crops_cls

        self.sn_bb = os.path.join(self.write_path, "sn_bb")
        # Get sequence names from the 'det/SNMOT-test/' folder
        det_folder = os.path.join(self.dataset_root)
        if not os.path.isdir(det_folder):
            raise ValueError(f"Detection folder not found: {det_folder}")

        self.sequences = sorted([
            seq for seq in os.listdir(det_folder)
            if os.path.isdir(os.path.join(det_folder, seq))
        ])
        if (self.deubug):
            self.sequences = ['SNMOT-200']

        if not self.sequences:
            raise ValueError(f"No sequences found in {det_folder}")

    def track_direct(self):
        torch.cuda.empty_cache()
        folder_path = self.write_path
        frames_path = self.direct_path
        sequence_name = "direct"
        frames_output_path = os.path.join(folder_path, "frames")
        frames_output_path = os.path.join(folder_path, "frames")
        labels_output_path = os.path.join(folder_path, "labels")
        os.makedirs(frames_output_path, exist_ok=True)
        os.makedirs(labels_output_path, exist_ok=True)
        tracker = HMTracker(frames_output_path,
                            use_high_res=True, res=self.res, use_crops_cls=self.use_crops_cls)
        tracker.process_frames_from_sorted(
            frames_folder=frames_path,
            write_path=frames_output_path,
            global_text_file_path=self.write_path,
            sequence_name=sequence_name,
            write_directly=True
        )

    def track(self, frames_path, sequence_name):
        """
        Track objects in a given sequence.

        Args:
            frames_path (str): Path to the folder containing frames for the sequence.
            sequence_name (str): Name of the sequence being processed.
        """
        torch.cuda.empty_cache()

        # Define write paths
        folder_path = os.path.join(self.write_path, sequence_name)
        frames_output_path = os.path.join(folder_path, "frames")
        labels_output_path = os.path.join(folder_path, "labels")

        # Create necessary directories
        # os.makedirs(frames_output_path, exist_ok=True)
        # os.makedirs(labels_output_path, exist_ok=True)

        # Initialize the tracker
        tracker = HMTracker(frames_output_path,
                            use_high_res=True, res=self.res, use_crops_cls=self.use_crops_cls)

        # Run tracking
        tracker.process_frames_from_folderg(
            frames_folder=frames_path,
            write_path=frames_output_path,
            global_text_file_path=self.write_path,
            sequence_name=sequence_name,
            write_directly=False
        )

    def track_fixed_bb(self, frames_path, sequence_name):
        """
        Track objects in a given sequence.

        Args:
            frames_path (str): Path to the folder containing frames for the sequence.
            sequence_name (str): Name of the sequence being processed.
        """
        torch.cuda.empty_cache()

        # Define write paths
        folder_path = os.path.join(self.write_path, sequence_name)
        frames_output_path = os.path.join(folder_path, "frames")
        labels_output_path = os.path.join(folder_path, "labels")

        # Create necessary directories
        # os.makedirs(frames_output_path, exist_ok=True)
        # os.makedirs(labels_output_path, exist_ok=True)

        # Initialize the tracker
        tracker = HMTracker(frames_output_path,
                            use_high_res=True, res=self.res, use_crops_cls=self.use_crops_cls)

        # Run tracking
        tracker.process_frames_from_folder_fixed_bb(
            frames_folder=frames_path,
            write_path=frames_output_path,
            global_text_file_path=self.write_path,
            sequence_name=sequence_name,
            write_directly=False,
            bb_folder=self.sn_bb
        )

    def track_fixed_bb_siamese(self, frames_path, sequence_name):
        """
        Track objects in a given sequence.

        Args:
            frames_path (str): Path to the folder containing frames for the sequence.
            sequence_name (str): Name of the sequence being processed.
        """
        torch.cuda.empty_cache()

        # Define write paths
        folder_path = os.path.join(self.write_path, sequence_name)
        frames_output_path = os.path.join(folder_path, "frames")
        labels_output_path = os.path.join(folder_path, "labels")

        # Create necessary directories
        # os.makedirs(frames_output_path, exist_ok=True)
        # os.makedirs(labels_output_path, exist_ok=True)

        # Initialize the tracker
        tracker = SiameseTracking(frames_output_path,
                                  write_path=frames_output_path)

        # Run tracking
        tracker.process_frames_from_folder_fixed_bb(
            frames_folder=frames_path,
            write_path=frames_output_path,
            global_text_file_path=self.write_path,
            sequence_name=sequence_name,
            write_directly=False,
            bb_folder=self.sn_bb
        )

    def track_all_sequences(self, fixed_bb=False):
        """
        Track all sequences in the dataset.
        """
        print(f"Tracking {len(self.sequences)} sequences...")

        for sequence_name in tqdm(self.sequences, desc="Tracking Progress"):
            frames_path = os.path.join(
                self.dataset_root, sequence_name, "img1")

            if not os.path.exists(frames_path):
                print(
                    f"Warning: Frames path {frames_path} does not exist. Skipping...")
                continue
            if (fixed_bb):
                self.track_fixed_bb(frames_path=frames_path,
                                    sequence_name=sequence_name)
            else:
                self.track(frames_path, sequence_name)


# Example Usage:
# output_path = "../logs/eval/tracking-soccernet/kp-random_training-adaptive-alpha"
# output_path = "../logs/eval/tracking-soccernet/yoloft-1280"
# os.makedirs(output_path, exist_ok=True)
# tracker = TrackSoccernet(
#     dataset_root="./raw_dataset/soccernet-tracking-test/raw/tracking/test",
#     write_path=output_path, res=1280)
# tracker.track_all_sequences()

# output_path = "../logs/eval/custom-plz/try2-ball-filter"
# raw_frames = "../logs/benchmarks/clip_1/raw_frames"
# os.makedirs(output_path, exist_ok=True)
# highres_tracker = TrackSoccernet(
#     dataset_root="./raw_dataset/soccernet-tracking-test/raw/tracking/test",
#     write_path=output_path, res=1920, debug=True, direct_path=raw_frames)
# # highres_tracker.track_all_sequences()
# highres_tracker.track_direct()

output_path = "../logs/eval/ablations-plz/no-fine-tune"
raw_frames = "../logs/benchmarks/clip_1/raw_frames"
debug = False
crops_cls = False
os.makedirs(output_path, exist_ok=True)
highres_tracker = TrackSoccernet(
    dataset_root="./raw_dataset/soccernet-tracking-test/raw/tracking/test",
    write_path=output_path, res=1920, debug=debug,
    direct_path=raw_frames, use_crops_cls=crops_cls)
highres_tracker.track_all_sequences(fixed_bb=False)
# highres_tracker.track_direct()

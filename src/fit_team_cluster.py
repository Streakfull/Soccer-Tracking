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


def get_crops(crops_path: str = "./raw_dataset/teams_clustering", target_size: int = 224):
    """
    Read and preprocess image crops from the given directory.

    Args:
        crops_path (str): Path to the directory containing image crops.
        target_size (int): Desired size (both width and height) for the crops.

    Returns:
        List[np.ndarray]: List of preprocessed image crops.
    """
    trainer = ModelTrainer(dataset_type=TeamClusterDataset,
                           options={"tdm_notebook": False})

    crops = []
    images = []

    for batch_index, batch_val in tqdm(enumerate(trainer.train_dataloader), total=len(
            trainer.train_dataloader)):
        crops.append(batch_val["path"])

    crops = [item for sublist in crops for item in sublist]
    for file_path in crops:
        image = cv2.imread(file_path)
    #         # if image is not None:
    #         #     processed_crop = preprocess_crop(image, target_size)
    #         #     crops.append(processed_crop)
        images.append(image)

    print("LEN IMAGES:", len(images))
    return images


def get_crops_batch(batch_val):
    crops = batch_val["path"]
    images = []
    for file_path in crops:
        image = cv2.imread(file_path)
        images.append(image)

    return images


conf_save_path = "../logs/eval/week12-redemption/team-classif"


def eval_supervised_model():
    trainer = ModelTrainer(dataset_type=TeamClusterDataset,
                           options={"tdm_notebook": False})
    model = trainer.model
    device = "cuda:0"
    total_confusion_matrix = {0: {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
                              1: {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
                              2: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}}

    for batch_index, batch_val in tqdm(enumerate(trainer.train_dataloader), total=len(
            trainer.train_dataloader)):

        with torch.no_grad():
            model.eval()
            TeamClusterDataset.move_batch_to_device(
                batch_val, device)
            y = model.inference(
                model.get_batch_input(batch_val))
            targets = batch_val["label"]
            confusion_matrix = calculate_confusion_matrix_multiclass(
                targets, y, 3)
            for cls in range(3):  # Assuming 3 classes
                total_confusion_matrix[cls]["TP"] += confusion_matrix[cls]["TP"]
                total_confusion_matrix[cls]["FP"] += confusion_matrix[cls]["FP"]
                total_confusion_matrix[cls]["FN"] += confusion_matrix[cls]["FN"]
                total_confusion_matrix[cls]["TN"] += confusion_matrix[cls]["TN"]
            # pdb.set_trace()
    with open(f"{conf_save_path}/supervised.json", "w") as f:
        json.dump(total_confusion_matrix, f, indent=4)


def calculate_confusion_matrix_multiclass(targets, y, num_classes=3):
    # Ensure the targets and y are of the same size
    assert targets.size() == y.size(), "Targets and predictions must have the same shape"

    # Initialize confusion matrix components for each class
    confusion_matrix = {i: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                        for i in range(num_classes)}

    for cls in range(num_classes):
        # Calculate TP, FP, FN, TN for class `cls`
        TP = torch.sum((targets == cls) & (y == cls)).item()
        FP = torch.sum((targets != cls) & (y == cls)).item()
        FN = torch.sum((targets == cls) & (y != cls)).item()
        TN = torch.sum((targets != cls) & (y != cls)).item()

        # Store the results for this class
        confusion_matrix[cls] = {"TP": TP, "FP": FP, "FN": FN, "TN": TN}
    return confusion_matrix


def calculate_total_precision_recall_f1(json_file):
    # Load confusion matrix data from JSON file
    with open(json_file, "r") as f:
        confusion_matrix = json.load(f)

    # Initialize total TP, FP, FN
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Accumulate TP, FP, and FN for each class
    for cls, metrics in confusion_matrix.items():
        total_tp += metrics["TP"]
        total_fp += metrics["FP"]
        total_fn += metrics["FN"]

    # Calculate total precision
    if total_tp + total_fp > 0:
        total_precision = total_tp / (total_tp + total_fp)
    else:
        total_precision = 0  # Avoid division by zero

    # Calculate total recall
    if total_tp + total_fn > 0:
        total_recall = total_tp / (total_tp + total_fn)
    else:
        total_recall = 0  # Avoid division by zero

    # Calculate total F1 score
    if total_precision + total_recall > 0:
        total_f1 = 2 * (total_precision * total_recall) / \
            (total_precision + total_recall)
    else:
        total_f1 = 0  # Avoid division by zero

    return total_precision, total_recall, total_f1


def fit():
    classifier = TeamClassifier(
        device="cuda", batch_size=32, cluster=2, save_path="./chkpts2/noumap.pkl")
    crops = get_crops()
    # pdb.set_trace()
    classifier.fit(crops)


def eval_unsupervised():
    trainer = ModelTrainer(dataset_type=TeamClusterDataset,
                           options={"tdm_notebook": False})
    classifier = TeamClassifier(
        device="cuda", batch_size=32, cluster=2, save_path="./chkpts2/noumap.pkl")
    classifier.load_model()
    total_confusion_matrix_a = {0: {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
                                1: {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
                                2: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}}
    total_confusion_matrix_b = {0: {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
                                1: {"TP": 0, "FP": 0, "FN": 0, "TN": 0},
                                2: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}}

    for batch_index, batch_val in tqdm(enumerate(trainer.train_dataloader), total=len(
            trainer.train_dataloader)):
        imgs = get_crops_batch(batch_val)
        targets = batch_val["label"]
        preds_a = classifier.predict(imgs)
        preds_b = 1 - preds_a
        preds_a = torch.Tensor(preds_a)
        preds_b = torch.Tensor(preds_b)

        confusion_matrix_a = calculate_confusion_matrix_multiclass(
            targets, preds_a, num_classes=3)
        confusion_matrix_b = calculate_confusion_matrix_multiclass(
            targets, preds_b, num_classes=3)
        for cls in range(3):  # Assuming 3 classes
            total_confusion_matrix_a[cls]["TP"] += confusion_matrix_a[cls]["TP"]
            total_confusion_matrix_a[cls]["FP"] += confusion_matrix_a[cls]["FP"]
            total_confusion_matrix_a[cls]["FN"] += confusion_matrix_a[cls]["FN"]
            total_confusion_matrix_a[cls]["TN"] += confusion_matrix_a[cls]["TN"]

        for cls in range(3):  # Assuming 3 classes
            total_confusion_matrix_b[cls]["TP"] += confusion_matrix_b[cls]["TP"]
            total_confusion_matrix_b[cls]["FP"] += confusion_matrix_b[cls]["FP"]
            total_confusion_matrix_b[cls]["FN"] += confusion_matrix_b[cls]["FN"]
            total_confusion_matrix_b[cls]["TN"] += confusion_matrix_b[cls]["TN"]

    with open(f"{conf_save_path}/noumap_a.json", "w") as f:
        json.dump(total_confusion_matrix_a, f, indent=4)
    with open(f"{conf_save_path}/noumap_b.json", "w") as f:
        json.dump(total_confusion_matrix_b, f, indent=4)


def eval_unsupervised_with_switches():
    trainer = ModelTrainer(dataset_type=TeamClusterDataset,
                           options={"tdm_notebook": False})
    classifier = TeamClassifier(
        device="cuda", batch_size=32, cluster=2, save_path="./chkpts2/3UMAP.pkl")
    classifier.load_model()

    num_classes = 3  # Assuming 3 classes
    all_possible_values = list(range(num_classes))  # Unique class values

    # Initialize total confusion matrices for all switch configurations
    total_confusion_matrices = []
    switch_permutations = list(itertools.permutations(all_possible_values))

    # Create an empty total confusion matrix for each switch configuration
    for _ in switch_permutations:
        total_confusion_matrices.append(
            {cls: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
                for cls in all_possible_values}
        )

    for batch_index, batch_val in tqdm(enumerate(trainer.train_dataloader), total=len(
            trainer.train_dataloader)):
        imgs = get_crops_batch(batch_val)
        targets = batch_val["label"]  # Ground truth labels
        preds = classifier.predict(imgs)  # Original predictions
        preds = torch.Tensor(preds)

        # Generate all simultaneous switch configurations for predictions
        switched_arrays = switch_with_simultaneous_changes(
            preds, all_possible_values)

        # Iterate over all switched arrays and accumulate their confusion matrices
        for switch_idx, switched_preds in enumerate(switched_arrays):
            # Calculate confusion matrix for this switched prediction
            confusion_matrix = calculate_confusion_matrix_multiclass(
                targets, switched_preds, num_classes=num_classes)

            # Accumulate confusion matrix into the total for this switch index
            for cls in all_possible_values:
                total_confusion_matrices[switch_idx][cls]["TP"] += confusion_matrix[cls]["TP"]
                total_confusion_matrices[switch_idx][cls]["FP"] += confusion_matrix[cls]["FP"]
                total_confusion_matrices[switch_idx][cls]["FN"] += confusion_matrix[cls]["FN"]
                total_confusion_matrices[switch_idx][cls]["TN"] += confusion_matrix[cls]["TN"]

    # Save each total confusion matrix as a separate JSON file
    for switch_idx, confusion_matrix in enumerate(total_confusion_matrices):
        with open(f"{conf_save_path}/simultaneous_switched_confusion_matrix_{switch_idx}.json", "w") as f:
            json.dump(confusion_matrix, f, indent=4)

    print(
        f"Saved {len(total_confusion_matrices)} confusion matrices to {conf_save_path}.")


def switch_to_all_full_replacements(array, values):
    """
    Generate all possible fully switched arrays by replacing all elements of one class
    with another class for the entire array.

    Args:
    - array (torch.Tensor): The tensor containing the original values.
    - values (list): The list of possible unique values in the array.

    Returns:
    - list: A list of tensors where each tensor corresponds to a fully switched array.
    """
    switched_arrays = []
    for val in values:
        for new_val in values:
            if val != new_val:  # Avoid switching to the same value
                switched = torch.clone(array)  # Clone the original tensor
                # Replace all occurrences of `val` with `new_val`
                switched[array == val] = new_val
                switched_arrays.append(switched)

    return switched_arrays


def switch_with_simultaneous_changes(array, values):
    """
    Generate all possible arrays by applying simultaneous switches between class values.

    Args:
    - array (torch.Tensor): The tensor containing the original values.
    - values (list): The list of possible unique values in the array.

    Returns:
    - list: A list of tensors representing all possible simultaneously switched arrays.
    """
    switched_arrays = []
    # Generate all possible permutations of the values (to account for simultaneous switches)
    permutations = itertools.permutations(values)

    for perm in permutations:
        # Create a mapping from original values to new values
        mapping = {original: new for original, new in zip(values, perm)}
        # Create a new tensor where all values are switched simultaneously
        switched = torch.clone(array)
        for original, new in mapping.items():
            switched[array == original] = new
        switched_arrays.append(switched)

    return switched_arrays


if __name__ == "__main__":
    # fit()
    # eval_supervised_model()
    # eval_unsupervised_with_switches()
    total_precision, total_recall, total_f1 = calculate_total_precision_recall_f1(
        "../logs/eval/week12-redemption/team-classif/simultaneous_switched_confusion_matrix_5.json")
    print(f"Total Precision: {total_precision:.4f}")
    print(f"Total Recall: {total_recall:.4f}")
    print(f"Total F1 Score: {total_f1:.4f}")
    # total_precision, total_recall, total_f1 = calculate_total_precision_recall_f1(
    #     "../logs/eval/week12-redemption/team-classif/noumap_b.json")
    # print(f"Total Precision: {total_precision:.4f}")
    # print(f"Total Recall: {total_recall:.4f}")
    # print(f"Total F1 Score: {total_f1:.4f}")

    # eval_unsupervised()

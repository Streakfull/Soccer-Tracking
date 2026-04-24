from torchvision import transforms
from clipreid.timmbackbone import OpenClipModel
from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel
import joblib
V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """

    def __init__(self, device: str = 'cpu',
                 batch_size: int = 32,
                 save_path="./chkpts2/model.pkl",
                 umapcomp=3,
                 cluster=2
                 ):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = self.load_features_model(device)

        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=umapcomp)
        self.cluster_model = KMeans(n_clusters=cluster)
        self.save_path = save_path

    def load_features_model(self, device):
        model = OpenClipModel("ViT-B/16",
                              "openai",
                              True)
        ckpt_path = self.save_path.replace("model.pkl", "epoch-latest.ckpt")
        state_dict = torch.load(ckpt_path)
        new_state_dict = {key.replace(
            "clip.model.", "model."): value for key, value in state_dict.items()}
        model.load_state_dict(new_state_dict)
        model = model.to(device)

        return model

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
                # inputs = self.processor(
                #     images=batch, return_tensors="pt").to(self.device)
                tensors = [self.transform(i) for i in batch]
                tensors = torch.stack(tensors).to(self.device)

                # outputs = self.features_model(**inputs)
                # outputs = self.features_model(inputs["pixel_values"])
                outputs = self.features_model(tensors)
                # embeddings = torch.mean(
                #     outputs.last_hidden_state, dim=1).cpu().numpy()
                embeddings = outputs
                data.append(embeddings.cpu())

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        # projections = data
        self.cluster_model.fit(projections)
        self.save_model(self.save_path)
        print("Done")

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        # projections = data
        return self.cluster_model.predict(projections)

    def save_model(self, file_path: str) -> None:
        """
        Save the trained cluster model to a file.

        Args:
            file_path (str): Path to save the cluster model.
        """

        data = {
            "reducer": self.reducer,
            "cluster_model": self.cluster_model
        }
        joblib.dump(data, file_path)

    def load_model(self, file_path=None) -> None:
        """
        Load a trained cluster model from a file.

        Args:
            file_path (str): Path to the saved cluster model.
        """

        if (file_path is None):
            file_path = self.save_path

        data = joblib.load(file_path)
        self.reducer = data["reducer"]
        self.cluster_model = data["cluster_model"]
        print("Loaded model from:", file_path)

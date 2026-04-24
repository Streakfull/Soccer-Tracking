import torch
import numpy as np
from training.ModelTrainer import ModelTrainer
from datasets.SimMatches import SimMatches
from datasets.VisualSimMatches import VisualSimMatches
from lutils.general import seed_all
seed_all(111)

trainer = ModelTrainer(dataset_type=VisualSimMatches,
                       options={"tdm_notebook": True})
dataset = trainer.data_loader_handler.dataset
torch.cuda.empty_cache()
model = trainer.model

trainer.train()

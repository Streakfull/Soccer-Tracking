from torch.cuda.amp import autocast
import torchinfo
from pytorchmodels.base_model import BaseModel
from termcolor import cprint
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
from clipreid.timmbackbone import OpenClipModel
import torch
import torch.nn.functional as F
import numpy as np
import random
from clipreid.loss import ClipLoss
import clipreid.metrics_reid as metrics
from torchreid.utils import re_ranking


class Clip(BaseModel):
    def __init__(self, configs, train_config):
        super().__init__()
        self.configs = configs

        self.clip = OpenClipModel(self.configs["name"],
                                  self.configs["pretrained"],
                                  True)

        self.clip_loss = ClipLoss("cuda:0")
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = OneCycleLR(
            optimizer=self.optimizer,
            max_lr=configs["lr"],
            total_steps=train_config['n_epochs'] * train_config['save_every'],
            pct_start=0.05,
            anneal_strategy='cos',
            div_factor=100,
            final_div_factor=1000
        )
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2.**10)

    def prepare_query_gallery(self, batch):
        """Organizes batch into query and gallery sets ensuring each query has a positive."""
        images = batch["img"]  # Shape: [B, C, H, W]
        pids = batch["pid"]  # Shape: [B]

        unique_pids = torch.unique(pids)  # Unique player IDs
        I_q, I_g = [], []
        pids_q, pids_g = [], []

        for pid in unique_pids:
            indices = (pids == pid).nonzero(as_tuple=True)[
                0]  # Find indices of this PID

            if len(indices) < 2:
                continue  # Skip if only one sample (no positive match)

            # Randomly select one query image from this PID
            query_idx = random.choice(indices.tolist())
            I_q.append(images[query_idx])
            pids_q.append(pid)

            # Remaining images act as positives in the gallery
            gallery_indices = indices[indices != query_idx]
            gallery_imgs = images[gallery_indices]

            I_g.append(gallery_imgs)  # Store positive matches
            pids_g.extend([pid] * len(gallery_imgs))  # Ensure PIDs are aligned

        # Convert lists to tensors
        I_q = torch.stack(I_q)  # [num_queries, C, H, W]
        I_g = torch.cat(I_g, dim=0) if I_g else torch.tensor(
            [])  # [num_gallery, C, H, W]
        pids_q = torch.tensor(pids_q)
        pids_g = torch.tensor(pids_g)

        return I_q, I_g, pids_q, pids_g

    def get_query_gallery_pairs(self, x1_images, x2_images, labels):
        """ 
        Helper function to process matching and mismatching pairs of images and assign unique IDs.
        - x1_images: The first image in the pair (query).
        - x2_images: The second image in the pair (gallery).
        - labels: 1 for match, 0 for mismatch.
        """
        I_q, I_g = [], []
        ids_q, ids_g = [], []  # Initialize empty lists for query and gallery IDs
        current_id = 0  # Counter to assign IDs

        for idx in range(len(labels)):
            # Match (query and gallery should have the same ID)
            if labels[idx] == 1:
                I_q.append(x1_images[idx])
                I_g.append(x2_images[idx])
                ids_q.append(current_id)  # Same ID for match
                ids_g.append(current_id)  # Same ID for match
            # Mismatch (query and gallery should have different IDs)
            elif labels[idx] == 0:
                I_q.append(x1_images[idx])
                I_g.append(x2_images[idx])
                ids_q.append(current_id)  # Assign a unique ID to query
                current_id += 1  # Increment ID for mismatch
                # Assign a unique ID to gallery for mismatch
                ids_g.append(current_id)
                current_id += 1  # Increment ID for mismatch

        # Convert lists to tensors
        I_q = torch.stack(I_q)  # [num_queries, C, H, W]
        I_g = torch.stack(I_g)  # [num_gallery, C, H, W]
        ids_q = torch.tensor(ids_q)
        ids_g = torch.tensor(ids_g)

        return I_q, I_g, ids_q, ids_g

    def forward(self, x):
        self.q, self.g, self.ids_g, self.ids_q = x["q"], x["g"], x["ids_g"], x["ids_q"]
        self.x1, self.x2 = self.clip(self.q, self.g)
        return self.x1, self.x2

    def set_loss(self):
        # self.loss, self.cont, self.nce = self.clip_loss(
        #     self.x1, self.x2, self.ids_q, self.ids_g, self.clip.model.logit_scale.exp())
        self.loss = self.clip_loss(
            self.x1, self.x2, self.ids_q, self.ids_g, self.clip.model.logit_scale.exp())

    def backward(self):
        self.set_loss()
        self.scaler.scale(self.loss).backward()

    def get_batch_input(self, batch):
        # Check if the batch contains x1, x2, and label (the new dataset format)
        if "x1_img" in batch and "x2_img" in batch and "label" in batch:
            x1_images = batch["x1_img"]  # [B, C, H, W] (Image 1)
            x2_images = batch["x2_img"]  # [B, C, H, W] (Image 2)
            labels = batch["label"]  # [B] (1 for match, 0 for mismatch)

            # Use the helper function to get query-gallery pairs and their IDs
            I_q, I_g, ids_q, ids_g = self.get_query_gallery_pairs(
                x1_images, x2_images, labels)

            return {
                "q": I_q.to("cuda:0"),
                "g": I_g.to("cuda:0"),
                "ids_q": ids_q.to("cuda:0"),  # IDs for query images
                "ids_g": ids_g.to("cuda:0"),  # IDs for gallery images
                # Assuming camid is still in the batch
                "camid":  torch.zeros(I_q.size(0), dtype=torch.long).to("cuda:0"),
            }
        else:
            # If the batch does not have x1, x2, label keys (fallback to default logic)
            I_q, I_g, pids_q, pids_g = self.prepare_query_gallery(batch)
            return {
                "q": I_q.to("cuda:0"),
                "g": I_g.to("cuda:0"),
                "ids_g": pids_g.to("cuda:0"),
                "ids_q": pids_q.to("cuda:0"),
                "camid": batch["camid"].to("cuda:0"),
            }

    def get_batch_val_input(self, batch_q, batch_g):
        q, ids_q = batch_q["img"], batch_q["pid"]  # Query images and IDs
        g, ids_g = batch_g["img"], batch_g["pid"]  # Gallery images and IDs
        ids_q = [int(pid) for pid in ids_q]  # ✅ Convert strings to integers
        ids_g = [int(pid) for pid in ids_g]  # ✅ Convert strings to integers
        return {
            "q": q.to("cuda:0"),
            "g": g.to("cuda:0"),
            "ids_g": torch.tensor(ids_g, dtype=torch.long, device="cuda:0"),
            "ids_q": torch.tensor(ids_q, dtype=torch.long, device="cuda:0"),
        }

    def inference(self, x):
        self.eval()
        x = self.forward(x)
        return x

    def step(self, x):
        with autocast():
            self.train()
            self.optimizer.zero_grad()
            x = self.forward(x)
            self.backward()
            # self.optimizer.step()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.update_lr()

    def get_metrics(self):
        with torch.no_grad():
            cosp, cosn = self.compute_avg_similarity()

        return {'loss': self.loss,
                'cosp': cosp,  # cos p,
                'cosn': cosn,  # cosn,
                # "cont": self.cont.detach(),
                # "nce": self.nce.detach()
                }

    def compute_avg_similarity(self):
        x1, x2, ids_q, ids_g = self.x1, self.x2, self.ids_q, self.ids_g
        """
        Computes the average cosine similarity between positive and negative pairs.

        Args:
            x1 (torch.Tensor): Query embeddings of shape [N, D]
            x2 (torch.Tensor): Gallery embeddings of shape [N, D]
            ids_q (torch.Tensor): IDs corresponding to x1 (query)
            ids_g (torch.Tensor): IDs corresponding to x2 (gallery)

        Returns:
            avg_positive_sim (float): Average cosine similarity between positive pairs
            avg_negative_sim (float): Average cosine similarity between negative pairs
        """
        # Compute pairwise cosine similarity (N x N)
        cosine_sim = F.cosine_similarity(x1.unsqueeze(
            1), x2.unsqueeze(0), dim=-1)  # Shape: (N, N)

        # Create mask for positive and negative pairs
        positive_mask = ids_q.unsqueeze(1) == ids_g.unsqueeze(
            0)  # True where IDs match (positive pairs)
        negative_mask = ~positive_mask  # Inverse for negative pairs

        # Compute average similarity for positives
        pos_sims = cosine_sim[positive_mask]
        avg_positive_sim = pos_sims.mean().item(
        ) if pos_sims.numel() > 0 else 0.0  # Avoid division by zero

        # Compute average similarity for negatives
        neg_sims = cosine_sim[negative_mask]
        avg_negative_sim = neg_sims.mean().item(
        ) if neg_sims.numel() > 0 else 0.0  # Avoid division by zero

        return avg_positive_sim, avg_negative_sim

    def prepare_visuals(self):
        """
        Prepares and returns two tensors: one containing all query images and
        another containing all gallery images, stacked together.

        Returns:
            query_images_tensor (torch.Tensor or None): Stacked query images
            gallery_images_tensor (torch.Tensor or None): Stacked gallery images
        """
        if not hasattr(self, 'q') or self.q is None or len(self.q) == 0:
            print("No query images to visualize")
            query_images_tensor = None
        else:
            query_images_tensor = self.q if isinstance(
                self.q, torch.Tensor) else torch.stack(self.q)

        if not hasattr(self, 'g') or self.g is None or len(self.g) == 0:
            print("No gallery images to visualize")
            gallery_images_tensor = None
        else:
            gallery_images_tensor = self.g if isinstance(
                self.g, torch.Tensor) else torch.stack(self.g)

        return query_images_tensor, gallery_images_tensor

    def name(self):
        return 'Clip'

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids

    def extract_features(self, imgs):
        imgs = self.clip(imgs)
        return imgs

    def test(
        self,
        dataloader,
        dist_metric='cosine',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        eval_metric='soccernetv3',
        ranks=[1, 5, 10, 20],
        rerank=False,
        export_ranking_results=False
    ):
        r"""Tests model on target datasets.

        .. note::

            This function has been called in ``run()``.

        .. note::

            The test pipeline implemented in this function suits both image- and
            video-reid. In general, a subclass of Engine only needs to re-implement
            ``extract_features()`` and ``parse_data_for_eval()`` (most of the time),
            but not a must. Please refer to the source code for more details.
        """
        self.eval()
        targets = list(dataloader.keys())
        last_rank1 = 0
        mAP = 0
        for name in targets:
            domain = 'target'
            print('##### Evaluating {} ({}) #####'.format(name, domain))
            query_loader = dataloader[name]['query']
            gallery_loader = dataloader[name]['gallery']
            rank1, mAP = self._evaluate(
                dataset_name=name,
                query_loader=query_loader,
                gallery_loader=gallery_loader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                eval_metric=eval_metric,
                ranks=ranks,
                rerank=rerank,
                export_ranking_results=export_ranking_results
            )

            # if self.writer is not None and rank1 is not None and mAP is not None:
            #     self.writer.add_scalar(f'Test/{name}/rank1', rank1, self.epoch)
            #     self.writer.add_scalar(f'Test/{name}/mAP', mAP, self.epoch)
            if rank1 is not None:
                last_rank1 = rank1

        return last_rank1, mAP

    @torch.no_grad()
    def _evaluate(
        self,
        dataset_name='',
        query_loader=None,
        gallery_loader=None,
        dist_metric='cosine',
        normalize_feature=False,
        visrank=False,
        visrank_topk=10,
        save_dir='',
        eval_metric='soccernetv3',
        ranks=[1, 5, 10, 20],
        rerank=False,
        export_ranking_results=False
    ):

        def _feature_extraction(data_loader):
            f_, pids_, camids_ = [], [], []
            for batch_idx, data in enumerate(data_loader):
                imgs, pids, camids = self.parse_data_for_eval(data)
                if True:
                    imgs = imgs.cuda()

                features = self.extract_features(imgs)
                features = features.cpu().clone()
                f_.append(features)
                pids_.extend(pids)
                camids_.extend(camids)
            f_ = torch.cat(f_, 0)
            pids_ = np.asarray(pids_)
            camids_ = np.asarray(camids_)
            return f_, pids_, camids_

        print('Extracting features from query set ...')
        qf, q_pids, q_camids = _feature_extraction(query_loader)
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        gf, g_pids, g_camids = _feature_extraction(gallery_loader)
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            qf = F.normalize(qf, p=2, dim=1)
            gf = F.normalize(gf, p=2, dim=1)

        print(
            'Computing distance matrix with metric={} ...'.format(
                dist_metric)
        )
        distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
        distmat = distmat.numpy()

        if rerank:
            print('Applying person re-ranking ...')
            distmat_qq = metrics.compute_distance_matrix(
                qf, qf, dist_metric)
            distmat_gg = metrics.compute_distance_matrix(
                gf, gf, dist_metric)
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        if export_ranking_results:
            self.export_ranking_results_for_ext_eval(
                distmat, q_pids, q_camids, g_pids, g_camids, save_dir, dataset_name)

        if not query_loader.dataset.hidden_labels:
            print('Computing CMC and mAP ...')
            cmc, mAP = metrics.evaluate_rank(
                distmat,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric
            )
            print('** Results **')
            print('mAP: {:.1%}'.format(mAP))
            print('CMC curve')
            for r in ranks:
                print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
            return cmc[0], mAP
        else:
            print("Couldn't compute CMC and mAP because of hidden identity labels.")
            return None, None

    def parse_data_for_eval(self, data):
        imgs = data['img']
        pids = data['pid']
        camids = data['camid']
        return imgs, pids, camids

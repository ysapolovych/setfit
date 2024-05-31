from typing import Optional, List, Union
import numpy as np
import pandas as pd
from setfit import SetFitModel
from setfit.modeling import SetFitHead
from setfit.data import SetFitDataset
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm, trange

from setfit.augmentations import mixup, cutmix, mixup_np, cutmix_np, emb_aug_np_batch


def prepare_optimizer(
    head_learning_rate: float,
    body_learning_rate: float,
    l2_weight: float,
    model_body,
    model_head
) -> torch.optim.Optimizer:

    body_learning_rate = body_learning_rate or head_learning_rate
    l2_weight = l2_weight or 1e-2
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model_body.parameters(),
                "lr": body_learning_rate,
                "weight_decay": l2_weight,
            },
            {"params": model_head.parameters(), "lr": head_learning_rate, "weight_decay": l2_weight},
        ],
    )

    return optimizer


def prepare_dataloader(
        x_train: List[str],
        y_train: Union[List[int], List[List[int]]],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        shuffle: bool = True,
        setfit_model = None
    ) -> DataLoader:
    max_acceptable_length = 512
    if max_length is None:
        max_length = max_acceptable_length
    if max_length > max_acceptable_length:
        max_length = max_acceptable_length

    dataset = SetFitDataset(
        x_train,
        y_train,
        tokenizer=setfit_model.model_body.tokenizer,
        max_length=max_length,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    return dataloader


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def collate_fn_custom(batch):
    x_batch, y_batch = zip(*batch)
    return torch.stack(x_batch), torch.stack(y_batch)


def prepare_dataloader_custom(x, y, batch_size):
    dataset = CustomDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_custom)


class NN_Trainer:
    def __init__(self, x_train: list[str], y_train: np.ndarray, setfit_model, params, optimizer_params, use_mixup, use_cutmix, alpha,
                 n_epochs: int = 24, batch_size: int = 2, device: str = 'cuda:0', **kwargs):
        self.x_train = x_train
        self.y_train = pd.get_dummies(y_train, dtype='float').to_numpy()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.setfit_model = setfit_model

        self.classifier = SetFitHead(**params)
        self.init_dataloader = prepare_dataloader(self.x_train, self.y_train, self.batch_size, 512, False, self.setfit_model)
        self.criterion = self.classifier.get_loss_fn()
        self.optimizer = prepare_optimizer(optimizer_params['head_learning_rate'],
                                           optimizer_params['body_learning_rate'],
                                           optimizer_params['l2_weight'],
                                           self.setfit_model.model_body,
                                           self.classifier)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.alpha = alpha

        self.device = device
        self.normalize_embeddings = False

    def fit(self, emb_aug_multipl: int = 1):

        x = []
        y = []

        for batch in self.init_dataloader:
            with torch.no_grad():
                features, labels = batch
                features = {k: v.to(self.device) for k, v in features.items()}

                labels = labels.to(self.device)

                outputs = self.setfit_model.model_body(features)

                x.append(outputs["sentence_embedding"])
                y.append(labels)

        x = torch.concat(x)
        y = torch.concat(y)

        aug_x, aug_y = emb_aug_np_batch(x, y, emb_aug_multipl, self.alpha, self.use_mixup, self.use_cutmix)

        x = torch.concat([x, aug_x])
        y = torch.concat([y, aug_y])

        dataloader = prepare_dataloader_custom(x, y, 2)

        for epoch_idx in trange(self.n_epochs, desc="Epoch", disable=not True):
            for batch in tqdm(dataloader, desc="Iteration", disable=not True, leave=False):
                features, labels = batch
                self.optimizer.zero_grad()

                if self.normalize_embeddings:
                        features = nn.functional.normalize(
                            features, p=2, dim=1
                        )

                outputs = self.classifier(features)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs['logits']

                loss: torch.Tensor = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()


def train_lr_classifier(train, setfit_model):
    classifier = LogisticRegression()

    y = train['target'].tolist()
    print(len(train['train_text'].tolist()))
    embeddings = setfit_model.encode(train['train_text'].tolist(), show_progress_bar=True)

    classifier.fit(embeddings, y)

    return classifier
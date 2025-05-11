import json
import os
import time

from typing import Any

import matplotlib.pyplot as plt

import fasttext
import easyocr
import fasttext.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.__version__)


class ConTextTransformer(nn.Module):
    def __init__(
        self, *, image_size, num_classes, dim, depth, heads, mlp_dim, channels=3
    ):
        super().__init__()

        resnet50 = torchvision.models.resnet50(pretrained=True)
        modules = list(resnet50.children())[:-2]
        self.resnet50 = nn.Sequential(*modules)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        self.num_cnn_features = 64  # 8x8
        self.dim_cnn_features = 2048
        self.dim_fasttext_features = 300

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_cnn_features + 1, dim)
        )
        self.cnn_feature_to_embedding = nn.Linear(self.dim_cnn_features, dim)
        self.fasttext_feature_to_embedding = nn.Linear(self.dim_fasttext_features, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True
        )
        encoder_norm = nn.LayerNorm(dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, txt, mask=None):
        x = self.resnet50(img)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.cnn_feature_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding

        x2 = self.fasttext_feature_to_embedding(txt.float())
        x = torch.cat((x, x2), dim=1)

        # tmp_mask = torch.zeros((img.shape[0], 1+self.num_cnn_features), dtype=torch.bool)
        # mask = torch.cat((tmp_mask.to(device), mask), dim=1)
        # x = self.transformer(x, src_key_padding_mask=mask)
        x = self.transformer(x)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)


def default_loader(path: str) -> Any:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class ConTextDataset(Dataset):
    def __init__(
        self, images_dir, labels_txt_file, transform=None, loader=default_loader
    ):
        self.images_dir = images_dir
        self.targets_file = labels_txt_file
        self.transform = transform
        self.loader = loader

        self.img_ids: list[str] = []
        self.targets: list[int] = []

        with open(labels_txt_file, "r") as f:
            for line in f:
                filename, class_label = line.strip().split()
                img_path = os.path.join(str(self.images_dir), filename)
                self.img_ids.append(img_path)
                self.targets.append(int(class_label))

        assert len(self.img_ids) == len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx) -> tuple[Any, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = self.img_ids[idx]
        img = self.loader(img_id)
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx] + 1


def dataloader_collate(batch):
    images = []
    labels = []

    for item in batch:
        img = item[0]
        label = torch.tensor(item[1])

        images.append(img)
        labels.append(label)


    images_batch = torch.stack(images, 0)
    labels_batch = torch.stack(labels, 0)

    return images_batch, labels_batch


category_mapping = {
    "Bakery": 1,
    "Barber": 2,
    "Bistro": 3,
    "Bookstore": 4,
    "Cafe": 5,
    "ComputerStore": 6,
    "CountryStore": 7,
    "Diner": 8,
    "DiscounHouse": 9,
    "Dry Cleaner": 10,
    "Funeral": 11,
    "Hotspot": 12,
    "MassageCenter": 13,
    "MedicalCenter": 14,
    "PackingStore": 15,
    "PawnShop": 16,
    "PetShop": 17,
    "Pharmacy": 18,
    "Pizzeria": 19,
    "RepairShop": 20,
    "Restaurant": 21,
    "School": 22,
    "SteakHouse": 23,
    "Tavern": 24,
    "TeaHouse": 25,
    "Theatre": 26,
    "Tobacco": 27,
    "Motel": 28,
}

inverted_category_mapping = {str(value): key for key, value in category_mapping.items()}


input_size = 256
data_transforms_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(input_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
data_transforms_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(input_size),
        torchvision.transforms.CenterCrop(input_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

images_dir = "data/images/"

train_set = ConTextDataset(images_dir, "data/train.txt", transform=data_transforms_train)
val_set = ConTextDataset(images_dir, "data/val.txt", transform=data_transforms_test)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True, num_workers=10, collate_fn=dataloader_collate,
)
test_loader = torch.utils.data.DataLoader(
    val_set, batch_size=64, shuffle=False, num_workers=10, collate_fn=dataloader_collate,
)

reader = easyocr.Reader(["en"])

def detect_text(img_filename):
    """Detects text in the image using EasyOCR"""
    results = reader.readtext(img_filename)

    words, boxes, confs = [], [], []
    for res in results:
        words.append(res[1])
        boxes.append(res[0])
        confs.append(res[2])
    return words, boxes, confs


def train_epoch(model, optimizer, data_loader, loss_history):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data_img, target) in enumerate(data_loader):
        data_img = data_img.to(device)
        target = target.to(device)

        cpu_images = data_img.cpu().numpy()
        batch_tokens = []
        for img in cpu_images:
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)

            detections = reader.readtext(img)

            tokens = [det[1] for det in detections]
            batch_tokens.append(tokens)

        batch_tokens = torch.tensor(batch_tokens)
        batch_tokens.to(device)

        optimizer.zero_grad()
        output = F.log_softmax(model(data_img, batch_tokens), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                "["
                + "{:5}".format(i * len(data_img))
                + "/"
                + "{:5}".format(total_samples)
                + " ("
                + "{:3.0f}".format(100 * i / len(data_loader))
                + "%)]  Loss: "
                + "{:6.4f}".format(loss.item())
            )
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data_img, data_txt, txt_mask, target in data_loader:
            data_img = data_img.to(device)
            data_txt = data_txt.to(device)
            txt_mask = txt_mask.to(device)
            target = target.to(device)
            output = F.log_softmax(model(data_img, data_txt, txt_mask), dim=1)
            loss = F.nll_loss(output, target, reduction="sum")
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print(
        "\nAverage test loss: "
        + "{:.4f}".format(avg_loss)
        + "  Accuracy:"
        + "{:5}".format(correct_samples)
        + "/"
        + "{:5}".format(total_samples)
        + " ("
        + "{:4.2f}".format(100.0 * correct_samples / total_samples)
        + "%)\n"
    )

    return correct_samples / total_samples


N_EPOCHS = 50
start_time = time.time()

model = ConTextTransformer(
    image_size=input_size,
    num_classes=28,
    channels=3,
    dim=256,
    depth=2,
    heads=4,
    mlp_dim=512,
)
model.to(device)
params_to_update = []

for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, lr=0.0001)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[15, 30], gamma=0.1
)

train_loss_history, test_loss_history = [], []
best_acc = 0.0

for epoch in range(1, N_EPOCHS + 1):
    print("Epoch:", epoch)
    train_epoch(model, optimizer, train_loader, train_loss_history)
    acc = evaluate(model, test_loader, test_loss_history)
    if acc > best_acc:
        torch.save(model.state_dict(), "all_best.pth")
    scheduler.step()

print("Execution time:", "{:5.2f}".format(time.time() - start_time), "seconds")

import glob
import os.path as path
import pickle
import random
import torch
import torchvision
import torchvision.models.detection as det
import torchvision.transforms as transforms
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"


class PetDataset(Dataset):

    def __init__(self,
                 root_dir,
                 xforms=transforms.Compose([]),
                 yforms=transforms.Compose([]),
                 augs=transforms.Compose([])):
        self.ann_dir = path.join(root_dir, "annotations", "trimaps")
        self.image_dir = path.join(root_dir, "images")
        self.image_files = glob.glob(path.join(self.image_dir, "*"))
        self.image_files = [
            x for x in self.image_files if path.splitext(x)[1] == ".jpg"
        ]
        self.image_files = [
            x for x in self.image_files if Image.open(x).format == "JPEG"
        ]
        self.image_files = [
            x for x in self.image_files if Image.open(x).mode == "RGB"
        ]
        self.last_mrcnn_idx = 0
        unique_breeds = set([
            path.basename("_".join(fname.split("_")[:-1]))
            for fname in self.image_files
        ])
        self.breed_assoc = {
            x: self.last_mrcnn_idx + idx
            for idx, x in enumerate(sorted(list(unique_breeds)))
        }
        self.num_classes = max([v for k, v in self.breed_assoc.items()]) + 1
        self.xforms = xforms
        self.yforms = yforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        imf = self.image_files[idx]
        bname = path.basename(path.splitext(imf)[0])
        ann = path.join(self.ann_dir, bname) + ".png"

        image = self.xforms(Image.open(imf)).to(device)

        mask = self.yforms(Image.open(ann)).to(device)
        unnormed_mask = (mask * 300).floor()
        unnormed_boundary = (unnormed_mask == 3.0).to(torch.float)
        unnormed_interior = (unnormed_mask == 1.0).to(torch.float)
        unnormed_exterior = (unnormed_mask == 2.0).to(torch.float)
        mask = 0.5 * unnormed_boundary + unnormed_interior

        category = path.basename("_".join(imf.split("_")[:-1]))
        labels = torch.tensor([self.breed_assoc[category]
                               ]).to(torch.int64).to(device)

        indices = torch.nonzero(mask.squeeze())

        if indices.numel() == 0:
            left_x = 0
            bottom_y = 0
            right_x = 224
            top_y = 224

        else:
            bottom_y = indices[:, 0].min()
            top_y = indices[:, 0].max()
            right_x = indices[:, 1].max()
            left_x = indices[:, 1].min()

        boxes = torch.tensor([left_x, bottom_y, right_x,
                              top_y]).unsqueeze(0).to(device)

        return image, {"boxes": boxes, "labels": labels, "masks": mask}


def gen_dataset(ds_path="."):

    transformx = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transformy = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()])
    augs = transforms.Compose([])
    ds = PetDataset(ds_path, transformx, transformy, augs)
    train_len = int(0.8 * len(ds))
    test_len = len(ds) - train_len

    train_dataset, test_dataset = random_split(
        ds, [train_len, test_len], generator=torch.Generator().manual_seed(42))
    return ds, train_dataset, test_dataset


def gen_loaders(train_dataset, test_dataset, batch_size=2):

    def custom_collate(batch):
        imgs = [b[0] for b in batch]
        targets = [b[1] for b in batch]

        return torch.stack(imgs, dim=0), targets

    dl = DataLoader(train_dataset,
                    shuffle=True,
                    collate_fn=custom_collate,
                    batch_size=batch_size)
    tl = DataLoader(test_dataset,
                    collate_fn=custom_collate,
                    batch_size=batch_size)
    return dl, tl


def gen_model(num_classes):
    model = det.maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                      num_classes).to(device)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_dim_reduced = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       mask_dim_reduced,
                                                       num_classes)
    model = model.to(device)
    return model


def run_one_epoch(epoch, model, opt, dl, tl):
    train_loss_arr = []
    test_loss_arr = []
    model.train()

    # train
    for x, y in tqdm.tqdm(dl, total=len(dl)):
        preds = model(x, y)
        losses = sum(loss for loss in preds.values())
        train_loss_arr.append(losses.item())
        opt.zero_grad()
        losses.backward()
        opt.step()

    # test
    with torch.no_grad():
        for x, y in tqdm.tqdm(tl, total=len(tl)):
            preds = model(x, y)
            losses = sum(loss for loss in preds.values())
            test_loss_arr.append(losses.item())

    return train_loss_arr, test_loss_arr


def run_epochs(num_epochs=100):
    fs, ds, ts = gen_dataset()
    dl, tl = gen_loaders(ds, ts)
    model = gen_model(fs.num_classes)
    all_epoch_train_losses = []
    all_epoch_test_losses = []

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(num_epochs):
        train_loss_arr, test_loss_arr = run_one_epoch(epoch, model, opt, dl,
                                                      tl)
        avg_train_loss = np.mean(train_loss_arr)
        avg_test_loss = np.mean(test_loss_arr)
        all_epoch_train_losses.append(avg_train_loss)
        all_epoch_test_losses.append(avg_test_loss)

        print(f"{epoch} train loss: {avg_train_loss:0.4f}")
        print(f"{epoch} test loss: {avg_test_loss:0.4f}")

        with open("train_losses.pkl", "wb") as fd:
            pickle.dump(all_epoch_train_losses, fd)

        with open("test_losses.pkl", "wb") as fd:
            pickle.dump(all_epoch_test_losses, fd)
        torch.save(model, f"checkpoint_{epoch}")


if __name__ == "__main__":
    run_epochs()

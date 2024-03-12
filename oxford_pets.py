#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os.path as path
import glob
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import torchvision
import torchvision.models.detection as det
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[ ]:


class PetDataset(Dataset):
    def __init__(self, root_dir, xforms, yforms):
        self.ann_dir = path.join(root_dir, "annotations", "trimaps")
        self.image_dir = path.join(root_dir, "images")
        self.image_files = glob.glob(path.join(self.image_dir, "*"))
        self.image_files = [x for x in self.image_files if path.splitext(x)[1] == ".jpg"]
        self.image_files = [x for x in self.image_files if Image.open(x).format == "JPEG"]
        self.image_files = [x for x in self.image_files if Image.open(x).mode == "RGB"]
        self.last_mrcnn_idx = 0
        self.breed_assoc = {x: self.last_mrcnn_idx + idx for 
                            idx, x in enumerate(sorted(list(set(
                                [ path.basename('_'.join(fname.split("_")[:-1])) for 
                                 fname in self.image_files]))))}
        self.xforms = xforms
        self.yforms = yforms
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        imf = self.image_files[idx]
        bname = path.basename(path.splitext(imf)[0])
        ann = path.join(self.ann_dir, bname) + '.png'

        # image
        image = self.xforms(Image.open(imf)).to(device)

        # mask
        mask = self.yforms(Image.open(ann)).to(device)

        # label
        category = path.basename('_'.join(imf.split("_")[:-1]))
        labels = torch.tensor([self.breed_assoc[category]]).to(torch.int64).to(device)
        
        # box
        get_edge_pixels = lambda x: ((x* 300 ).floor() - 1) == 2.0
        edge_pixels = get_edge_pixels(mask.squeeze())
        indices = torch.nonzero(edge_pixels)
        
        if indices.numel() == 0:
            left_x = 0
            bottom_y = 0
            right_x = 224
            top_y = 224

        else:
            left_x = indices[:,0].min()
            right_x = indices[:,0].max()
            top_y = indices[:,1].max()
            bottom_y = indices[:,1].min()
            
        boxes = torch.tensor([left_x,bottom_y,right_x,top_y]).unsqueeze(0).to(device)
        
        return image, {"boxes": boxes, "labels": labels, "masks": mask}


# In[ ]:


transformx = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
transformy = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
ds = PetDataset(".", transformx, transformy)

def custom_collate(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]  # Keep targets as a list of dicts
    return torch.stack(imgs, dim=0), targets

dl = DataLoader(ds, shuffle=True, collate_fn=custom_collate, batch_size=4)


test_cut = int(0.8 * len(ds))
train_idxs = [x for x in range(len(ds))]
test_idxs = train_idxs[test_cut:]
train_idxs = train_idxs[:test_cut]
to_pil = transforms.ToPILImage()


# In[ ]:


model = det.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-6)
all_epoch_losses = []


# In[ ]:


num_epochs = 4

for epoch in range(num_epochs):
    loss_arr = []
    for x, y in tqdm.tqdm(dl, total=len(dl)):
        preds = model(x,y)
        losses = sum(loss for loss in preds.values())
        loss_arr.append(losses.item())
        opt.zero_grad()
        losses.backward()
        opt.step()
    all_epoch_losses.append(loss_arr)
    print(sum(loss_arr)/len(loss_arr))


# In[ ]:


# plt.plot(range(len(loss_arr)), loss_arr)


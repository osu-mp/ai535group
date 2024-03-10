import os.path as path
import glob
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PetDataset(Dataset):
    def __init__(self, root_dir, xforms, yforms, train=True, test_size=0.2):
        self.ann_dir = path.join(root_dir, "annotations", "trimaps")
        self.image_dir = path.join(root_dir, "images")
        self.image_files = glob.glob(path.join(self.image_dir, "*"))
        self.image_files = [x for x in self.image_files if path.splitext(x)[1] == ".jpg"]
        self.image_files = [x for x in self.image_files if Image.open(x).format == "JPEG"]
        self.image_files = [x for x in self.image_files if Image.open(x).mode == "RGB"]
        self.last_mrcnn_idx = 0
        self.breed_assoc = {x: self.last_mrcnn_idx + idx for
                            idx, x in enumerate(sorted(list(set(
                [path.basename('_'.join(fname.split("_")[:-1])) for
                 fname in self.image_files]))))}
        self.xforms = xforms
        self.yforms = yforms

        if train:
            self.image_files = self.image_files[:int(len(self.image_files) * (1 - test_size))]
        else:
            self.image_files = self.image_files[int(len(self.image_files) * (1 - test_size)):]

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
        get_edge_pixels = lambda x: ((x * 300).floor() - 1) == 2.0
        edge_pixels = get_edge_pixels(mask.squeeze())
        indices = torch.nonzero(edge_pixels)

        if indices.numel() == 0:
            left_x = 0
            bottom_y = 0
            right_x = 224
            top_y = 224

        else:
            left_x = indices[:, 0].min()
            right_x = indices[:, 0].max()
            top_y = indices[:, 1].max()
            bottom_y = indices[:, 1].min()

        boxes = torch.tensor([left_x, bottom_y, right_x, top_y]).unsqueeze(0).to(device)

        return image, {"boxes": boxes, "labels": labels, "masks": mask}

def main():
    transformx = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transformy = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    # Initialize dataset with train/test split
    train_dataset = PetDataset(root_dir=".", xforms=transformx, yforms=transformy, train=True, test_size=0.2)
    test_dataset = PetDataset(root_dir=".", xforms=transformx, yforms=transformy, train=False, test_size=0.2)

    # Print sizes of train and test datasets
    print("Size of train dataset:", len(train_dataset))
    print("Size of test dataset:", len(test_dataset))

if __name__ == "__main__":
    main()
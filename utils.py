import os
#import natsort
import torch
from PIL import Image


class CustomDataSet(torch.utils.data.Dataset):
    def _init_(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs#natsort.natsorted(all_imgs)

    def _len_(self):
        return len(self.total_imgs)

    def _getitem_(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SimpleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

        assert len(self.image_paths) == len(self.labels), "Number of images and labels must be equal"
        assert len(np.shape(self.labels)) == len(np.shape(self.image_paths)), "Dimensionality of labels must match dimensionality of image paths"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if len(np.shape(self.image_paths)) == 1:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            # Load image using PIL
            image = Image.open(image_path)
            # Apply transformations, if specified
            if self.transform:
                image = self.transform(image)

        elif len(np.shape(self.image_paths)) == 2:
            image = np.empty(np.shape(self.image_paths)[1], dtype=object)
            label = np.empty(np.shape(self.image_paths)[1], dtype=object)
            for i in range(np.shape(self.image_paths)[1]):
                image_path = self.image_paths[idx][i]
                label[i] = self.labels[idx][i]
                # Load image using PIL
                image[i] = Image.open(image_path)
                # Apply transformations, if specified
                if self.transform:
                    image[i] = self.transform(image[i])               
        else:
            raise ValueError("Dimensionality of image paths must be 1 or 2")    

        return label
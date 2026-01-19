from PIL import Image
from torch.utils.data import Dataset


class DefaultDataset(Dataset):
    def __init__(self, root):
        self.pixels = []
        self.labels = []

        with open(root) as f:
            # READ here
            print("TO BE CREATED")

    def __getitem__(self, index):
        label = self.labels[index]
        pixel = self.pixels[index]
        return {
            'index': index,
            'pixel': pixel,
            'label': label
        }
    
    def __len__(self):
        return len(self.pixels)
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class DogBreedDataset(Dataset):
    def __init__(self, root_dir, labels_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels = self.load_labels(labels_file)
        self.classes = sorted(list(set(self.labels['breed'])))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # # Print class-to-index mapping
        # print("Class to Index Mapping:")
        # for cls, idx in self.class_to_idx.items():
        #     print(f"{cls}: {idx}")

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_id = self.labels.iloc[idx, 0]
        img_path = f'{self.root_dir}/train/{img_id}.jpg'
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[self.labels.iloc[idx, 1]]
        
        return image, label
    
    def load_labels(self, labels_file):
        return pd.read_csv(labels_file)

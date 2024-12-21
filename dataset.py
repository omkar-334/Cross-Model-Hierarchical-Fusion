import os

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Constants
SEQ_LEN = 128
FEATURE_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiModalDataset(Dataset):
    def __init__(self, folder_path):
        # self.text_features = text_features
        # self.audio_features = audio_features
        # self.video_features = video_features
        # self.labels = labels
        self.folder_path = folder_path
        self.file_names = self.get_file_names()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def get_file_names(self):
        files = os.listdir(self.folder_path)
        file_names = set([os.path.splitext(f)[0] for f in files])
        return sorted(file_names)

    def preprocess_text(self, text_file):
        with open(text_file, "r") as f:
            text = f.read().strip()
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=SEQ_LEN,
            return_tensors="pt",
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def preprocess_audio(self, file):
        pass

    def preprocess_video(self, file):
        pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.text_features[idx], self.audio_features[idx], self.video_features[idx], self.labels[idx])

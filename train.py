import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MultiModalDataset
from model import MultiModalSentimentModel, compute_loss

# Constants
SEQ_LEN_TEXT = 128
SEQ_LEN_AUDIO = 128
SEQ_LEN_VIDEO = 128
EMBED_DIM_TEXT = 768
EMBED_DIM_AUDIO = 128
EMBED_DIM_VIDEO = 128
HIDDEN_DIM = 256
NUM_HEADS = 8
OUTPUT_DIM = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for text_input, audio_input, video_input, labels in dataloader:
            text_input, audio_input, video_input, labels = (
                text_input.to(DEVICE),
                audio_input.to(DEVICE),
                video_input.to(DEVICE),
                labels.to(DEVICE),
            )
            optimizer.zero_grad()
            y_unimodal, y_bimodal, y_global, y_fusion = model(text_input, audio_input, video_input)
            loss = compute_loss(y_unimodal, y_bimodal, y_global, y_fusion, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


if __name__ == "__main__":
    num_samples = 100
    text_data = None
    audio_data = None
    video_data = None
    labels = None

    dataset = MultiModalDataset(text_data, audio_data, video_data, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MultiModalSentimentModel().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_model(model, dataloader, optimizer, num_epochs=5)

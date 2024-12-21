import torch
import torch.nn as nn

from audio_video import AudioVideoFeatureExtractor
from text import TextFeatureExtractor

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


# for bimodal and trimodal (global) fusion
class TensorFusionNetwork(nn.Module):
    def __init__(self):
        super(TensorFusionNetwork, self).__init__()

    def forward(self, modality1, modality2):
        # Cartesian product
        fused = torch.einsum("bi,bj->bij", modality1, modality2)  # Shape: (B, D1, D2)
        return fused.view(fused.size(0), -1)  # Shape: (B, D1 * D2)


# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        return attn_output


# Final Multi-Task Sentiment Model
class MultiModalSentimentModel(nn.Module):
    def __init__(self):
        super(MultiModalSentimentModel, self).__init__()
        self.text_extractor = TextFeatureExtractor()
        self.audio_extractor = AudioVideoFeatureExtractor(EMBED_DIM_AUDIO, HIDDEN_DIM)
        self.video_extractor = AudioVideoFeatureExtractor(EMBED_DIM_VIDEO, HIDDEN_DIM)
        self.tfn = TensorFusionNetwork()
        self.attention = MultiHeadAttention(embed_dim=3 * HIDDEN_DIM, num_heads=NUM_HEADS)
        self.fc_unimodal = nn.Linear(3 * HIDDEN_DIM, OUTPUT_DIM)
        self.fc_bimodal = nn.Linear(HIDDEN_DIM**2, OUTPUT_DIM)
        self.fc_global = nn.Linear(3 * HIDDEN_DIM, OUTPUT_DIM)
        self.fc_fusion = nn.Linear(3 * OUTPUT_DIM, OUTPUT_DIM)

    def forward(self, text_input, audio_input, video_input):
        # Extract features
        text_features = self.text_extractor(text_input)  # Shape: (B, SEQ_LEN_TEXT, 2*HIDDEN_DIM)
        audio_features = self.audio_extractor(audio_input)  # Shape: (B, HIDDEN_DIM)
        video_features = self.video_extractor(video_input)  # Shape: (B, HIDDEN_DIM)

        # Unimodal Fusion

        f_unimodal = torch.cat(
            [text_features[:, -1, :], audio_features, video_features], dim=1
        )  # Shape: (B, 3*HIDDEN_DIM)

        # Bimodal Fusion

        f_text_audio = self.tfn(text_features[:, -1, :], audio_features)  # Shape: (B, HIDDEN_DIM**2)
        f_audio_video = self.tfn(audio_features, video_features)
        f_video_text = self.tfn(video_features, text_features[:, -1, :])
        f_bimodal = torch.cat([f_text_audio, f_audio_video, f_video_text], dim=1)  # Shape: (B, HIDDEN_DIM**2)

        # Global Fusion with Multi-Head Attention

        f_global = torch.cat([f_text_audio, f_audio_video, f_video_text], dim=1).unsqueeze(
            1
        )  # Shape: (B, 1, HIDDEN_DIM**2)
        f_global = self.attention(f_global, f_global, f_global).squeeze(1)  # Shape: (B, 3*HIDDEN_DIM)

        # Predictions

        y_unimodal = self.fc_unimodal(f_unimodal)
        y_bimodal = self.fc_bimodal(f_bimodal)
        y_global = self.fc_global(f_global)

        # Final Fusion

        f_fusion = torch.cat([y_unimodal, y_bimodal, y_global], dim=1)  # Shape: (B, 3)
        y_fusion = self.fc_fusion(f_fusion)  # Shape: (B, OUTPUT_DIM)

        return y_unimodal, y_bimodal, y_global, y_fusion


def compute_loss(y_unimodal, y_bimodal, y_global, y_fusion, labels):
    loss_fn = nn.BCEWithLogitsLoss()
    loss = (
        loss_fn(y_unimodal, labels) + loss_fn(y_bimodal, labels) + loss_fn(y_global, labels) + loss_fn(y_fusion, labels)
    )
    return loss

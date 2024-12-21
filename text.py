import torch
import torch.nn as nn
from transformers import BertModel

EMBED_DIM_TEXT = 768
HIDDEN_DIM = 256


# BERT and  Bidirectional GRU
class TextFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextFeatureExtractor, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bigru = nn.GRU(
            input_size=EMBED_DIM_TEXT,
            hidden_size=HIDDEN_DIM,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, text_input):
        with torch.no_grad():
            bert_output = self.bert(text_input)["last_hidden_state"]
            # Shape: (B, SEQ_LEN_TEXT, EMBED_DIM_TEXT)
        bigru_output, _ = self.bigru(bert_output)
        # Shape: (B, SEQ_LEN_TEXT, 2*HIDDEN_DIM)
        return bigru_output

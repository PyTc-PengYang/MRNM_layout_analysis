import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEmbedding(nn.Module):
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', embedding_dim=1024):
        super(TextEmbedding, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self.projection = nn.Linear(self.model.config.hidden_size, embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.attention_fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.attention_fc2 = nn.Linear(embedding_dim, 1)

    def forward(self, texts, is_paragraph=False):
        if not texts:
            return torch.zeros((0, self.projection.out_features), device=self.projection.weight.device)

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state

        if is_paragraph:
            attention_scores = self.attention_fc2(torch.tanh(self.attention_fc1(token_embeddings)))
            attention_weights = torch.softmax(attention_scores, dim=1)
            text_embedding = torch.sum(token_embeddings * attention_weights, dim=1)
        else:
            mask = inputs['attention_mask'].unsqueeze(-1)
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1)
            text_embedding = sum_embeddings / sum_mask

        text_embedding = self.projection(text_embedding)
        text_embedding = self.relu(text_embedding)
        text_embedding = self.layer_norm(text_embedding)

        return text_embedding

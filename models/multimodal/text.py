import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEmbedding(nn.Module):
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', embedding_dim=1024):
        super(TextEmbedding, self).__init__()

        # 加载预训练的Sentence-BERT模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # 输出投影层
        self.projection = nn.Linear(self.model.config.hidden_size, embedding_dim)
        self.relu = nn.ReLU(inplace=True)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 用于计算注意力权重的层
        self.attention_fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.attention_fc2 = nn.Linear(embedding_dim, 1)

    def forward(self, texts, is_paragraph=False):
        if not texts:
            return torch.zeros((0, self.projection.out_features), device=self.projection.weight.device)

        # 对文本进行分词
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 获取BERT的输出
        outputs = self.model(**inputs)
        token_embeddings = outputs.last_hidden_state

        if is_paragraph:
            # 对于段落，使用注意力机制
            attention_scores = self.attention_fc2(torch.tanh(self.attention_fc1(token_embeddings)))
            attention_weights = torch.softmax(attention_scores, dim=1)
            text_embedding = torch.sum(token_embeddings * attention_weights, dim=1)
        else:
            # 对于单个文本行，使用平均池化
            # 创建一个掩码来排除填充标记
            mask = inputs['attention_mask'].unsqueeze(-1)
            sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
            sum_mask = torch.sum(mask, dim=1)
            text_embedding = sum_embeddings / sum_mask

        # 投影到所需的嵌入维度
        text_embedding = self.projection(text_embedding)
        text_embedding = self.relu(text_embedding)
        text_embedding = self.layer_norm(text_embedding)

        return text_embedding
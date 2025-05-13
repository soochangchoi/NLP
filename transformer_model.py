import torch
import torch.nn as nn
import math
import numpy as np

# ===== 문자 사전 (고정된 vocab 사용) =====
vocab_size = 979  # 학습된 모델에 맞춰 고정

# 필요한 경우 수동으로 char2idx / idx2char 정의
# 여기선 예시로 placeholder
char2idx = {chr(i): i for i in range(vocab_size)}  # 예시용 placeholder
idx2char = {i: chr(i) for i in range(vocab_size)}  # 실제 학습 시 사용한 문자셋 필요

# ===== Positional Encoding =====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ===== Transformer 모델 (이름 맞춤) =====
class TransformerPoet(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, ff_dim=512, max_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)  # 이름 변경
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory=None):
        tgt_embed = self.embedding(tgt)
        tgt_embed = self.pos_encoder(tgt_embed)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        output = self.transformer_decoder(tgt_embed.transpose(0, 1), memory=torch.zeros_like(tgt_embed).transpose(0, 1), tgt_mask=tgt_mask)
        return self.fc_out(output.transpose(0, 1))

# ===== 시 생성 함수 =====
def generate_transformer_poem(model, start_text="달빛", length=100, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device
    input_seq = [char2idx.get(ch, 0) for ch in start_text]
    input_tensor = torch.tensor(input_seq).unsqueeze(0).to(device)
    result = start_text

    with torch.no_grad():
        for _ in range(length):
            output = model(input_tensor)
            last_logits = output[0, -1] / temperature
            prob = torch.softmax(last_logits, dim=0).cpu().numpy()
            next_idx = np.random.choice(len(prob), p=prob)
            result += idx2char.get(next_idx, "?")  # unknown idx 방지
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_idx]]).to(device)], dim=1)

    return result

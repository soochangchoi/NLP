import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
import matplotlib.pyplot as plt

# ======================
# 1. 유틸 함수
# ======================
def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_stopwords(path):
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def tokenize_text(text):
    okt = Okt()
    return okt.morphs(text, stem=True)  # 형태소 분석

# ======================
# 2. 전처리
# ======================
text_path = "../project/data/hawsan.txt"
stopwords_path = "../project/data/poetry.txt"

stopwords = load_stopwords(stopwords_path)
raw = load_text(text_path)
tokens = tokenize_text(raw)
filtered = [t for t in tokens if t not in stopwords and len(t) > 1]

# vocab 생성
vocab = sorted(set(filtered))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

data = [word2idx[w] for w in filtered]

# ======================
# 3. 하이퍼파라미터
# ======================
seq_len = 30
embed_dim = 128
hidden_dim = 256
batch_size = 128
num_epochs = 50
early_stop_patience = 3
lr = 0.002

# ======================
# 4. 데이터 분할
# ======================
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_size = int(len(data) * 0.1)
val_data = temp_data[:val_size]
test_data = temp_data[val_size:]

# ======================
# 5. Dataset 클래스
# ======================
class WordDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+self.seq_len], dtype=torch.long)
        return x, y

train_loader = DataLoader(WordDataset(train_data, seq_len), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(WordDataset(val_data, seq_len), batch_size=batch_size)
test_loader = DataLoader(WordDataset(test_data, seq_len), batch_size=batch_size)

# ======================
# 6. LSTM 모델 정의
# ======================
class WordLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = WordLSTM(len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ======================
# 7. 모델 저장 함수
# ======================
def save_model(model, path, word2idx, idx2word, vocab):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'word2idx': word2idx,
        'idx2word': idx2word,
        'vocab': vocab
    }, path)
    print(f"✅ 모델 저장 완료: {path}")

# ======================
# 8. 학습 루프 (중간 손실 저장 포함)
# ======================
train_losses = []
val_losses = []
best_loss = float('inf')
patience = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in val_loader:
            val_output = model(x_val)
            val_loss += criterion(val_output, y_val).item()
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience = 0
        save_model(model, f"models/story_model_epoch{epoch+1}.pth", word2idx, idx2word, vocab)
    else:
        patience += 1
        if patience >= early_stop_patience:
            print("🛑 조기 종료")
            break

# ======================
# 9. 테스트 평가
# ======================
model.eval()
test_loss = 0
with torch.no_grad():
    for x_test, y_test in test_loader:
        output = model(x_test)
        test_loss += criterion(output, y_test).item()
avg_test_loss = test_loss / len(test_loader)

# ======================
# 10. 학습 시각화
# ======================
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("📉 Training & Validation Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("models/loss_curve.png")
plt.show()

# ======================
# 11. 최종 점수 출력
# ======================
print(f"\n✅ 최종 점수:")
print(f"Train Loss: {train_losses[-1]:.4f}")
print(f"Val Loss  : {val_losses[-1]:.4f}")
print(f"Test Loss : {avg_test_loss:.4f}")

# ======================
# 12. 텍스트 생성 함수
# ======================
def generate_text(model, start_text, word2idx, idx2word, max_len=100):
    model.eval()
    words = start_text.split()
    input_seq = [word2idx.get(w, 0) for w in words]

    for _ in range(max_len):
        input_tensor = torch.tensor(input_seq[-seq_len:], dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        next_idx = torch.argmax(output, dim=1).item()
        input_seq.append(next_idx)

    return ' '.join([idx2word[i] for i in input_seq])

# ======================
# 13. 소설 생성 예시
# ======================
start = "화산 귀환"
result = generate_text(model, start, word2idx, idx2word, max_len=150)
print("\n📝 생성된 설명체 소설:\n", result)

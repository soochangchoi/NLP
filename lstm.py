import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

# 1. Load poem text from poetry.txt
with open("../project/poetry.txt", "r", encoding="utf-8-sig") as f:
    raw_text = f.read()

# 2. Clean and prepare text
lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
all_text = "\n".join(lines)

# 3. Character-level tokenization
chars = sorted(list(set(all_text)))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(chars)
seq_length = 20 

embedding_dim = 128
hidden_dim = 256
data = [char2idx[ch] for ch in all_text]

# 4. Dataset
class PoemDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    def __len__(self):
        return len(self.data) - self.seq_length
    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_length]),
            torch.tensor(self.data[idx+1:idx+self.seq_length+1])
        )

# 5. Dataloader
dataset = PoemDataset(data, seq_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 6. Model
class LSTMPoet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

model = LSTMPoet(vocab_size, embedding_dim, hidden_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Training
def train(model, dataloader, epochs=20, save_path="poet_model.pt"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            output, _ = model(x)
            loss = loss_fn(output.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Avg Loss: {total_loss/len(dataloader):.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# 8. Generate poem
def sample_with_temperature(prob, temperature=1.0):
    prob = np.log(prob + 1e-10) / temperature
    prob = np.exp(prob) / np.sum(np.exp(prob))
    return np.random.choice(len(prob), p=prob)

def generate_poem(model, start_text="봄", length=80, temperature=1.0):
    model.eval()
    input_seq = [char2idx[ch] for ch in start_text if ch in char2idx]
    if len(input_seq) == 0:
        return "⚠️ 입력된 단어가 문자 집합에 없습니다."
    while len(input_seq) < seq_length:
        input_seq.insert(0, char2idx.get(' ', 0))
    input_seq = torch.tensor(input_seq).unsqueeze(0)

    generated = start_text
    hidden = None
    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_seq, hidden)
            prob = torch.softmax(output[0, -1], dim=0).numpy()
            next_idx = sample_with_temperature(prob, temperature)
            generated += idx2char[next_idx]
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[next_idx]])], dim=1)
    return generated

# 9. Main
if __name__ == "__main__":
    if not os.path.exists("poet_model.pt"):
        print("Training model...")
        train(model, dataloader, epochs=20)
    else:
        print("Loading trained model...")
        model.load_state_dict(torch.load("poet_model.pt"))

    while True:
        user_input = input("시작 문장을 입력하세요 (exit 입력 시 종료): ")
        if user_input.lower() == "exit":
            break
        temp_input = input("temperature [0.5 ~ 1.5] (기본: 1.0): ")
        try:
            temp = float(temp_input)
        except:
            temp = 1.0
        print("\n생성된 시:\n", generate_poem(model, start_text=user_input, temperature=temp))

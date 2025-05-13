# 📚 화산귀환 & 달빛 AI 자연어 생성 프로젝트

> 한국어 자연어 생성 프로젝트  
> - 📖 **화산귀환**: 설명체 스타일 소설 생성 (LSTM 기반)  
> - 🌙 **달빛**: 감성 시 생성 (LSTM, Transformer 기반)

---

## 📂 프로젝트 구성

| 파트                  | 주요 기능                        | 스크립트                 | 모델 파일                   |
|---------------------|-----------------------------|------------------------|--------------------------|
| 화산귀환 설명체 소설 생성   | 형태소 단위 LSTM 소설 생성            | `hwa.py`               | `story_model_epochN.pth` |
| 달빛 시 생성 (LSTM)     | 문자 단위 LSTM 기반 감성 시 생성        | `lstm.py`              | `poet_model.pt`          |
| 달빛 시 생성 (Transformer) | Transformer 기반 감성 시 생성           | `transformer_model.py` | `transformer_poet.pt`    |

---

## 🛠 사용 기술

- Python 3.x
- PyTorch
- KoNLPy (Okt 형태소 분석)
- Numpy, Scikit-learn
- Matplotlib

---

## ▶ 실행 방법

### 1. 📖 화산귀환 설명체 소설 생성
```bash
python hwa.py
데이터: hawsan.txt

전처리: Okt 형태소 분석 + 불용어 제거(poetry.txt)

모델: LSTM (WordLSTM)

결과: 화산 귀환으로 시작하는 설명체 소설 생성 예시 출력

2. 🌙 달빛 LSTM 시 생성

python lstm.py
데이터: poetry.txt

전처리: 문자 단위

모델: LSTM (LSTMPoet)

결과: 시작 문자열 입력 → 감성 시 생성

3. 🌙 달빛 Transformer 시 생성

from transformer_model import TransformerPoet, generate_transformer_poem

model = TransformerPoet(vocab_size)
model.load_state_dict(torch.load('transformer_poet.pt'))
result = generate_transformer_poem(model, start_text="달빛", length=150, temperature=1.0)
print(result)

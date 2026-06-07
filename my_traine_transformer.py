import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import math

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Пристрій для навчання: {device}")

EMBEDDING_DIM = 100
MAX_LEN = 383
NUM_HEADS = 4       # Кількість "голів" уваги (100 ділиться на 4)
NUM_LAYERS = 2      # Кількість шарів трансформера
HIDDEN_DIM = 256    # Розмір прихованого шару
EPOCHS = 15

# Downloading data. ЗАВАНТАЖЕННЯ ДАНИХ ТА WORD2VEC
print("\nЗавантаження даних...")
with open('datafile_4_mers_2100_per_class_3_classes.pkl', 'rb') as f:
    data = pickle.load(f)

sentences = [item['seq'] for item in data]
raw_labels = [item['class'].strip() for item in data]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(raw_labels)

w2v_model = Word2Vec.load("biological_word2vec.model")

# СТВОРЕННЯ СЛОВНИКА (Creating VOCABULARY)
# Для Трансформера потрібні не вектори, а індекси (ID) кожного 4-мера
print("Створення словника...")
word2idx = {'<PAD>': 0, '<UNK>': 1} # 0 для пустих місць, 1 для невідомих
idx = 2
for kmer in w2v_model.wv.index_to_key:
    word2idx[kmer] = idx
    idx += 1

vocab_size = len(word2idx)

# Створюємо матрицю ваг для шару Embedding
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, i in word2idx.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]
    elif word == '<UNK>':
        embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBEDDING_DIM,))
# Для <PAD> залишаються нулі

# ТОКЕНІЗАЦІЯ ПОСЛІДОВНОСТЕЙ
def tokenize_sequence(seq, max_len):
    tokens = np.zeros(max_len, dtype=np.int64) # Заповнено <PAD> (нулями)
    for i, kmer in enumerate(seq):
        if i >= max_len:
            break
        tokens[i] = word2idx.get(kmer, 1) # Якщо немає, ставимо <UNK>
    return tokens

X = np.array([tokenize_sequence(seq, MAX_LEN) for seq in sentences])

# РОЗБИТТЯ НА ТЕНЗОРИ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32, shuffle=False)

# АРХІТЕКТУРА ТРАНСФОРМЕРА
# Позиційне кодування (бо Трансформер не знає, де початок, а де кінець)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Основна модель
class SequenceTransformer(nn.Module):
    def __init__(self, num_classes):
        super(SequenceTransformer, self).__init__()
        # Завантажуємо наші Word2Vec вектори
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.pos_encoder = PositionalEncoding(EMBEDDING_DIM, MAX_LEN)
        
        # Блок Трансформера (Self-Attention)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=EMBEDDING_DIM, 
            nhead=NUM_HEADS, 
            dim_feedforward=HIDDEN_DIM, 
            dropout=0.3, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, NUM_LAYERS)
        
        # Класифікаційна "голова"
        self.fc1 = nn.Linear(EMBEDDING_DIM, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Вказуємо моделі ігнорувати пусті місця (<PAD> = 0)
        padding_mask = (x == 0)
        
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Прохід через Трансформер
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Усереднюємо інформацію зі всієї послідовності (Global Average Pooling)
        x = x.mean(dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_attention_matrix(self, x):
        """
        Пропускає дані через модель і "перехоплює" матрицю уваги з останнього шару.
        """
        padding_mask = (x == 0)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Пропускаємо через усі шари, крім останнього
        for i in range(NUM_LAYERS - 1):
            x = self.transformer_encoder.layers[i](x, src_key_padding_mask=padding_mask)
            
        # Беремо останній шар Трансформера
        last_layer = self.transformer_encoder.layers[-1]
        
        # Викликаємо механізм MultiheadAttention напряму, вимагаючи повернути ваги (need_weights=True)
        attn_output, attn_weights = last_layer.self_attn(
            x, x, x, 
            key_padding_mask=padding_mask, 
            need_weights=True, 
            average_attn_weights=True # Усереднюємо по всіх "головах"
        )
        return attn_weights

model = SequenceTransformer(num_classes=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# ЦИКЛ НАВЧАННЯ
print("\nПочинаємо навчання Трансформера на відеокарті...")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Епоха {epoch+1}/{EPOCHS} - Втрати (Loss): {running_loss/len(train_loader):.4f}")

# ОЦІНКА ТОЧНОСТІ
print("\nГенеруємо фінальний звіт...")
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n=== РЕЗУЛЬТАТИ TRANSFORMER КЛАСИФІКАЦІЇ ===")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

import matplotlib.pyplot as plt
import seaborn as sns

print("\n=== ГЕНЕРАЦІЯ ІНТЕРПРЕТОВАНОГО ЗВІТУ (ATTENTION HEATMAP) ===")

# Створюємо зворотний словник, щоб перетворити ID назад у 4-мери (ДНК)
idx2word = {v: k for k, v in word2idx.items()}

# Беремо один батч із тестової вибірки
inputs, labels = next(iter(test_loader))

# Вибираємо першу послідовність з батчу (індекс 0)
sample_x = inputs[0:1].to(device)  # Зберігаємо розмірність [1, MAX_LEN]
sample_label = labels[0].item()
class_name = label_encoder.classes_[sample_label]

# Витягуємо матрицю уваги
model.eval()
with torch.no_grad():
    # Отримуємо матрицю розміром [1, MAX_LEN, MAX_LEN]
    attn_weights = model.get_attention_matrix(sample_x)
    matrix = attn_weights[0].cpu().numpy()

# послідовність з датасету має довжину 383, але більшість з неї - це нулі <PAD>.
# Тому ми знаходимо реальну довжину послідовності без <PAD>
actual_len = (sample_x[0] != 0).sum().item()

# Для наочності графіка візьмемо перші 30 токенів (або actual_len, якщо він менший)
plot_len = min(actual_len, 30)

# Обрізаємо матрицю та послідовність до розміру plot_len
matrix_cropped = matrix[:plot_len, :plot_len]
kmer_sequence = [idx2word[idx.item()] for idx in sample_x[0][:plot_len]]

# ПОБУДОВА ГРАФІКА (Plot the Attention Heatmap)
plt.figure(figsize=(12, 10))
sns.set_theme(style="white")

ax = sns.heatmap(
    matrix_cropped,
    xticklabels=kmer_sequence,
    yticklabels=kmer_sequence,
    cmap="viridis",
    annot=True,         # Показувати цифри
    fmt=".2f",          # 2 знаки після коми
    annot_kws={"size": 8},
    cbar_kws={'label': 'Вага уваги (Attention Weight)'},
    square=True
)

plt.title(f"Self-Attention Matrix\nКлас пацієнта: {class_name}", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Target K-mer", fontsize=12)
plt.ylabel("Source K-mer", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("attention_heatmap_real.png", dpi=300)
print("Графік успішно збережено у файл 'attention_heatmap_real.png'")
plt.show()
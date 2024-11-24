import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import string


# 1. Data Loading and Preprocessing
class SMSDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def tokenize(self, text):
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
        return text.split()

    def numericalize(self, tokens):
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]

    def __getitem__(self, idx):
        tokens = self.tokenize(self.texts[idx])
        numericalized = self.numericalize(tokens)
        if len(numericalized) < self.max_len:
            numericalized += [self.vocab['<PAD>']] * (self.max_len - len(numericalized))
        else:
            numericalized = numericalized[:self.max_len]
        return torch.tensor(numericalized), torch.tensor(self.labels[idx])


# 2. Building Vocabulary
def build_vocab(texts, min_freq=1):
    freq = defaultdict(int)
    for text in texts:
        tokens = text.lower().split()
        for token in tokens:
            freq[token] += 1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = len(vocab)
    return vocab


# 3. Transformer Model
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes, max_len, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        try:
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        except RuntimeError as e:
            print(e)
            raise(e)
        except Exception as e:
            print(e)
            raise(e)
        self.fc = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len).unsqueeze(0).to(x.device)
        x = self.embedding(x) + self.pos_embedding(positions)
        x = self.dropout(x)
        x = self.transformer(x.transpose(0, 1))  # Transformer expects [seq_len, batch, embed_dim]
        x = x.mean(dim=0)  # Global average pooling
        out = self.fc(x)
        return out


# 4. Hyperparameters
EMBED_DIM = 128
NUM_HEADS = 8
HIDDEN_DIM = 256
NUM_LAYERS = 2
NUM_CLASSES = 2
MAX_LEN = 50
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4

# 5. Load Dataset
# Assuming the dataset is in a CSV file with 'label' and 'text' columns
# You can download it from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
data = pd.read_csv('data/SMSSpamCollection.txt', sep='\t', names=['label', 'text'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# 6. Split Data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

import torch
print(torch.__version__)

# 7. Build Vocabulary
vocab = build_vocab(train_texts, min_freq=2)
vocab_size = len(vocab)

# 8. Create Datasets and DataLoaders
train_dataset = SMSDataset(train_texts, train_labels, vocab, MAX_LEN)
val_dataset = SMSDataset(val_texts, val_labels, vocab, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 9. Initialize Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    model = TransformerClassifier(vocab_size, EMBED_DIM, NUM_HEADS, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES, MAX_LEN).to(device)
except Exception as e:
    raise e
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 10. Training Loop
for epoch in range(EPOCHS):
    model.train()
    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in val_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    print(f'Epoch {epoch + 1}/{EPOCHS}, Validation Accuracy: {acc:.4f}')

# 11. Save the Model (Optional)
torch.save(model.state_dict(), 'transformer_sms_spam.pth')

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm.auto import tqdm
import warnings

# Optional: Suppress FutureWarnings and MKL warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Support of Intel(R) Streaming SIMD Extensions")

class SMSSpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def main():
    print("Loading dataset...")
    # 1. Load and Prepare the Dataset
    # Ensure the dataset file 'SMSSpamCollection.txt' is placed in the 'data/' directory
    try:
        data = pd.read_csv('data/SMSSpamCollection.txt', sep='\t', names=['label', 'text'])
    except FileNotFoundError:
        print("Dataset file 'SMSSpamCollection.txt' not found in the 'data/' directory.")
        return

    data['label'] = data['label'].map({'ham': 0, 'spam': 1})

    # Optionally, use a subset for faster execution
    # data = data.iloc[:1000]

    # Split into train and validation sets
    print("Splitting dataset into training and validation sets...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        data['text'].tolist(),
        data['label'].tolist(),
        test_size=0.2,
        random_state=42
    )

    # 2. Tokenization
    print("Initializing tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Create Dataset and DataLoader
    BATCH_SIZE = 16

    print("Creating datasets and dataloaders...")
    train_dataset = SMSSpamDataset(train_texts, train_labels, tokenizer)
    val_dataset = SMSSpamDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0)

    # 3. Initialize the Model
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 4. Set up Optimizer
    print("Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=2e-5)  # Use torch.optim.AdamW

    # 5. Training Loop
    EPOCHS = 2  # Reduced epochs for faster training in interview setting

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{EPOCHS}")
        for batch in progress_bar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation
        print("Starting validation...")
        model.eval()
        all_preds = []
        all_labels = []
        progress_bar_val = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{EPOCHS}")
        with torch.no_grad():
            for batch in progress_bar_val:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Accuracy: {acc:.4f}")

    # 6. Save the Model (Optional)
    print("Saving the model and tokenizer...")
    model.save_pretrained('distilbert_sms_spam_model')
    tokenizer.save_pretrained('distilbert_sms_spam_model')
    print("Model and tokenizer saved successfully.")

if __name__ == '__main__':
    main()

"""
Python 3.11.5
Control Case - Matrix Multiplication Method

DATASET - https://huggingface.co/datasets/microsoft/xglue

"""



"""
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\qenv\Scripts\Activate.ps1
cd c:/Users/saart/OneDrive/Desktop
python qmsan_quant.py
"""

# Required Libraries
from qiskit_aer import Aer, AerSimulator
from qiskit import QuantumCircuit, transpile
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, XLMRobertaModel
from torch.optim import AdamW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torch import amp
import time


scaler = amp.GradScaler('cuda')




if __name__ == "__main__":

    # =========================
    # Data Loading and Preprocessing
    # =========================

    # Load English training dataset
    file_path = "./xglue/xglue_full_dataset/NC/xglue.nc.en.train"
    df = pd.read_csv(file_path, sep="\t", header=None, names=["title", "description", "category"], on_bad_lines='skip', encoding='utf-8')

    # Generate label2id mapping
    unique_labels = df['category'].unique()
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    df['category'] = df['category'].map(label2id)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Split into Train and Test Sets (80/20 split)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    test_dataset = split["test"]

    # Load multilingual tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    def preprocess_function(examples):
        return tokenizer(examples['description'], padding="max_length", truncation=True, max_length=128)

    # Apply tokenization
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Rename columns for PyTorch compatibility
    train_dataset = train_dataset.rename_column("category", "labels")
    test_dataset = test_dataset.rename_column("category", "labels")

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Reduce sample size
    train_dataset = train_dataset.select(range(5000))
    test_dataset = test_dataset.select(range(5000))

# DataLoader
def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

if __name__ == "__main__":

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # =========================
    # Quantum Attention Layer (Classical Placeholder)
    # =========================

    class QuantumAttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)

        def quantum_similarity(self, q, k):
            # Placeholder: Classical similarity (dot product)
            # Replace this with quantum circuit logic as you develop it
            return torch.matmul(q, k.transpose(-1, -2))

        def forward(self, embeddings):
            Q = self.query(embeddings)
            K = self.key(embeddings)
            attn_weights = self.quantum_similarity(Q, K)
            attn_weights = attn_weights.softmax(dim=-1)
            return torch.matmul(attn_weights, embeddings)

    # =========================
    # Main Model: QMSAN
    # =========================

    class QMSAN(nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            self.transformer = XLMRobertaModel.from_pretrained("xlm-roberta-base")
            self.quantum_attention = QuantumAttentionLayer(self.transformer.config.hidden_size)
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            attended = self.quantum_attention(embeddings)
            cls_embedding = attended[:, 0, :]  # (batch_size, hidden_size)
            logits = self.classifier(cls_embedding)
            return logits

    # =========================
    # Training Setup
    # =========================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = QMSAN(num_labels=len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # =========================
    # Training Loop
    # =========================


    def train_epoch(model, dataloader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        start_time = time.time()
        for i, batch in enumerate(dataloader):
            print(f"Loaded batch {i} with size {batch['input_ids'].shape}")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            if device.type == 'cuda':
                torch.cuda.synchronize()  
            if i % 10 == 0:
                print(f"Batch {i}/{len(dataloader)} processed in {time.time()-start_time:.4f} seconds")
                start_time = time.time()
        return total_loss / len(dataloader)

    print("Starting training...")

    for epoch in range(3):  # Train for 3 epochs
        loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # =========================
    # Evaluation Function
    # =========================

    def evaluate_model(model, dataloader):
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        # Convert to numpy arrays after collecting all data
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        # Remove NaN entries
        valid_indices = ~np.isnan(all_labels)
        all_labels = all_labels[valid_indices]
        all_preds = all_preds[valid_indices]
        print(classification_report(all_labels, all_preds, target_names=list(label2id.keys())))


    # Evaluate on test set
    evaluate_model(model, test_dataloader)

    # =========================
    # Cross-Lingual Evaluation on All NC Languages
    # =========================

    languages = ["en", "de", "es", "fr", "ru"]
    for lang in languages:
        start_time = time.time()
        test_file_path = f"./xglue/xglue_full_dataset/NC/xglue.nc.{lang}.test"
        test_df = pd.read_csv(test_file_path, sep="\t", header=None, names=["title", "description", "category"], on_bad_lines='skip', encoding='utf-8')
        test_df['category'] = test_df['category'].map(label2id)
        test_dataset = Dataset.from_pandas(test_df).map(preprocess_function, batched=True)
        test_dataset = test_dataset.rename_column("category", "labels")
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
        start_time = time.time()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        print(f"\nEvaluation on {lang.upper()} test set:")
        evaluate_model(model, test_dataloader)
        print(f"Time taken for {lang.upper()} evaluation: {time.time() - start_time:.4f} seconds")

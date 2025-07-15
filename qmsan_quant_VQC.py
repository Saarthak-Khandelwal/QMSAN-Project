"""
Python 3.11.5

Variational Quantum Classifier (VQC) algorithm

DATASET - https://huggingface.co/datasets/microsoft/xglue

Usage:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\qenv\Scripts\Activate.ps1

python qmsan_quant_VQC.py
"""

# =============================
# Required Libraries
# =============================

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import AutoTokenizer, XLMRobertaModel
import pennylane as qml
import time
import numpy as np

# =============================
# Quantum Circuit Setup
# =============================

n_qubits = 4
n_layers = 2
dev = qml.device("default.qubit", wires=n_qubits)

def vqc_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(
    qml.QNode(vqc_circuit, dev, interface="torch", diff_method="backprop"),
    weight_shapes,
)

# =============================
# Data Loading and Preprocessing
# =============================

def preprocess_function(examples):
    return tokenizer(examples['description'], padding="max_length", truncation=True, max_length=128)

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

if __name__ == "__main__":

    # Load training data
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

    # Apply tokenization
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Rename columns for PyTorch compatibility
    train_dataset = train_dataset.rename_column("category", "labels")
    test_dataset = test_dataset.rename_column("category", "labels")

    # Set format for PyTorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Reduce sample size for faster experimentation
    train_dataset = train_dataset.select(range(5000))
    test_dataset = test_dataset.select(range(5000))

    # DataLoader setup
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

    # =============================
    # Model Definition
    # =============================

    # Quantum attention mechanism using VQC Layer
    class QuantumAttentionLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.query = nn.Linear(hidden_size, n_qubits)
            self.key = nn.Linear(hidden_size, n_qubits)
            self.vqc = qlayer

        def quantum_similarity(self, q, k):
            # Classical similarity as placeholder
            return torch.matmul(q, k.transpose(-2, -1))

        def forward(self, embeddings):
            Q = self.query(embeddings)
            K = self.key(embeddings)
            Q_vqc = self.vqc(Q)
            K_vqc = self.vqc(K)
            attn_weights = torch.matmul(Q_vqc, K_vqc.transpose(-2, -1))
            attn_weights = attn_weights.softmax(dim=-1)
            return torch.matmul(attn_weights, embeddings)

    # QMSAN model with QuantumAttentionLayer
    class QMSAN(nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            self.transformer = XLMRobertaModel.from_pretrained("xlm-roberta-base")
            self.quantum_attention = QuantumAttentionLayer(self.transformer.config.hidden_size)
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state
            attended = self.quantum_attention(embeddings)
            cls_embedding = attended[:, 0, :]
            logits = self.classifier(cls_embedding)
            return logits

    # =============================
    # Training Setup
    # =============================

    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = QMSAN(num_labels=len(label2id)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # =============================
    # Training Loop
    # =============================

    def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
        model.train()
        total_loss = 0
        for i, batch in enumerate(dataloader):
            start_time = time.time()
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
                print(f"Batch {i}/{len(dataloader)} processed in {time.time() - start_time:.4f} seconds")
        return total_loss / len(dataloader)

    # =============================
    # Model Training
    # =============================

    for epoch in range(3):
        loss = train_epoch(model, train_dataloader, optimizer, criterion, device, scaler)
        print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    # =============================
    # Evaluation Function
    # =============================

    def evaluate_model(model, dataloader, device, label2id):
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
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        valid_indices = ~np.isnan(all_labels)
        all_labels = all_labels[valid_indices]
        all_preds = all_preds[valid_indices]
        print(classification_report(all_labels, all_preds, target_names=list(label2id.keys())))

    # =============================
    # Evaluation on Validation/Test Set
    # =============================

    evaluate_model(model, test_dataloader, device, label2id)

    # =============================
    # Cross-Lingual Evaluation on All NC Languages
    # =============================

    languages = ["en", "de", "es", "fr", "ru"]

    for lang in languages:
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
        evaluate_model(model, test_dataloader, device, label2id)
        print(f"Evaluation on {lang.upper()} test set completed in {time.time() - start_time:.4f} seconds")

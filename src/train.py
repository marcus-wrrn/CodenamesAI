from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR

from loss import TripletMeanLoss
from torch.utils.data import DataLoader
from model import SimpleCodeGiver
from dataset import CodeGiverDataset
import numpy as np
import matplotlib.pyplot as plt
import datetime

def init_hyperparameters(model: SimpleCodeGiver):
    loss_fn = TripletMeanLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    return loss_fn, optimizer, scheduler

@torch.no_grad()
def validate(model: SimpleCodeGiver, valid_loader: DataLoader, loss_fn: TripletMeanLoss, device: torch.device):
    model.eval()
    total_loss = 0.0
    for i, data in enumerate(valid_loader, 0):
        pos_sents, neg_sents, pos_embeddings, neg_embeddings = data
        pos_embeddings, neg_embeddings = pos_embeddings.to(device), neg_embeddings.to(device)

        logits = model(pos_sents, neg_sents)
        loss = loss_fn(logits, pos_embeddings, neg_embeddings)
        total_loss += loss.item()
    model.train()
    return total_loss / len(valid_loader)

def train(n_epochs: int, model: SimpleCodeGiver, train_loader: DataLoader, valid_dataloader: DataLoader, device: torch.device, model_path: str):
    loss_fn, optimizer, scheduler = init_hyperparameters(model)
    print("Training")
    model.train()

    losses_train = []
    losses_valid = []
    print(f"Starting training at: {datetime.datetime.now()}")
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        for i, data in enumerate(train_loader, 0):
            if (i % 100 == 0):
                print(f"{datetime.datetime.now()}: Iteration: {i}/{len(train_loader)}")
            pos_sents, neg_sents, pos_embeddings, neg_embeddings = data
            # Put embeddings on device
            pos_embeddings = pos_embeddings.to(device)
            neg_embeddings = neg_embeddings.to(device)
            
            optimizer.zero_grad()
            logits = model(pos_sents, neg_sents)

            loss = loss_fn(logits, pos_embeddings, neg_embeddings)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            
        scheduler.step()
        avg_loss = loss_train / len(train_loader)
        losses_train.append(avg_loss)
        # Validate model output
        validation_loss = validate(model, valid_dataloader, loss_fn, device)
        losses_valid.append(validation_loss)
        # Log and print save model parameters
        training_str = f"{datetime.datetime.now()}, Epoch: {epoch}, Training Loss: {avg_loss}, Validation Loss: {validation_loss}"
        print(training_str)
        if len(losses_train) == 1 or losses_train[-1] < losses_train[-2]:
                torch.save(model.state_dict(), model_path)
        
    return losses_train, losses_valid

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    train_dataset = CodeGiverDataset(code_dir="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json", game_dir="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/three_word_data_medium.json")
    train_dataloader = DataLoader(train_dataset, batch_size=400, num_workers=4)

    valid_dataset = CodeGiverDataset(code_dir="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/words.json", game_dir="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/data/three_word_data.json")
    valid_dataloader = DataLoader(valid_dataset, batch_size=50, num_workers=4)

    model = SimpleCodeGiver()
    model.to(device)


    losses_train, losses_valid = train(n_epochs=10, model=model, train_loader=train_dataloader, valid_dataloader=valid_dataloader, device=device, model_path="/home/marcuswrrn/Projects/Machine_Learning/NLP/codenames/saved_models/first_model_medium_validation.out")
    

if __name__ == "__main__":
    main()
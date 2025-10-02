
import torch
from utils import accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train(model, dataloader, epochs, optimizer, loss_fn, scheduler):

    history = {
        "train_loss": [],
        "train_acc": []
    }

    model.train()
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        for indices, labels in dataloader:
            indices, labels = indices.to(device), labels.to(device)

            logits = model(indices)
            loss = loss_fn(logits, labels)
            acc = accuracy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(dataloader)
        train_acc /= len(dataloader)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        print(f"{epoch+1}/{epochs} | train_loss: {train_loss:.5f} | train_acc: {train_acc:.2f}% | lr: {optimizer.param_groups[0]['lr']}")


    return history

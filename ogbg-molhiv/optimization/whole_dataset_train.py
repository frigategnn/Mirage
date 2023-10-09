import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def train(model, dataloaders, criterion, device, n_epochs, filename):
    opt = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    with open(f"{filename}", "a") as log_file:
        # do better naming convention here, easily identifiable
        log_file.write(
            "Time, epoch, train_loss, train_auc_roc, train_acc, train_f1, val_loss, val_auc_roc, val_acc, val_f1"
            ", test_loss, test_auc_roc, test_acc, test_f1"
        )
        loop = tqdm(range(n_epochs), ascii=True, unit="epoch")
        loop.set_description("Training")
        for epoch in loop:
            train_loss = train_step(model, train_loader, criterion, device, opt)
            train_eval_loss, train_score, train_acc, train_f1 = val_step(
                model, train_loader, criterion, device
            )
            val_loss, score, acc, f1 = val_step(model, val_loader, criterion, device)
            test_loss, test_score, test_acc, test_f1 = val_step(model, test_loader, criterion, device)
            loop.set_postfix(
                {
                    "train-loss": f"{train_loss:.2f}",
                    "train-auc-roc": f"{train_score:.2f}",
                    "train-auc-roc": f"{train_score:.2f}",
                    "train-acc": f"{train_acc:.2f}",
                    "val-loss": f"{val_loss:.2f}",
                    "val-auc-roc": f"{score:.2f}",
                    "acc": f"{acc:.2f}",
                    "f1": f"{f1:.2f}",
                }
            )
            log_file.write(
                f"\n{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}, {epoch},"
                f" {train_loss}, {train_score}, {train_acc}, {train_f1},"
                f" {val_loss}, {score}, {acc}, {f1},"
                f" {test_loss}, {test_score}, {test_acc}, {test_f1}"
            )
            log_file.flush()


def train_step(model, loader, criterion, device, opt):
    model.train()
    losses = 0
    nb = len(loader)
    for data in loader:
        data = data.to(device)
        rte = data.roots_to_embed if hasattr(data, "roots_to_embed") else None
        out = model(data.x, data.edge_index, data.edge_attr, data.batch, rte)
        loss = criterion(out, data.y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses += loss.item() / nb
    return losses


def val_step(model, loader, criterion, device):
    model.eval()
    losses = 0
    score = 0
    accuracy = 0
    f1 = 0
    nb = len(loader)
    probs, ys, yhats = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            rte = data.roots_to_embed if hasattr(data, "roots_to_embed") else None
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, rte)
            yhat = nn.functional.softmax(out, dim=1)
            prob_class_1 = yhat[:, 1].detach().cpu().view(-1).numpy()
            probs.append(prob_class_1)
            ys.append(data.y.cpu().view(-1).numpy())
            yhat = yhat.argmax(dim=1)
            yhats.append(yhat.cpu().detach().view(-1).numpy())
            loss = criterion(out, data.y.view(-1))
            losses += loss.item() / nb
        probs = np.concatenate(probs)
        ys = np.concatenate(ys)
        yhats = np.concatenate(yhats)
        score = roc_auc_score(ys, probs)
        accuracy = accuracy_score(ys, yhats)
        f1 = f1_score(ys, yhats)
    return losses, score, accuracy, f1

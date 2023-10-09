r"""This file contains the optimization code of the model. It consists
of `train` function which runs the training step n_runs number of times.

No model saving implemented yet as this is just the testing phase of the
distillation scheme.
"""
import os
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def train(model, dataloaders, criterion, device, n_epochs, n_runs, filename, output_dir, opt_args=None):
    r"""Train outer loops. Initializes the optimizer and writes to log
    file given as param `filename`.
    """
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    test_loader = dataloaders["test"]
    best_vals = []
    best_test = []
    logfilename = filename[filename.rindex('/')+1:]
    logfilename = logfilename[:logfilename.rindex('.')]
    with open(f"{filename}", "a") as log_file:
        # do better naming convention here, easily identifiable
        log_file.write(
            "Time, epoch, train_loss, train_auc_roc, train_acc, train_f1,"
            " val_loss, val_auc_roc, val_acc, val_f1,"
            " test_loss, test_auc_roc, test_acc, test_f1"
        )
        model_filename = f'{logfilename}_{datetime.now().strftime("%d-%m-%Y_%H_%M_%S")}_model.pt'
        for run in range(n_runs):
            best_vals.append(0)
            best_test.append(0)
            model.reset_parameters()
            if opt_args is None:
                opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
            else:
                opt = optim.Adam(model.parameters(), **opt_args)
            log_file.write(f"\nStarting run number {run}")
            loop = tqdm(range(n_epochs), ascii=True, unit="epoch")
            loop.set_description("Training")
            for epoch in loop:
                val_loss, score, acc, f1 = val_step(
                    model, val_loader, criterion, device
                )
                test_loss, test_score, test_acc, test_f1 = val_step(
                    model, test_loader, criterion, device
                )
                if best_vals[run] < score:
                    best_vals[run] = score
                    best_test[run] = test_score
                    torch.save(model.state_dict, os.path.join(output_dir, f'{model_filename}'))
                train_loss = train_step(model, train_loader, criterion, device, opt)
                (_, train_score, train_acc, train_f1) = val_step(
                    model, train_loader, criterion, device
                )
                loop.set_postfix(
                    {
                        "train-loss": f"{train_loss:.2f}",
                        "train-auc-roc": f"{train_score:.2f}",
                        # "train-acc": F"{train_acc:.2f}",
                        # "train-f1": F"{train_f1:.2f}",
                        "val-loss": f"{val_loss:.2f}",
                        "val-auc-roc": f"{score:.2f}",
                        # "acc": F"{acc:.2f}",
                        # "f1": F"{f1:.2f}",
                        "test-auc-roc": f"{test_score:.2f}",
                    }
                )
                log_file.write(
                    f"\n{datetime.now().strftime('%d-%m-%Y %H:%M:%S')}, {epoch},"
                    f" {train_loss}, {train_score}, {train_acc}, {train_f1}, "
                    f"{val_loss}, {score}, {acc}, {f1},"
                    f" {test_loss}, {test_score}, {test_acc}, {test_f1}"
                )
                log_file.flush()
        log_file.write(f'vals {best_vals}\n')
        log_file.write(f'tests {best_test}\n')
        sort_arg = np.argsort(best_vals)[::-1] #sort non-increasing
        top_5 = np.asarray(best_test)[sort_arg[:5]]
        mean = np.mean(top_5)
        std = np.std(top_5)
        log_file.write(f'mean = {mean}, std = {std}\n')
        log_file.write(f'{mean:.6f}+-{std:.6f}\n')
        log_file.write(f'Best performance = {np.max(top_5)}\n')
        print(f'Performance summary {mean:.6f}+-{std:.6f}')
        print(f'Best performance = {np.max(top_5)}')


def train_step(model, loader, criterion, device, opt):
    r"""Single optimization step (over all the batches in dataloader).
    """
    model.train()
    losses = 0
    number_batches = len(loader)
    for data in loader:
        data = data.to(device)
        rte = data.roots_to_embed if hasattr(data, "roots_to_embed") else None
        x = data.node_attr if hasattr(data, "node_attr") else data.x
        out = model(x, data.edge_index, data.edge_attr, data.batch, rte)
        loss = criterion(out, data.y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses += loss.item() / number_batches
    return losses


def val_step(model, loader, criterion, device):
    r"""Evaluation code. Can be used for both validation and test split.
    """
    model.eval()
    losses = 0
    score = 0
    accuracy = 0
    f1 = 0
    number_batches = len(loader)
    auc_roc_input1, auc_roc_input2 = [], []
    f1_input2 = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            rte = data.roots_to_embed if hasattr(data, "roots_to_embed") else None
            x = data.node_attr.reshape(-1, 1) if hasattr(data, "node_attr") else data.x
            out = model(x, data.edge_index, data.edge_attr, data.batch, rte)
            yhat = nn.functional.softmax(out, dim=1)
            prob_class_1 = yhat[:, 1].detach().cpu().view(-1).numpy()
            auc_roc_input1.extend(data.y.cpu().view(-1).numpy().tolist())
            auc_roc_input2.extend(prob_class_1.tolist())
            yhat = yhat.argmax(dim=1)
            accuracy += (
                accuracy_score(
                    data.y.cpu().view(-1).numpy(), yhat.detach().cpu().view(-1).numpy()
                )
                / number_batches
            )
            f1_input2.extend(yhat.detach().cpu().view(-1).numpy().tolist())
            loss = criterion(out, data.y.view(-1))
            losses += loss.item() / number_batches
    f1 = f1_score(auc_roc_input1, f1_input2)
    score = roc_auc_score(auc_roc_input1, auc_roc_input2)
    return losses, score, accuracy, f1

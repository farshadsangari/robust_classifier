import torch
import pandas as pd
from tqdm import tqdm
import os
import models as models
import util as util

from util import accuracy, load_model, save_model, AverageMeter
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np


def trainer(
    train_loader,
    val_loader,
    model,
    embedding,
    epsilon,
    criterion,
    model_name,
    epochs,
    learning_rate,
    gamma,
    device,
    load_saved_model,
    ckpt_save_freq,
    ckpt_save_root,
    ckpt_load_path,
    report_root,
):

    model = model.to(device)

    # # loss function
    # criterion = nn.CrossEntropyLoss()

    # optimzier
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_saved_model:
        model, optimizer = load_model(
            ckpt_path=ckpt_load_path, model=model, optimizer=optimizer
        )

    lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
    report = pd.DataFrame(
        columns=[
            "model_name",
            "mode",
            "image_type",
            "epoch",
            "learning_rate",
            "batch_size",
            "batch_index",
            "loss_batch",
            "avg_train_loss_till_current_batch",
            "avg_train_top1_acc_till_current_batch",
            "avg_val_loss_till_current_batch",
            "avg_val_top1_acc_till_current_batch",
        ]
    )

    for epoch in tqdm(range(1, epochs + 1)):
        top1_acc_train = AverageMeter()
        loss_avg_train = AverageMeter()
        top1_acc_val = AverageMeter()
        loss_avg_val = AverageMeter()

        model.train()
        mode = "train"
        loop_train = tqdm(
            enumerate(train_loader, 1),
            total=len(train_loader),
            desc="train",
            position=0,
            leave=True,
        )

        for batch_idx, (images, labels) in loop_train:
            images = images.to(device).float()
            if epsilon:
                images = fast_gradient_method(
                    model_fn=model, x=images, eps=epsilon, norm=np.inf
                )
            labels = labels.to(device)
            labels_pred = model(images, embedding)
            loss = criterion(labels_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc1 = accuracy(labels_pred, labels)
            top1_acc_train.update(acc1[0], images.size(0))
            loss_avg_train.update(loss.item(), images.size(0))

            new_row = pd.DataFrame(
                {
                    "model_name": model_name,
                    "mode": mode,
                    "image_type": "original",
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "batch_size": images.size(0),
                    "batch_index": batch_idx,
                    "loss_batch": loss.detach().item(),
                    "avg_train_loss_till_current_batch": loss_avg_train.avg,
                    "avg_train_top1_acc_till_current_batch": top1_acc_train.avg,
                    "avg_val_loss_till_current_batch": None,
                    "avg_val_top1_acc_till_current_batch": None,
                },
                index=[0],
            )

            report.loc[len(report)] = new_row.values[0]

            loop_train.set_description(f"Train - iteration : {epoch}")
            loop_train.set_postfix(
                loss_batch="{:.4f}".format(loss.detach().item()),
                avg_train_loss_till_current_batch="{:.4f}".format(loss_avg_train.avg),
                top1_accuracy_train="{:.4f}".format(top1_acc_train.avg),
                max_len=2,
                refresh=True,
            )
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=ckpt_save_root,
                file_name=f"ckpt_{model_name}_epoch{epoch}.pth",
                model=model,
                optimizer=optimizer,
            )
        if val_loader:
            model.eval()
            mode = "val"
            loop_val = tqdm(
                enumerate(val_loader, 1),
                total=len(val_loader),
                desc="val",
                position=0,
                leave=True,
            )
            for batch_idx, (images, labels) in loop_val:
                images = images.to(device).float()
                labels = labels.to(device)
                if epsilon:
                    images = fast_gradient_method(
                        model_fn=model, x=images, eps=epsilon, norm=np.inf
                    )
                labels_pred = model(images, embedding).detach()
                loss = criterion(labels_pred, labels)
                acc1 = accuracy(labels_pred, labels)
                top1_acc_val.update(acc1[0], images.size(0))
                loss_avg_val.update(loss.item(), images.size(0))
                new_row = pd.DataFrame(
                    {
                        "model_name": model_name,
                        "mode": mode,
                        "image_type": "original",
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "batch_size": images.size(0),
                        "batch_index": batch_idx,
                        "loss_batch": loss.detach().item(),
                        "avg_train_loss_till_current_batch": None,
                        "avg_train_top1_acc_till_current_batch": None,
                        "avg_val_loss_till_current_batch": loss_avg_val.avg,
                        "avg_val_top1_acc_till_current_batch": top1_acc_val.avg,
                    },
                    index=[0],
                )

                report.loc[len(report)] = new_row.values[0]
                loop_val.set_description(f"Validation - iteration : {epoch}")
                loop_val.set_postfix(
                    loss_batch="{:.4f}".format(loss.detach().item()),
                    avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
                    top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
                    refresh=True,
                )
        lr_scheduler.step()
    report.to_csv(f"{report_root}/{model_name}_report.csv")
    return model, optimizer, report


def Inference_mode(
    model, dataloader, dataloader_title, device, criterion, embedding, epsilon
):
    top1_acc_val = AverageMeter()
    loss_avg_val = AverageMeter()
    model.eval()
    loop_val = tqdm(
        enumerate(dataloader, 1),
        total=len(dataloader),
        desc="val",
        position=0,
        leave=True,
    )
    for batch_idx, (images, labels) in loop_val:
        images = images.to(device).float()
        labels = labels.to(device)
        if epsilon:
            images = fast_gradient_method(
                model_fn=model, x=images, eps=epsilon, norm=np.inf
            )
        labels_pred = model(images, embedding).detach()
        loss = criterion(labels_pred, labels)
        acc1 = accuracy(labels_pred, labels)
        top1_acc_val.update(acc1[0], images.size(0))
        loss_avg_val.update(loss.item(), images.size(0))

        loop_val.set_description(f"Inference mode: ")
        loop_val.set_postfix(
            loss_batch="{:.4f}".format(loss.detach().item()),
            avg_val_loss_till_current_batch="{:.4f}".format(loss_avg_val.avg),
            top1_accuracy_val="{:.4f}".format(top1_acc_val.avg),
            refresh=True,
        )
    print(f"Loss on {dataloader_title}: {round(loss_avg_val.avg,3)}")
    print(f"Accuracy on {dataloader_title}: {round(top1_acc_val.avg,3)}%")
    return loss_avg_val.avg, top1_acc_val.avg

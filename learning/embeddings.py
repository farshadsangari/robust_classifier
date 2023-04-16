from tqdm import tqdm
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
import umap.umap_ as umap
import numpy as np
import torch


def get_backbone_embedding(model, data_loader, device, epsilon):
    model = model.to(device)
    model.eval()
    loop_data = tqdm(
        enumerate(data_loader, 1), total=len(data_loader), position=0, leave=True
    )

    embeddings = []
    labels = []
    for _, (images, labels_) in loop_data:
        images = images.to(device)
        labels_ = labels_.to(device)
        if epsilon:
            images = fast_gradient_method(
                model_fn=model, x=images, eps=epsilon, norm=np.inf
            )

        embedding = model(images, embedding=True).detach()
        embeddings.append(embedding.cpu().numpy())
        labels.extend(labels_.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, labels


def get_umap_embedding(model, origin_train_embeddings, origin_test_embeddings):
    if model is None:
        model = umap.UMAP()
        model.fit(origin_train_embeddings)
    dest_train_embeddings = model.transform(origin_train_embeddings)
    dest_test_embeddings = model.transform(origin_test_embeddings)
    return model, dest_train_embeddings, dest_test_embeddings


def dataloaders_embedding(
    model,
    to_get_umap_representation,
    train_loader,
    val_loader,
    test_loader,
    device,
    train_epsilon,
    test_epsilon,
):

    train_embeddings, train_labels = get_backbone_embedding(
        model=model, data_loader=train_loader, device=device, epsilon=train_epsilon
    )

    val_embeddings, val_labels = get_backbone_embedding(
        model=model, data_loader=val_loader, device=device, epsilon=test_epsilon
    )

    test_embeddings, test_labels = get_backbone_embedding(
        model=model, data_loader=test_loader, device=device, epsilon=test_epsilon
    )
    if to_get_umap_representation:
        umap_model, _, val_embeddings = get_umap_embedding(
            model=None,
            origin_train_embeddings=train_embeddings,
            origin_test_embeddings=val_embeddings,
        )

        umap_model, train_embeddings, test_embeddings = get_umap_embedding(
            model=umap_model,
            origin_train_embeddings=train_embeddings,
            origin_test_embeddings=test_embeddings,
        )

    return (
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        test_embeddings,
        test_labels,
    )

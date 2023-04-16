from dataloader import (
    train_test_val_split,
    CIFAR10Dataset,
    transformation,
    my_dataloader,
)
from learning import trainer
from util import load_model_params, random_seed
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "model",
    type=str,
    nargs="?",
    default="resnet_angular",
    help="""
                    model names:
                    1. resnet_ce
                    2. resnet_robust_ce
                    3. resnet_angular
                    """,
)
args = parser.parse_args()


def main(params):
    random_seed(99)
    train_paths, val_paths, test_paths = train_test_val_split(
        root1=params["train_data_path"],
        root2=params["test_data_path"],
        split_percentage=params["split_percentage"],
    )
    train_dataset_clean = CIFAR10Dataset(
        imgs_paths=train_paths,
        transforms=transformation.default_cifar10_transforms(to_augment_data=False),
    )
    val_dataset_clean = CIFAR10Dataset(
        imgs_paths=val_paths,
        transforms=transformation.default_cifar10_transforms(to_augment_data=False),
    )

    train_loader_clean = my_dataloader(
        train_dataset_clean,
        to_sampling_random=False,
        kwargs_random=None,
        kwargs_balanced={
            "n_classes": params["n_classes"],
            "n_samples": params["n_samples_per_class"],
        },
    )
    val_loader_clean = my_dataloader(
        val_dataset_clean,
        to_sampling_random=False,
        kwargs_random=None,
        kwargs_balanced={
            "n_classes": params["n_classes"],
            "n_samples": params["n_samples_per_class"],
        },
    )

    model = params["model"]
    criterion = params["criterion"]

    model, optimizer, report = trainer(
        train_loader=train_loader_clean,
        val_loader=val_loader_clean,
        model=model,
        embedding=params["embedding"],
        epsilon=params["epsilon"],
        criterion=criterion,
        model_name="resnet18_angular",
        epochs=params["epochs"],
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        device=params["device"],
        load_saved_model=params["load_saved_model"],
        ckpt_save_freq=params["ckpt_save_freq"],
        ckpt_save_root=params["ckpt_save_root"],
        ckpt_load_path=params["ckpt_load_path"],
        report_root=params["report_root"],
    )


if __name__ == "__main__":
    params = params = load_model_params(model_name=args.model)
    main(params)

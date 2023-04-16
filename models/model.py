import torchvision
import torch.nn as nn
import torch
from sklearn.neighbors import KNeighborsClassifier


class MyResNet18(nn.Module):
    def __init__(
        self, in_channels=3, num_classes=10, pretrained=None, embedding_reduction=False
    ):
        super(MyResNet18, self).__init__()

        self.resnet18 = torchvision.models.resnet18(weights=pretrained)
        self.resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )
        self.resnet18.fc = nn.Linear(
            in_features=512, out_features=num_classes, bias=True
        )
        self.backbone = nn.Sequential(*list(self.resnet18.children())[:-1])
        self.classifier = nn.Sequential(self.resnet18.fc)

        self.embedding_reduction = embedding_reduction

        self.fc_embeding_reduction = nn.Sequential(
            nn.Linear(in_features=512, out_features=128),
        )

    def forward(self, x, embedding=False):
        x = self.backbone(x)
        x = torch.flatten(x, start_dim=1)
        if embedding and self.embedding_reduction:
            x = self.fc_embeding_reduction(x)
            x = torch.nn.functional.normalize(x)
        if not embedding:
            x = self.classifier(x)
        return x

    def fix_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


def KNN_classifier(
    model, X_train, y_train, X_test, y_test, n_neighbors, test_data_title
):
    if model is None:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

    test_acc = model.score(X_test, y_test)

    print(f"Accuracy of on {test_data_title}: {round(test_acc*100)}%")
    return model

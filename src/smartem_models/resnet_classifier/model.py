import torch
from torchvision import models


class Net(torch.nn.Module):
    def __init__(
        self,
        feature_extractor: models.resnet.ResNet,
        num_classes: int,
        centroid_size: int,
        model_output_size: int,
        length_scale: float,
        gamma: float,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = torch.nn.Parameter(torch.zeros(centroid_size, num_classes, model_output_size))
        torch.nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.feature_extractor = feature_extractor
        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer("m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05))
        self.m: torch.Tensor = self.m * self.N

        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        y_pred = (diff**2).mean(1).div(2 * self.sigma**2).mul(-1).exp()

        return z, y_pred

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

    def forward(self, x):
        features = self.feature_extractor(x)
        z, y_pred = self.rbf(features)

        return z, y_pred

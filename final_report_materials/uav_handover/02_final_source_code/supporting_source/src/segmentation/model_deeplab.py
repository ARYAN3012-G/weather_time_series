import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus(nn.Module):
    def __init__(
        self,
        num_classes=7,
        input_size=512,
        pretrained_id=None,
        pretrained_local_path=None,
        in_channels=9,
    ):
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)

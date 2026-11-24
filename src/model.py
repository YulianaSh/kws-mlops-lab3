"""Архітектура легкої CNN для розпізнавання ключових слів (з ЛР №1)."""
from torch import nn

ARCHITECTURE = "CNN: 3x[Conv3x3-BN-ReLU-MaxPool](16-32-64) + GAP + Dropout(0.3) + FC"


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class KWSNet(nn.Module):
    """Вхід [B, 1, 40, 101] (лог-Mel), вихід — логіти [B, n_classes]."""

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 16), conv_block(16, 32), conv_block(32, 64))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.dropout(x))


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

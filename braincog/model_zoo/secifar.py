from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from braincog.datasets import is_dvs_data


@register_model
class SeCifarNet(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 channels=128,
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        assert not is_dvs_data(self.dataset), 'SNN7_tiny only support static datasets now'

        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(nn.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(nn.BatchNorm1d(channels))
                conv.append(self.node())

            conv.append(nn.AvgPool1d(2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 8, channels * 8 // 4),
            self.node(),
            nn.Linear(channels * 8 // 4, num_classes),
        )

    def forward(self, inputs):
        # inputs = self.encoder(inputs)
        inputs = inputs.permute(3, 0, 1, 2)
        self.reset()

        if self.layer_by_layer:
            inputs = rearrange(inputs, 'w b c h -> (w b) c h')
            x = self.conv(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.conv(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


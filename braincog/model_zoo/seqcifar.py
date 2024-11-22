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
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        self.dataset = kwargs['dataset']
        assert not is_dvs_data(self.dataset), 'SNN7_tiny only support static datasets now'

        self.feature = nn.Sequential(
            BaseConvModule(3, 16, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(16, 64, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(64, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(128, 128, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(128, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            BaseConvModule(256, 256, kernel_size=(3, 3), padding=(1, 1), node=self.node),
            nn.MaxPool2d(2),
            BaseConvModule(256, 512, kernel_size=(3, 3), padding=(1, 1), node=self.node),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, self.num_classes),
        )

        conv = []
        for i in range(2):
            for j in range(3):
                if conv.__len__() == 0:
                    in_channels = 3
                else:
                    in_channels = channels
                conv.append(layer.Conv1d(in_channels, channels, kernel_size=3, padding=1, bias=False))
                conv.append(layer.BatchNorm1d(channels))
                conv.append(
                    create_neuron(neu, T=T, features=channels, surrogate_function=surrogate.ATan(), channels=channels,
                                  P=P, exp_init=exp_init))

            conv.append(layer.AvgPool1d(2))

        self.conv = nn.Sequential(*conv)

        self.fc = nn.Sequential(
            layer.Flatten(),
            layer.Linear(channels * 8, channels * 8 // 4),
            create_neuron(neu, T=T, features=channels * 8 // 4, surrogate_function=surrogate.ATan(), P=P,
                          exp_init=exp_init),
            layer.Linear(channels * 8 // 4, class_num),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()

        if self.layer_by_layer:
            x = self.feature(inputs)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> t b c', t=self.step).mean(0)
            return x

        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                x = self.feature(x)
                x = self.fc(x)
                outputs.append(x)

            return sum(outputs) / len(outputs)


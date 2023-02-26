import torch
import json


def init_weights(m: torch.nn.Module, mean=0.0, std=0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def load_checkpoint(filepath: str, device: str):
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path: str) -> AttrDict:
    with open(config_path) as f:
        data = f.read()
    json_config = json.loads(data)
    config = AttrDict(json_config)
    return config

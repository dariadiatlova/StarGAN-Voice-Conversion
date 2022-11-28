import numpy as np
import torch


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        # other_dim = [x.shape[1] for x in inputs]

        # other_dim = [np.shape(x)[1] for x in inputs]
        # print(f"OTHER DIM {other_dim}")
        # print(f"MAX LEN {max_len}")
        res = [pad(x, max_len) for x in inputs]
        # try:
        output = np.stack(res)
        # except ValueError:
        #     print([x.shape for x in res])

    return output


def get_mask_from_lengths(lengths, device):
    batch_size = lengths.shape[0]
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    lengths = lengths.to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask

import os
import numpy as np


def _conv2d(x, w, b, stride=1, pad=0, chunk_rows=16):
    c, h, width = x.shape
    oc, ic, k, _ = w.shape

    if pad:
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
    else:
        xp = x

    ho = (h + 2 * pad - k) // stride + 1
    wo = (width + 2 * pad - k) // stride + 1

    out = np.empty((oc, ho, wo), dtype=np.float32)
    wf = w.reshape(oc, -1).T

    for r0 in range(0, ho, chunk_rows):
        r1 = min(ho, r0 + chunk_rows)
        region = xp[
            :,
            r0 * stride:r0 * stride + (r1 - r0 - 1) * stride + k,
            :
        ]

        windows = np.lib.stride_tricks.sliding_window_view(
            region, (k, k), axis=(1, 2)
        )
        windows = windows[:, ::stride, ::stride, :, :]

        patches = windows.transpose(1, 2, 0, 3, 4)
        patches = patches.reshape(-1, c * k * k)

        y = patches @ wf + b

        out[:, r0:r1, :] = y.reshape(
            r1 - r0, wo, oc
        ).transpose(2, 0, 1)

    return out


def _relu(x):
    return np.maximum(x, 0)


def _maxpool(x, k=3, stride=2, pad=1):
    xp = np.pad(
        x,
        ((0, 0), (pad, pad), (pad, pad)),
        constant_values=-np.inf
    )

    windows = np.lib.stride_tricks.sliding_window_view(
        xp, (k, k), axis=(1, 2)
    )

    return windows[:, ::stride, ::stride, :, :].max(axis=(-1, -2))


def _block(x, weights, layer, block):
    base = f"layer{layer}.{block}"
    stride = 2 if layer > 1 and block == 0 else 1

    y = _relu(_conv2d(
        x,
        weights[f"{base}.conv1_w"],
        weights[f"{base}.conv1_b"],
        stride=stride,
        pad=1
    ))

    y = _conv2d(
        y,
        weights[f"{base}.conv2_w"],
        weights[f"{base}.conv2_b"],
        stride=1,
        pad=1
    )

    if f"{base}.down_w" in weights:
        identity = _conv2d(
            x,
            weights[f"{base}.down_w"],
            weights[f"{base}.down_b"],
            stride=2,
            pad=0
        )
    else:
        identity = x

    return _relu(y + identity)


class ResNet18Numpy:
    def __init__(self, model_path):
        self.weights = dict(np.load(model_path, allow_pickle=False))

    def predict_logits(self, x):
        w = self.weights

        x = _relu(_conv2d(
            x, w["conv1_w"], w["conv1_b"],
            stride=2, pad=3
        ))

        x = _maxpool(x)

        for layer in range(1, 5):
            for block in range(2):
                x = _block(x, w, layer, block)

        x = x.mean(axis=(1, 2))

        return w["fc_w"] @ x + w["fc_b"]

    def predict(self, x):
        logits = self.predict_logits(x)
        shifted = logits - logits.max()
        exp = np.exp(shifted)
        probabilities = exp / exp.sum()
        index = int(probabilities.argmax())
        return index, probabilities

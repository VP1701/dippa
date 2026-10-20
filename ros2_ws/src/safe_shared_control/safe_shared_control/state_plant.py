"""Actuator network classes, the same as in the training scripts.

Checkpoints describe their network in config: "arch" ("mlp" when absent,
for older checkpoints), "inputs", "hidden", "activation" and, for a TDNN,
"channels" and "kernel".  Always build with build_plant(config): an
activation has no parameters, so loading weights into a network with the
wrong activation or class raises no error but gives wrong outputs.
"""
import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


def mlp(size, hidden, activation, outputs=2):
    layers = []
    for width in hidden:
        layers += [nn.Linear(size, width), ACTIVATIONS[activation]()]
        size = width
    layers.append(nn.Linear(size, outputs))
    return nn.Sequential(*layers)


class StatePlant(nn.Module):
    """MLP: [v, omega], gamma and both command histories in, rates out."""

    def __init__(self, inputs, hidden, activation="relu"):
        super().__init__()
        self.net = mlp(3 + 2 * inputs, hidden, activation)

    def forward(self, x):
        return self.net(x)


class TDNNPlant(nn.Module):
    """TDNN: convolutions over the two command histories, MLP head.

    Input x = [v, omega, gamma, u_v history, u_w history], histories oldest
    first, the same layout as StatePlant.  Works on a batch (N, D) and on
    a single sample (D,), as torch.func.vmap in the MPC needs.
    """

    def __init__(self, inputs, channels, kernel, hidden, activation="tanh"):
        super().__init__()
        length = inputs - 2 * (kernel - 1)
        if length < 1:
            raise ValueError(f"history of {inputs} is too short for two "
                             f"convolutions of kernel {kernel}")
        self.inputs = inputs
        act = ACTIVATIONS[activation]
        self.conv = nn.Sequential(
            nn.Conv1d(2, channels, kernel), act(),
            nn.Conv1d(channels, channels, kernel), act())
        self.head = mlp(3 + channels * length, hidden, activation)

    def forward(self, x):
        state, cmds = x[..., :3], x[..., 3:]
        feat = self.conv(cmds.unflatten(-1, (2, self.inputs))).flatten(-2)
        return self.head(torch.cat([state, feat], dim=-1))


def build_plant(config):
    """The network a checkpoint config describes."""
    activation = config.get("activation", "relu")
    if config.get("arch", "mlp") == "tdnn":
        return TDNNPlant(config["inputs"], config["channels"],
                         config["kernel"], config["hidden"], activation)
    return StatePlant(config["inputs"], config["hidden"], activation)
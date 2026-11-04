import pytest
import torch
import torch.nn as nn
from src.core.decolle_layer import DECOLLELayer, DECOLLENetwork, FastSigmoidSurrogate


def test_surrogate_gradient():
    x = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    spikes = FastSigmoidSurrogate.apply(x, 1.0, 10.0)
    assert torch.all(spikes == torch.tensor([0.0, 1.0, 1.0]))

    loss = spikes.sum()
    loss.backward()
    assert x.grad is not None
    # At x=1.0, diff=0, surrogate grad is 1.0
    assert torch.isclose(x.grad[1], torch.tensor(1.0), atol=1e-3)
    # At x=0.5, diff=0.5, surrogate grad is 1/(1+5)^2 = 1/36 ~ 0.0277
    assert torch.isclose(x.grad[0], torch.tensor(1.0 / 36.0), atol=1e-3)


def test_decolle_layer_forward_step():
    batch_size = 4
    in_features = 16
    out_features = 8
    num_classes = 5

    layer = DECOLLELayer(in_features=in_features, out_features=out_features, num_classes=num_classes)
    x = (torch.rand(batch_size, in_features) > 0.5).float()

    spikes, state, logits = layer.forward_step(x)
    u, s_bar = state

    assert spikes.shape == (batch_size, out_features)
    assert u.shape == (batch_size, out_features)
    assert s_bar.shape == (batch_size, out_features)
    assert logits.shape == (batch_size, num_classes)
    assert torch.all((spikes == 0.0) | (spikes == 1.0))


def test_decolle_network_forward_and_local_learning():
    batch_size = 4
    time_steps = 10
    in_features = 16
    hidden_features = 32
    num_classes = 4

    net = DECOLLENetwork(layer_sizes=[in_features, hidden_features, 16], num_classes=num_classes)
    optimizers = net.get_adamax_optimizers(lr=1e-3)

    assert len(optimizers) == 2
    # Verify optimizer configuration: beta1=0, beta2=0.95
    for opt in optimizers:
        assert isinstance(opt, torch.optim.Adamax)
        assert opt.defaults["betas"] == (0.0, 0.95)

    spike_input = (torch.rand(batch_size, time_steps, in_features) > 0.8).float()
    targets = torch.randint(0, num_classes, (batch_size,))

    all_layer_spikes, all_layer_readouts = net.forward_sequence(spike_input)
    assert len(all_layer_spikes) == 2
    assert len(all_layer_readouts) == 2

    for i, (layer, spikes, readout, opt) in enumerate(zip(net.layers, all_layer_spikes, all_layer_readouts, optimizers)):
        opt.zero_grad()
        loss = layer.compute_local_loss(readout, targets, spikes, time_steps)
        assert loss.item() > 0
        loss.backward()
        assert layer.linear.weight.grad is not None
        # Verify random readout has no gradients (fixed projection)
        assert layer.random_readout.weight.grad is None
        opt.step()

import pytest
import torch
from src.core.dataset import NeuromorphicDataset, get_neuromorphic_dataloader
from src.core.pipeline import NeuromorphicCognitivePipeline


def test_neuromorphic_dataset_and_loader():
    num_samples = 20
    time_steps = 8
    input_dim = 32
    num_classes = 4
    batch_size = 5

    loader = get_neuromorphic_dataloader(
        batch_size=batch_size,
        num_samples=num_samples,
        time_steps=time_steps,
        input_dim=input_dim,
        num_classes=num_classes,
    )

    batches = list(loader)
    assert len(batches) == num_samples // batch_size
    sample_x, sample_y = batches[0]
    assert sample_x.shape == (batch_size, time_steps, input_dim)
    assert sample_y.shape == (batch_size,)
    assert torch.all((sample_x == 0.0) | (sample_x == 1.0))


def test_full_cognitive_pipeline_cycle():
    device = torch.device("cpu")
    input_dim = 16
    hidden_dims = [32, 16]
    num_classes = 4

    pipeline = NeuromorphicCognitivePipeline(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        device=device,
        lr=1e-3,
    )

    inputs = (torch.rand(4, 6, input_dim) > 0.8).float()
    targets = torch.randint(0, num_classes, (4,))

    stats = pipeline.process_batch(inputs, targets, train_step=True)

    assert "accuracy" in stats
    assert "surprise" in stats
    assert "route" in stats
    assert stats["route"] in ["FAST_SYSTEM_1", "SLOW_SYSTEM_2"]
    assert "behavioral_mode" in stats
    assert len(stats["state_vector"]) == 4

    # Verify live neuron network activities are tracked
    assert "layer_activities" in stats
    assert len(stats["layer_activities"]) == 2
    assert "firing_rate" in stats["layer_activities"][0]
    assert "sample_neuron_activity" in stats["layer_activities"][0]

    # Run sleep consolidation
    c_stats = pipeline.sleep_and_consolidate()
    assert "processed_events" in c_stats
    assert c_stats["processed_events"] >= 1

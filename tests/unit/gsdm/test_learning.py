"""The model must learn a task that is only solvable by reasoning over graph structure."""

import numpy as np

from gsdm.graph import StructuralGraph
from gsdm.nn import GSDMConfig, GSDModel
from gsdm.optim import Adam
from gsdm.training import Trainer, TrainingExample, predict_logits

NODE_TYPES = 4
TYPE_PROBABILITIES = [0.08, 0.22, 0.22, 0.48]


def chain_example(rng: np.random.Generator) -> TrainingExample:
    node_count = int(rng.integers(5, 10))
    types = rng.choice(NODE_TYPES, size=node_count, p=TYPE_PROBABILITIES)
    features = np.eye(NODE_TYPES)[types]
    forward = [(i, i + 1) for i in range(node_count - 1)]
    backward = [(i + 1, i) for i in range(node_count - 1)]
    has_type_zero = bool((types == 0).any())
    has_adjacent_one_then_two = any(types[i] == 1 and types[i + 1] == 2 for i in range(node_count - 1))
    labels = np.array([has_type_zero, has_adjacent_one_then_two], dtype=float)
    return TrainingExample(StructuralGraph.from_edge_lists(features, [forward, backward]), labels)


def accuracy(model: GSDModel, examples: list[TrainingExample]) -> float:
    logits = predict_logits(model, [example.graph for example in examples])
    labels = np.array([example.labels for example in examples])
    return float(((logits > 0) == (labels > 0.5)).mean())


def test_model_learns_a_relational_task_and_generalizes():
    rng = np.random.default_rng(11)
    training = [chain_example(rng) for _ in range(160)]
    held_out = [chain_example(rng) for _ in range(80)]
    model = GSDModel(GSDMConfig(feature_size=NODE_TYPES, relation_count=2, decision_count=2, hidden_size=16, reasoning_steps=2, seed=1))
    trainer = Trainer(model, Adam(model.parameters(), learning_rate=0.01), batch_size=8, seed=2)

    losses = trainer.fit(training, epochs=40)

    assert losses[-1] < losses[0] / 4
    assert accuracy(model, training) >= 0.97
    assert accuracy(model, held_out) >= 0.9

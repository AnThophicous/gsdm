import numpy as np
import pytest

from gsdm.autograd import Tensor, binary_cross_entropy_with_logits
from gsdm.graph import StructuralGraph
from gsdm.nn import GSDMConfig, GSDModel, LayerNorm, Linear
from tests.gradient_check import assert_gradients_match

pytestmark = pytest.mark.usefixtures("float64_precision")


def small_graph(rng: np.random.Generator, feature_size: int = 5) -> StructuralGraph:
    edges = [[(0, 1), (1, 2), (2, 3)], [(1, 0), (2, 1), (3, 2)], [(0, 3)]]
    return StructuralGraph.from_edge_lists(rng.normal(size=(4, feature_size)), edges)


def small_model(**overrides) -> GSDModel:
    values = dict(feature_size=5, relation_count=3, decision_count=2, hidden_size=4, reasoning_steps=2, seed=3)
    values.update(overrides)
    return GSDModel(GSDMConfig(**values))


def test_layer_norm_gradients():
    rng = np.random.default_rng(0)
    norm = LayerNorm(4)
    norm.gain.data = rng.normal(size=4)
    inputs = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    weights = rng.normal(size=(3, 4))
    assert_gradients_match(lambda: (norm(inputs) * weights).sum(), [inputs, norm.gain, norm.bias])


def test_layer_norm_output_is_normalized():
    output = LayerNorm(6)(Tensor(np.random.default_rng(1).normal(3.0, 5.0, size=(2, 6))))
    np.testing.assert_allclose(output.data.mean(axis=1), 0.0, atol=1e-9)
    np.testing.assert_allclose(output.data.std(axis=1), 1.0, atol=1e-4)


def test_linear_gradients():
    rng = np.random.default_rng(2)
    layer = Linear(3, 2, rng)
    inputs = Tensor(rng.normal(size=(4, 3)), requires_grad=True)
    assert_gradients_match(lambda: (layer(inputs) * layer(inputs)).sum(), [inputs, layer.weight, layer.bias])


def test_full_model_gradients_through_every_parameter():
    rng = np.random.default_rng(4)
    graph = small_graph(rng)
    model = small_model()
    targets = np.array([1.0, 0.0])
    assert_gradients_match(
        lambda: binary_cross_entropy_with_logits(model(graph).logits, targets),
        model.parameters(),
        tolerance=1e-5,
    )


def test_batched_forward_matches_each_graph_alone():
    rng = np.random.default_rng(8)
    graphs = [small_graph(rng) for _ in range(3)]
    graphs.append(StructuralGraph.from_edge_lists(rng.normal(size=(6, 5)), [[(0, 5)], [], [(2, 3), (4, 3)]]))
    model = small_model()
    batch_output = model.forward_batch(StructuralGraph.batch(graphs))

    assert batch_output.logits.shape == (4, 2)
    for index, graph in enumerate(graphs):
        alone = model(graph)
        np.testing.assert_allclose(batch_output.logits.data[index], alone.logits.data, rtol=1e-9, atol=1e-12)
        for decision in range(2):
            assert batch_output.evidence_node(index, decision) == alone.evidence_node(decision)


def test_batched_gradients_through_every_parameter():
    rng = np.random.default_rng(9)
    batch = StructuralGraph.batch([small_graph(rng), small_graph(rng)])
    model = small_model()
    targets = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert_gradients_match(
        lambda: binary_cross_entropy_with_logits(model.forward_batch(batch).logits, targets),
        model.parameters(),
        tolerance=1e-5,
    )


def test_single_graph_call_rejects_a_batch():
    rng = np.random.default_rng(10)
    with pytest.raises(ValueError):
        small_model()(StructuralGraph.batch([small_graph(rng), small_graph(rng)]))


def test_parameter_count():
    model = small_model(relation_count=2)
    assert model.parameter_count() == sum(value.size for value in model.state_dict().values())


def test_model_output_shapes():
    output = small_model()(small_graph(np.random.default_rng(5)))
    assert output.logits.shape == (2,)
    assert output.attention.shape == (4, 2)
    np.testing.assert_allclose(output.attention.sum(axis=0), 1.0)
    assert 0 <= output.evidence_node(1) < 4


def test_reasoning_steps_can_change_at_inference():
    graph = small_graph(np.random.default_rng(6))
    model = small_model(reasoning_steps=2)
    shallow = model(graph, reasoning_steps=1).logits.data
    deep = model(graph, reasoning_steps=6).logits.data
    assert not np.allclose(shallow, deep)
    with pytest.raises(ValueError):
        model(graph, reasoning_steps=0)


def test_model_initialization_is_deterministic_per_seed():
    first, second, other = small_model(seed=9), small_model(seed=9), small_model(seed=10)
    for name, value in first.state_dict().items():
        np.testing.assert_array_equal(value, second.state_dict()[name])
    assert any(not np.array_equal(v, other.state_dict()[k]) for k, v in first.state_dict().items())


def test_model_rejects_graph_with_wrong_shape():
    rng = np.random.default_rng(7)
    with pytest.raises(ValueError):
        small_model()(small_graph(rng, feature_size=6))
    with pytest.raises(ValueError):
        small_model(relation_count=2)(small_graph(rng))


def test_state_dict_round_trip_and_mismatch_detection():
    source, target = small_model(seed=1), small_model(seed=2)
    target.load_state_dict(source.state_dict())
    for name, value in source.state_dict().items():
        np.testing.assert_array_equal(value, target.state_dict()[name])
    with pytest.raises(ValueError):
        target.load_state_dict({})
    with pytest.raises(ValueError):
        small_model(hidden_size=5).load_state_dict(source.state_dict())


def test_parameter_names_are_stable():
    names = [name for name, _ in small_model(relation_count=2).named_parameters()]
    assert names == [
        "encoder.input_projection.weight",
        "encoder.input_projection.bias",
        "encoder.step.relation_weights.0",
        "encoder.step.relation_weights.1",
        "encoder.step.self_update.weight",
        "encoder.step.self_update.bias",
        "encoder.step.normalization.gain",
        "encoder.step.normalization.bias",
        "head.attention_scores.weight",
        "head.attention_scores.bias",
        "head.decision_weights",
        "head.decision_bias",
    ]

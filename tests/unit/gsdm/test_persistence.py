import numpy as np
import pytest

from gsdm.graph import StructuralGraph
from gsdm.nn import GSDMConfig, GSDModel
from gsdm.persistence import ModelBundle, load_bundle, save_bundle


def make_bundle() -> ModelBundle:
    model = GSDModel(GSDMConfig(feature_size=3, relation_count=1, decision_count=2, hidden_size=4, seed=5))
    return ModelBundle(model=model, temperatures=np.array([1.5, 0.8]), metadata={"feature_schema": 1})


def test_round_trip_preserves_weights_temperatures_metadata_and_outputs(tmp_path):
    bundle = make_bundle()
    path = tmp_path / "model.npz"
    save_bundle(path, bundle)
    loaded = load_bundle(path)

    graph = StructuralGraph.from_edge_lists(np.random.default_rng(0).normal(size=(3, 3)), [[(0, 1), (1, 2)]])
    np.testing.assert_array_equal(loaded.model(graph).logits.data, bundle.model(graph).logits.data)
    np.testing.assert_array_equal(loaded.temperatures, bundle.temperatures)
    assert loaded.metadata == {"feature_schema": 1}
    assert loaded.model.config == bundle.model.config


def test_saved_file_contains_no_pickled_objects(tmp_path):
    path = tmp_path / "model.npz"
    save_bundle(path, make_bundle())
    with np.load(path, allow_pickle=False) as archive:
        for key in archive.files:
            assert archive[key].dtype != object


def test_save_rejects_wrong_number_of_temperatures(tmp_path):
    bundle = make_bundle()
    bundle.temperatures = np.array([1.0])
    with pytest.raises(ValueError):
        save_bundle(tmp_path / "model.npz", bundle)

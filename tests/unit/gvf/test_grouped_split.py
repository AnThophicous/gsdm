import numpy as np
import pytest

from gsdm.graph import StructuralGraph
from gsdm.training import TrainingExample
from gvf.infrastructure.model.training_pipeline import (
    DatasetError,
    _grouped_split,
    categories_without_validation_positives,
)


def examples(count: int) -> list[TrainingExample]:
    graph = StructuralGraph.from_edge_lists(np.ones((1, 1)), [[]])
    return [TrainingExample(graph, np.array([float(i)])) for i in range(count)]


def groups_of(split: list[TrainingExample], groups: list[str], pool: list[TrainingExample]) -> set[str]:
    return {groups[pool.index(example)] for example in split}


def test_groups_never_cross_the_split():
    pool = examples(30)
    groups = [f"g{i // 3}" for i in range(30)]
    training, validation = _grouped_split(pool, groups, 0.3, seed=1)
    assert len(training) + len(validation) == 30
    assert groups_of(training, groups, pool).isdisjoint(groups_of(validation, groups, pool))
    assert len(validation) == 9


def test_split_is_deterministic_per_seed():
    pool, groups = examples(20), [f"g{i % 7}" for i in range(20)]
    first = _grouped_split(pool, groups, 0.25, seed=3)[1]
    second = _grouped_split(pool, groups, 0.25, seed=3)[1]
    assert [e.labels[0] for e in first] == [e.labels[0] for e in second]


def test_training_is_never_left_empty():
    pool, groups = examples(4), ["big", "big", "big", "small"]
    training, validation = _grouped_split(pool, groups, 0.9, seed=0)
    assert training and validation


def test_reports_categories_missing_from_validation():
    graph = StructuralGraph.from_edge_lists(np.ones((1, 1)), [[]])
    validation = [TrainingExample(graph, np.array([1.0, 0.0, 0.0])), TrainingExample(graph, np.array([0.0, 0.0, 0.0]))]
    assert categories_without_validation_positives(validation) == ["VC-2", "VC-3"]
    assert categories_without_validation_positives([]) == []


def test_invalid_splits_are_rejected():
    with pytest.raises(DatasetError):
        _grouped_split(examples(4), ["one"] * 4, 0.5, seed=0)
    with pytest.raises(DatasetError):
        _grouped_split(examples(4), ["a", "b", "c", "d"], 1.0, seed=0)
    assert _grouped_split(examples(3), ["a", "b", "c"], 0.0, seed=0)[1] == []

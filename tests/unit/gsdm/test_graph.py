import numpy as np
import pytest

from gsdm.graph import Relation, StructuralGraph


def test_relation_weights_average_over_in_degree():
    relation = Relation.from_edges([(0, 2), (1, 2), (2, 0)], node_count=3)
    np.testing.assert_allclose(relation.weights, [0.5, 0.5, 1.0])


def test_relation_accepts_no_edges():
    relation = Relation.from_edges([], node_count=3)
    assert relation.edge_count == 0


def test_relation_rejects_edges_outside_the_graph():
    with pytest.raises(ValueError):
        Relation.from_edges([(0, 5)], node_count=3)


def test_graph_exposes_its_shape():
    graph = StructuralGraph.from_edge_lists(np.zeros((4, 3)), [[(0, 1)], [], [(3, 2)]])
    assert (graph.node_count, graph.feature_size, graph.relation_count) == (4, 3, 3)


def test_graph_rejects_empty_or_malformed_features():
    with pytest.raises(ValueError):
        StructuralGraph.from_edge_lists(np.zeros((0, 3)), [])
    with pytest.raises(ValueError):
        StructuralGraph(np.zeros(3), [])


def test_batch_shifts_edges_and_keeps_one_segment_per_graph():
    first = StructuralGraph.from_edge_lists(np.ones((2, 3)), [[(0, 1)], [(1, 0)]])
    second = StructuralGraph.from_edge_lists(np.zeros((3, 3)), [[(0, 2), (1, 2)], []])
    batch = StructuralGraph.batch([first, second])

    assert (batch.node_count, batch.graph_count) == (5, 2)
    np.testing.assert_array_equal(batch.segments.sizes, [2, 3])
    np.testing.assert_array_equal(batch.relations[0].sources, [0, 2, 3])
    np.testing.assert_array_equal(batch.relations[0].targets, [1, 4, 4])
    np.testing.assert_allclose(batch.relations[0].weights, [1.0, 0.5, 0.5])
    np.testing.assert_array_equal(batch.relations[1].sources, [1])


def test_batch_of_batches_keeps_every_graph_separate():
    graph = StructuralGraph.from_edge_lists(np.ones((2, 1)), [[(0, 1)]])
    nested = StructuralGraph.batch([StructuralGraph.batch([graph, graph]), graph])
    np.testing.assert_array_equal(nested.segments.sizes, [2, 2, 2])


def test_batch_rejects_incompatible_graphs():
    with pytest.raises(ValueError):
        StructuralGraph.batch([])
    with pytest.raises(ValueError):
        StructuralGraph.batch(
            [StructuralGraph.from_edge_lists(np.ones((2, 3)), [[]]), StructuralGraph.from_edge_lists(np.ones((2, 4)), [[]])]
        )


def test_graph_rejects_relation_built_for_a_larger_graph():
    relation = Relation.from_edges([(0, 4)], node_count=5)
    with pytest.raises(ValueError):
        StructuralGraph(np.zeros((2, 1)), [relation])

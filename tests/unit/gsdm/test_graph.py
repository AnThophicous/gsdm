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


def test_graph_rejects_relation_built_for_a_larger_graph():
    relation = Relation.from_edges([(0, 4)], node_count=5)
    with pytest.raises(ValueError):
        StructuralGraph(np.zeros((2, 1)), [relation])

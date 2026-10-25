import ast

import numpy as np
import pytest

from gvf.domain import CodeAnalysisError, CodeUnit, SkipReason
from gvf.infrastructure.python_graph import FEATURE_SIZE, RELATIONS, PythonGraphBuilder
from gvf.infrastructure.python_graph.features import (
    NODE_TYPES,
    identifier_tokens,
    shannon_entropy,
    token_bucket,
)


def build(source: str):
    return PythonGraphBuilder().build(CodeUnit("sample.py", source))


def node_index(graph, source: str, predicate) -> list[int]:
    """Indices follow the builder's preorder, which matches a preorder walk skipping folded nodes."""
    from gvf.infrastructure.python_graph.features import FOLDED_NODE_KINDS

    order, pending = [], [ast.parse(source)]
    while pending:
        node = pending.pop()
        order.append(node)
        children = [c for c in ast.iter_child_nodes(node) if not isinstance(c, FOLDED_NODE_KINDS)]
        pending.extend(reversed(children))
    assert len(order) == graph.structure.node_count
    return [index for index, node in enumerate(order) if predicate(node)]


def edges(graph, relation: str) -> set[tuple[int, int]]:
    rel = graph.structure.relations[RELATIONS.index(relation)]
    return set(zip(rel.sources.tolist(), rel.targets.tolist()))


def test_graph_has_one_node_per_unfolded_ast_node_and_six_relations():
    graph = build("x = 1\nprint(x)\n")
    assert graph.structure.relation_count == len(RELATIONS) == 6
    assert graph.structure.feature_size == FEATURE_SIZE
    assert len(graph.locations) == graph.structure.node_count
    for relation in ("ast_child", "next_statement", "definition_to_use"):
        assert edges(graph, relation) == {(t, s) for s, t in edges(graph, RELATIONS[RELATIONS.index(relation) + 1])}


def test_context_and_operator_nodes_are_folded_into_parent_features():
    graph = build("y = a + b\n")
    binop = node_index(graph, "y = a + b\n", lambda n: isinstance(n, ast.BinOp))[0]
    features = graph.structure.node_features[binop]
    assert features[NODE_TYPES.index("BinOp")] == 1.0
    assert features[NODE_TYPES.index("Add")] == 1.0


def test_import_alias_flows_to_its_use():
    source = "import pickle\n\ndef load(blob):\n    return pickle.loads(blob)\n"
    graph = build(source)
    alias = node_index(graph, source, lambda n: isinstance(n, ast.alias))[0]
    use = node_index(graph, source, lambda n: isinstance(n, ast.Name) and n.id == "pickle")[0]
    argument = node_index(graph, source, lambda n: isinstance(n, ast.arg))[0]
    blob_use = node_index(graph, source, lambda n: isinstance(n, ast.Name) and n.id == "blob")[0]
    assert (alias, use) in edges(graph, "definition_to_use")
    assert (argument, blob_use) in edges(graph, "definition_to_use")


def test_reassignment_reads_the_previous_definition():
    source = "q = 'a'\nq = q + user\nrun(q)\n"
    graph = build(source)
    first, second = node_index(graph, source, lambda n: isinstance(n, ast.Name) and n.id == "q" and isinstance(n.ctx, ast.Store))
    read_in_second, read_in_run = node_index(
        graph, source, lambda n: isinstance(n, ast.Name) and n.id == "q" and isinstance(n.ctx, ast.Load)
    )
    flows = edges(graph, "definition_to_use")
    assert (first, read_in_second) in flows
    assert (second, read_in_run) in flows
    assert (second, read_in_second) not in flows


def test_consecutive_statements_are_linked():
    source = "a = 1\nb = 2\nc = 3\n"
    graph = build(source)
    statements = node_index(graph, source, lambda n: isinstance(n, ast.Assign))
    assert edges(graph, "next_statement") == {(statements[0], statements[1]), (statements[1], statements[2])}


def test_locations_point_to_source_positions():
    source = "import os\nos.system(cmd)\n"
    graph = build(source)
    call = node_index(graph, source, lambda n: isinstance(n, ast.Call))[0]
    assert (graph.locations[call].line, graph.locations[call].column) == (2, 0)
    assert (graph.locations[0].line, graph.locations[0].column) == (1, 0)


def test_syntax_errors_become_analysis_errors():
    with pytest.raises(CodeAnalysisError) as raised:
        build("def broken(:\n")
    assert raised.value.reason is SkipReason.SYNTAX_ERROR
    with pytest.raises(CodeAnalysisError):
        build("x = 1\x00")


def test_empty_module_still_produces_a_graph():
    assert build("").structure.node_count == 1


def test_features_are_deterministic_across_builds():
    source = "API_KEY = 'sk_live_51HdF8s9xQ'\n"
    np.testing.assert_array_equal(build(source).structure.node_features, build(source).structure.node_features)


def test_identifier_tokens_split_snake_and_camel_case():
    assert identifier_tokens("API_KEY") == ["api_key", "api", "key"]
    assert identifier_tokens("cursorExecute") == ["cursorexecute", "cursor", "execute"]
    assert identifier_tokens("HTTPServer") == ["httpserver", "http", "server"]


def test_token_bucket_is_stable():
    assert token_bucket("execute") == token_bucket("execute")
    assert 0 <= token_bucket("loads") < 256


def test_shannon_entropy():
    assert shannon_entropy("aaaa") == 0.0
    assert shannon_entropy("abcd") == pytest.approx(2.0)


def test_high_entropy_strings_get_higher_entropy_feature():
    secret = build("k = 'xK9#qL2$vB7!mZ4@'\n").structure.node_features
    plain = build("k = 'aaaaaaaaaaaaaaaaa'\n").structure.node_features
    entropy_column = FEATURE_SIZE - 8 + 2
    assert secret[:, entropy_column].max() > plain[:, entropy_column].max()

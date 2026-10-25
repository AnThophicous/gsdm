# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Python source -> GSDM structural graph: AST, statement sequence and simplified def-use relations."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np

from gsdm.graph import StructuralGraph
from gvf.domain import CodeAnalysisError, CodeUnit, SkipReason, SourceLocation
from gvf.infrastructure.python_graph.features import FOLDED_NODE_KINDS, encode_node

RELATIONS = (
    "ast_child",
    "ast_parent",
    "next_statement",
    "previous_statement",
    "definition_to_use",
    "use_to_definition",
)

_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

Position = tuple[int, int]


@dataclass(frozen=True, eq=False)
class PythonCodeGraph:
    structure: StructuralGraph
    locations: tuple[SourceLocation, ...]


class PythonGraphBuilder:
    def build(self, code_unit: CodeUnit) -> PythonCodeGraph:
        try:
            tree = ast.parse(code_unit.source, filename=code_unit.path)
        except (SyntaxError, ValueError) as error:
            raise CodeAnalysisError(code_unit.path, SkipReason.SYNTAX_ERROR) from error
        return _GraphConstruction(tree).result()


@dataclass(frozen=True)
class _Visit:
    node: ast.AST
    parent_index: int | None
    scope: int
    statement_start: Position
    location: Position


@dataclass(frozen=True, order=True)
class _Definition:
    position: Position
    index: int


@dataclass(frozen=True)
class _Use:
    index: int
    scope: int
    name: str
    statement_start: Position


@dataclass
class _GraphConstruction:
    tree: ast.Module
    features: list[np.ndarray] = field(default_factory=list)
    locations: list[SourceLocation] = field(default_factory=list)
    index_of: dict[int, int] = field(default_factory=dict)
    child_edges: list[tuple[int, int]] = field(default_factory=list)
    statement_sequences: list[list[ast.stmt]] = field(default_factory=list)
    definitions: dict[tuple[int, str], list[_Definition]] = field(default_factory=dict)
    uses: list[_Use] = field(default_factory=list)
    parent_scope: dict[int, int | None] = field(default_factory=lambda: {0: None})

    def result(self) -> PythonCodeGraph:
        self._traverse()
        statement_edges = self._statement_edges()
        flow_edges = self._definition_use_edges()
        edge_lists = [
            self.child_edges,
            _reversed(self.child_edges),
            statement_edges,
            _reversed(statement_edges),
            flow_edges,
            _reversed(flow_edges),
        ]
        structure = StructuralGraph.from_edge_lists(np.array(self.features), edge_lists)
        return PythonCodeGraph(structure=structure, locations=tuple(self.locations))

    def _traverse(self) -> None:
        pending = [_Visit(self.tree, None, scope=0, statement_start=(1, 0), location=(1, 0))]
        while pending:
            visit = pending.pop()
            pending.extend(reversed(self._visit(visit)))

    def _visit(self, visit: _Visit) -> list[_Visit]:
        node = visit.node
        index = len(self.features)
        self.index_of[id(node)] = index
        if visit.parent_index is not None:
            self.child_edges.append((visit.parent_index, index))

        location = _position_of(node) or visit.location
        statement_start = _position_of(node) if isinstance(node, ast.stmt) else visit.statement_start
        children = list(ast.iter_child_nodes(node))
        folded = [child for child in children if isinstance(child, FOLDED_NODE_KINDS)]
        self.features.append(encode_node(node, folded))
        self.locations.append(SourceLocation(line=max(1, location[0]), column=max(0, location[1])))
        self._record_statement_sequences(node)
        self._record_names(node, index, visit.scope, statement_start)

        child_scope = self._new_scope(visit.scope) if isinstance(node, _SCOPE_NODES) else visit.scope
        return [
            _Visit(child, index, child_scope, statement_start, location)
            for child in children
            if not isinstance(child, FOLDED_NODE_KINDS)
        ]

    def _new_scope(self, parent: int) -> int:
        scope = len(self.parent_scope)
        self.parent_scope[scope] = parent
        return scope

    def _record_statement_sequences(self, node: ast.AST) -> None:
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and len(value) > 1 and all(isinstance(item, ast.stmt) for item in value):
                self.statement_sequences.append(value)

    def _record_names(self, node: ast.AST, index: int, scope: int, statement_start: Position) -> None:
        position = _position_of(node) or statement_start
        for name in _defined_names(node):
            self.definitions.setdefault((scope, name), []).append(_Definition(position, index))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            self.uses.append(_Use(index, scope, node.id, statement_start))

    def _statement_edges(self) -> list[tuple[int, int]]:
        return [
            (self.index_of[id(current)], self.index_of[id(following)])
            for sequence in self.statement_sequences
            for current, following in zip(sequence, sequence[1:])
        ]

    def _definition_use_edges(self) -> list[tuple[int, int]]:
        edges = []
        for use in self.uses:
            definition = self._reaching_definition(use)
            if definition is not None:
                edges.append((definition.index, use.index))
        return edges

    def _reaching_definition(self, use: _Use) -> _Definition | None:
        local = self.definitions.get((use.scope, use.name), [])
        preceding = [definition for definition in local if definition.position < use.statement_start]
        if preceding:
            return max(preceding)
        if local:
            return max(local)
        scope = self.parent_scope[use.scope]
        while scope is not None:
            enclosing = self.definitions.get((scope, use.name))
            if enclosing:
                return max(enclosing)
            scope = self.parent_scope[scope]
        return None


def _defined_names(node: ast.AST) -> list[str]:
    match node:
        case ast.Name(id=name, ctx=ast.Store()):
            return [name]
        case ast.arg(arg=name):
            return [name]
        case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
            return [name]
        case ast.alias(name=name, asname=alias) if name != "*":
            return [alias or name.split(".")[0]]
    return []


def _position_of(node: ast.AST) -> Position | None:
    line = getattr(node, "lineno", None)
    return None if line is None else (line, getattr(node, "col_offset", 0))


def _reversed(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(target, source) for source, target in edges]

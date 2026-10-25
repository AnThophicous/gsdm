# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Node feature encoding, schema version 1 (ADR-0008, item 9). Changing anything here requires a new version."""

from __future__ import annotations

import ast
import math
import re
import zlib
from collections import Counter

import numpy as np

FEATURE_SCHEMA_VERSION = 1

NODE_TYPES = (
    "Module", "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Delete", "Assign", "AugAssign",
    "AnnAssign", "TypeAlias", "For", "AsyncFor", "While", "If", "With", "AsyncWith", "Match", "Raise", "Try",
    "TryStar", "Assert", "Import", "ImportFrom", "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
    "BoolOp", "NamedExpr", "BinOp", "UnaryOp", "Lambda", "IfExp", "Dict", "Set", "ListComp", "SetComp",
    "DictComp", "GeneratorExp", "Await", "Yield", "YieldFrom", "Compare", "Call", "FormattedValue",
    "Interpolation", "JoinedStr", "TemplateStr", "Constant", "Attribute", "Subscript", "Starred", "Name",
    "List", "Tuple", "Slice", "Load", "Store", "Del", "And", "Or", "Add", "Sub", "Mult", "MatMult", "Div",
    "Mod", "Pow", "LShift", "RShift", "BitOr", "BitXor", "BitAnd", "FloorDiv", "Invert", "Not", "UAdd",
    "USub", "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn", "comprehension",
    "ExceptHandler", "arguments", "arg", "keyword", "alias", "withitem", "match_case", "MatchValue",
    "MatchSingleton", "MatchSequence", "MatchMapping", "MatchClass", "MatchStar", "MatchAs", "MatchOr",
    "TypeVar", "ParamSpec", "TypeVarTuple",
)  # fmt: skip

TOKEN_BUCKETS = 256
LITERAL_STATISTICS = 8
MAXIMUM_STRING_WORDS = 16

_TYPE_INDEX = {name: index for index, name in enumerate(NODE_TYPES)}
_UNKNOWN_TYPE_OFFSET = len(NODE_TYPES)
_TOKEN_OFFSET = _UNKNOWN_TYPE_OFFSET + 1
_LITERAL_OFFSET = _TOKEN_OFFSET + TOKEN_BUCKETS
FEATURE_SIZE = _LITERAL_OFFSET + LITERAL_STATISTICS

FOLDED_NODE_KINDS = (ast.expr_context, ast.operator, ast.boolop, ast.unaryop, ast.cmpop)

_NON_ALPHANUMERIC = re.compile(r"[^0-9A-Za-z]+")
_CAMEL_CASE_PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_WORD = re.compile(r"[a-z]+")


def encode_node(node: ast.AST, folded_children: list[ast.AST]) -> np.ndarray:
    features = np.zeros(FEATURE_SIZE)
    for kind in [node, *folded_children]:
        features[_type_feature_index(kind)] = 1.0
    for token in node_tokens(node):
        features[_TOKEN_OFFSET + token_bucket(token)] = 1.0
    features[_LITERAL_OFFSET:] = literal_statistics(node)
    return features


def node_tokens(node: ast.AST) -> list[str]:
    tokens = [token for identifier in _identifiers(node) for token in identifier_tokens(identifier)]
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        tokens += [f"str:{word}" for word in _WORD.findall(_as_text(node.value).lower())[:MAXIMUM_STRING_WORDS]]
    return tokens


def identifier_tokens(identifier: str) -> list[str]:
    parts = [part.lower() for chunk in _NON_ALPHANUMERIC.split(identifier) for part in _CAMEL_CASE_PART.findall(chunk)]
    return list(dict.fromkeys([identifier.lower(), *parts]))


def token_bucket(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) % TOKEN_BUCKETS


def literal_statistics(node: ast.AST) -> np.ndarray:
    statistics = np.zeros(LITERAL_STATISTICS)
    if not isinstance(node, ast.Constant):
        return statistics
    if isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool):
        statistics[7] = 1.0
    if isinstance(node.value, (str, bytes)):
        statistics[:7] = _text_statistics(_as_text(node.value))
    return statistics


def _text_statistics(text: str) -> list[float]:
    length = len(text)
    if length == 0:
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        1.0,
        min(1.0, math.log1p(length) / math.log1p(512)),
        min(1.0, shannon_entropy(text) / 8.0),
        sum(c.isdigit() for c in text) / length,
        sum(c.isupper() for c in text) / length,
        sum(not c.isalnum() and not c.isspace() for c in text) / length,
        sum(c.isspace() for c in text) / length,
    ]


def shannon_entropy(text: str) -> float:
    counts = Counter(text)
    total = len(text)
    return -sum(count / total * math.log2(count / total) for count in counts.values())


def _identifiers(node: ast.AST) -> list[str]:
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name) | ast.arg(arg=name):
            return [name]
        case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
            return [name]
        case ast.keyword(arg=str() as name):
            return [name]
        case ast.alias(name=name, asname=alias):
            return [name] if alias is None else [name, alias]
        case ast.ImportFrom(module=str() as module):
            return [module]
        case ast.Global(names=names) | ast.Nonlocal(names=names):
            return list(names)
    return []


def _type_feature_index(node: ast.AST) -> int:
    return _TYPE_INDEX.get(type(node).__name__, _UNKNOWN_TYPE_OFFSET)


def _as_text(value: str | bytes) -> str:
    return value.decode("latin-1") if isinstance(value, bytes) else value

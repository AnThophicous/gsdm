# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Measures graph building, inference and training throughput on this machine.

Run: python benchmarks/benchmark.py
"""

from __future__ import annotations

import contextlib
import io
import platform
import time
import tracemalloc

import numpy as np

import gsdm
from gsdm.autograd import no_grad
from gsdm.optim import Adam
from gsdm.training import Trainer, TrainingExample
from gvf.domain import CodeUnit
from gvf.infrastructure.model.detector import new_model
from gvf.infrastructure.python_graph import PythonGraphBuilder

FUNCTION_TEMPLATE = '''
def handler_{index}(cursor, request, payload_{index}):
    user_{index} = request.args.get("user")
    query = "SELECT * FROM table_{index} WHERE name = " + user_{index}
    cursor.execute(query)
    data = pickle.loads(payload_{index})
    token = "sk_live_{index:08d}abcdef"
    if data and token:
        return [item for item in data if item]
    return None
'''

TRAINING_GRAPHS = 64
EPOCHS = 2


def synthetic_source(target_lines: int) -> str:
    header = "import pickle\nimport os\n"
    lines_per_function = FUNCTION_TEMPLATE.count("\n")
    functions = [FUNCTION_TEMPLATE.format(index=i) for i in range(target_lines // lines_per_function)]
    return header + "".join(functions)


def simd_level() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_runtime()
    text = buffer.getvalue()
    start = text.find("'simd_extensions'")
    return " ".join(text[start : text.find("}", start) + 1].split()) if start >= 0 else "unknown"


def measure(label: str, action, repeat: int = 3) -> float:
    durations = []
    for _ in range(repeat):
        started = time.perf_counter()
        action()
        durations.append(time.perf_counter() - started)
    best = min(durations)
    print(f"{label:<44} {best * 1000:10.1f} ms")
    return best


def main() -> None:
    print(f"gsdm {gsdm.__version__} | numpy {np.__version__} | python {platform.python_version()}")
    print(f"cpu: {platform.processor()} | {simd_level()}\n")

    builder = PythonGraphBuilder()
    large = CodeUnit("large.py", synthetic_source(2000))
    graph = builder.build(large).structure
    edges = sum(relation.edge_count for relation in graph.relations)
    print(f"2,000-line file: {graph.node_count:,} nodes, {edges:,} edges")
    measure("build graph (2,000 lines)", lambda: builder.build(large))

    model = new_model(hidden_size=64, reasoning_steps=4, seed=0)
    parameters = sum(p.data.size for p in model.parameters())
    print(f"model: {parameters:,} parameters, dtype {model.parameters()[0].data.dtype}")

    def infer() -> None:
        with no_grad():
            model(graph)

    measure("inference (2,000 lines, 4 reasoning steps)", infer)
    tracemalloc.start()
    infer()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    print(f"{'inference peak memory':<44} {peak / 2**20:10.1f} MB")

    small = [builder.build(CodeUnit(f"f{i}.py", FUNCTION_TEMPLATE.format(index=i))).structure for i in range(TRAINING_GRAPHS)]
    examples = [TrainingExample(g, np.array([1.0, 1.0, 1.0])) for g in small]
    trainer = Trainer(model, Adam(model.parameters(), learning_rate=1e-3), batch_size=32)
    seconds = measure(f"train {EPOCHS} epochs x {TRAINING_GRAPHS} graphs (batch 32)", lambda: trainer.fit(examples, EPOCHS), repeat=1)
    print(f"{'training throughput':<44} {EPOCHS * TRAINING_GRAPHS / seconds:10.1f} graphs/s")


if __name__ == "__main__":
    main()

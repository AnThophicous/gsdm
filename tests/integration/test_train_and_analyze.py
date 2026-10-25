"""End to end through the composition root: train -> save -> load -> analyze -> XML."""

import json
import string
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest

from gsdm.persistence import load_bundle, save_bundle
from gvf.domain import CodeUnit, Confidence
from gvf.infrastructure.model.detector import GsdmVulnerabilityDetector, IncompatibleModelError
from gvf.main import EXIT_CLEAN, EXIT_ERROR, EXIT_FINDINGS, main

VULNERABLE = {
    "VC-1": [
        'def find(cursor, {v}):\n    query = "SELECT * FROM {t} WHERE id = " + {v}\n    cursor.execute(query)\n',
        'import os\n\ndef ping({v}):\n    os.system("ping -c 1 " + {v})\n',
        'import subprocess\n\ndef listing({v}):\n    subprocess.call("ls " + {v}, shell=True)\n',
    ],
    "VC-2": [
        "import pickle\n\ndef restore({v}):\n    return pickle.loads({v})\n",
        "import yaml\n\ndef parse({v}):\n    return yaml.load({v})\n",
    ],
    "VC-3": [
        'API_KEY = "{secret}"\n',
        'import db\n\npassword = "{secret}"\ndb.connect(user="admin", password=password)\n',
    ],
}

CLEAN = [
    'def find(cursor, {v}):\n    cursor.execute("SELECT * FROM {t} WHERE id = %s", ({v},))\n',
    'import subprocess\n\ndef ping({v}):\n    subprocess.run(["ping", "-c", "1", {v}])\n',
    "import json\n\ndef restore({v}):\n    return json.loads({v})\n",
    "import yaml\n\ndef parse({v}):\n    return yaml.safe_load({v})\n",
    'import os\n\nAPI_KEY = os.environ["API_KEY"]\n',
    'greeting = "hello {t}"\nprint(greeting)\n',
]

NAMES = [("user_id", "users"), ("payload", "orders"), ("value", "items")]
DATASET_SIZE = (sum(len(templates) for templates in VULNERABLE.values()) + len(CLEAN)) * len(NAMES)


def write_dataset(directory: Path) -> Path:
    rng = np.random.default_rng(0)
    alphabet = list(string.ascii_letters + string.digits)
    entries = []
    templates = [(t, [code]) for code, ts in VULNERABLE.items() for t in ts] + [(t, []) for t in CLEAN]
    for template_index, (template, labels) in enumerate(templates):
        for variant, (variable, table) in enumerate(NAMES):
            secret = "".join(rng.choice(alphabet, size=32))
            relative = f"sample_{template_index}_{variant}.py"
            (directory / relative).write_text(template.format(v=variable, t=table, secret=secret), encoding="utf-8")
            entries.append(json.dumps({"path": relative, "labels": labels}))
    manifest = directory / "manifest.jsonl"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dataset")
    manifest = write_dataset(directory)
    model = directory / "gvf.npz"
    exit_code = main([
        "train", str(manifest), "--output", str(model), "--epochs", "40", "--hidden-size", "16",
        "--reasoning-steps", "2", "--learning-rate", "0.01", "--validation-fraction", "0", "--seed", "1",
    ])  # fmt: skip
    assert exit_code == EXIT_CLEAN
    return directory, manifest, model


def test_trained_model_fits_its_training_set(trained):
    directory, manifest, model = trained
    detector = GsdmVulnerabilityDetector.from_file(model)
    threshold = Confidence(0.5)
    correct = total = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        source = (directory / entry["path"]).read_text(encoding="utf-8")
        for assessment in detector.assess(CodeUnit(entry["path"], source)):
            correct += assessment.is_flagged(threshold) == (assessment.category.code in entry["labels"])
            total += 1
    assert correct / total >= 0.95


def test_analyze_prints_schema_v1_xml_and_signals_findings(trained, capsys):
    directory, _, model = trained
    exit_code = main(["analyze", str(directory), "--model", str(model)])
    document = capsys.readouterr().out
    root = ElementTree.fromstring(document.split("\n", 1)[1])

    assert exit_code == EXIT_FINDINGS
    assert root.attrib["schema-version"] == "1"
    files = root.findall("file")
    assert len(files) == DATASET_SIZE
    for file in files:
        assert [a.attrib["category"] for a in file.findall("assessment")] == ["VC-1", "VC-2", "VC-3"]
        for assessment in file.findall("assessment"):
            assert 0.0 <= float(assessment.attrib["confidence"]) <= 1.0
            assert assessment.find("location") is not None


def test_analyze_reports_clean_code_with_clean_exit(trained, tmp_path, capsys):
    _, _, model = trained
    clean = tmp_path / "clean.py"
    clean.write_text('greeting = "hello users"\nprint(greeting)\n', encoding="utf-8")
    assert main(["analyze", str(clean), "--model", str(model)]) == EXIT_CLEAN


def test_unparseable_files_are_reported_as_skipped(trained, tmp_path, capsys):
    _, _, model = trained
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    main(["analyze", str(tmp_path), "--model", str(model)])
    root = ElementTree.fromstring(capsys.readouterr().out.split("\n", 1)[1])
    assert root.find("skipped").attrib["reason"] == "syntax-error"


def test_training_with_validation_reports_calibration(trained, tmp_path, capsys):
    _, manifest, _ = trained
    exit_code = main([
        "train", str(manifest), "--output", str(tmp_path / "m.npz"), "--epochs", "2", "--hidden-size", "8",
        "--reasoning-steps", "1", "--validation-fraction", "0.3",
    ])  # fmt: skip
    output = capsys.readouterr().out
    assert exit_code == EXIT_CLEAN
    assert f"validation_examples: {round(0.3 * DATASET_SIZE)}" in output
    assert "calibration_error_after:" in output
    temperatures = load_bundle(tmp_path / "m.npz").temperatures
    assert temperatures.shape == (3,) and np.all(temperatures > 0)


def test_incompatible_model_is_refused(trained, tmp_path, capsys):
    _, _, model = trained
    bundle = load_bundle(model)
    bundle.metadata["feature_schema_version"] = 999
    tampered = tmp_path / "tampered.npz"
    save_bundle(tampered, bundle)
    with pytest.raises(IncompatibleModelError):
        GsdmVulnerabilityDetector.from_file(tampered)
    assert main(["analyze", str(tmp_path), "--model", str(tampered)]) == EXIT_ERROR


def test_missing_path_and_bad_manifest_are_errors(trained, tmp_path, capsys):
    _, _, model = trained
    assert main(["analyze", str(tmp_path / "nowhere"), "--model", str(model)]) == EXIT_ERROR
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"path": "x.py", "labels": ["VC-9"]}\n', encoding="utf-8")
    assert main(["train", str(bad), "--output", str(tmp_path / "o.npz")]) == EXIT_ERROR
    assert "manifest line 1" in capsys.readouterr().err

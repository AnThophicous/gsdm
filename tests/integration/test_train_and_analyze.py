"""End to end through the composition root: train -> save -> load -> analyze -> XML."""

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pytest

from benchmarks.synthetic_dataset import NAMES, template_count, write_dataset
from gsdm.persistence import load_bundle, save_bundle
from gvf.domain import CodeUnit, Confidence
from gvf.infrastructure.model.detector import GsdmVulnerabilityDetector, IncompatibleModelError
from gvf.main import EXIT_CLEAN, EXIT_ERROR, EXIT_FINDINGS, main

DATASET_SIZE = template_count() * len(NAMES)


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


def test_training_writes_logs_dashboard_and_checkpoint(trained):
    directory, _, model = trained
    run = directory / "gvf.run"
    lines = (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
    assert json.loads(lines[-1])["epoch"] == 40
    assert "<svg" in (run / "dashboard.html").read_text(encoding="utf-8")
    assert (run / "checkpoint.npz").is_file()


def test_resume_continues_the_same_run(trained, tmp_path, capsys):
    _, manifest, _ = trained
    output = tmp_path / "m.npz"
    common = ["--output", str(output), "--hidden-size", "8", "--reasoning-steps", "1", "--validation-fraction", "0.3"]
    assert main(["train", str(manifest), "--epochs", "2", "--patience", "0", *common]) == EXIT_CLEAN
    capsys.readouterr()
    assert main(["train", str(manifest), "--epochs", "4", "--resume", *common]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "resuming from checkpoint" in out
    assert "epochs_run: 4" in out
    metrics = (tmp_path / "m.run" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in metrics] == [1, 2, 3, 4]


def test_resume_refuses_a_changed_manifest(trained, tmp_path, capsys):
    directory, manifest, _ = trained
    copy = directory / "copy.jsonl"
    copy.write_text(manifest.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "m.npz"
    common = ["--output", str(output), "--hidden-size", "8", "--reasoning-steps", "1"]
    assert main(["train", str(copy), "--epochs", "1", *common]) == EXIT_CLEAN
    copy.write_text("\n".join(manifest.read_text(encoding="utf-8").splitlines()[:-1]) + "\n", encoding="utf-8")
    assert main(["train", str(copy), "--epochs", "2", "--resume", *common]) == EXIT_ERROR
    assert "manifest changed" in capsys.readouterr().err
    assert main(["train", str(copy), "--epochs", "2", "--resume", "--output", str(tmp_path / "none.npz")]) == EXIT_ERROR


def test_ungrouped_validation_prints_a_warning(trained, tmp_path, capsys):
    directory, manifest, _ = trained
    ungrouped = directory / "ungrouped.jsonl"
    entries = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    ungrouped.write_text("\n".join(json.dumps({k: v for k, v in e.items() if k != "group"}) for e in entries), encoding="utf-8")
    main(["train", str(ungrouped), "--output", str(tmp_path / "u.npz"), "--epochs", "1", "--hidden-size", "8"])
    out = capsys.readouterr().out
    assert "validation is not grouped" in out
    assert "grouped_validation: False" in out


def test_capacity_command(capsys):
    assert main(["capacity", "--hidden-size", "64"]) == EXIT_CLEAN
    assert "parameters" in capsys.readouterr().out
    assert main(["capacity", "--target-parameters", "1e10"]) == EXIT_CLEAN
    out = capsys.readouterr().out
    assert "training state" in out and "GB" in out


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

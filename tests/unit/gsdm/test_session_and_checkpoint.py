import json

import numpy as np
import pytest

from gsdm.checkpoint import CheckpointWriter, load_checkpoint, save_checkpoint
from gsdm.graph import StructuralGraph
from gsdm.nn import GSDMConfig, GSDModel
from gsdm.optim import Adam
from gsdm.session import EpochReport, TrainingObserver, TrainingSession, decision_quality
from gsdm.training import Trainer, TrainingExample, evaluate


def example(rng: np.random.Generator) -> TrainingExample:
    features = rng.normal(size=(4, 3))
    label = float(features[:, 0].sum() > 0)
    graph = StructuralGraph.from_edge_lists(features, [[(0, 1), (1, 2), (2, 3)]])
    return TrainingExample(graph, np.array([label, 1.0 - label]))


def dataset(seed: int, size: int) -> list[TrainingExample]:
    rng = np.random.default_rng(seed)
    return [example(rng) for _ in range(size)]


def new_session(training, validation, patience=None, observers=()) -> TrainingSession:
    model = GSDModel(GSDMConfig(feature_size=3, relation_count=1, decision_count=2, hidden_size=6, reasoning_steps=2, seed=4))
    trainer = Trainer(model, Adam(model.parameters(), learning_rate=0.02), batch_size=4, seed=5)
    return TrainingSession(trainer, training, validation, ["positive", "negative"], observers, patience)


class Recorder(TrainingObserver):
    def __init__(self) -> None:
        self.epochs: list[int] = []
        self.ended = False

    def on_epoch_end(self, report, session) -> None:
        self.epochs.append(report.epoch)

    def on_training_end(self, session) -> None:
        self.ended = True


def test_session_reports_every_epoch_and_notifies_observers():
    recorder = Recorder()
    history = new_session(dataset(1, 16), dataset(2, 8), observers=[recorder]).run(4)

    assert [report.epoch for report in history] == [1, 2, 3, 4] == recorder.epochs
    assert recorder.ended
    first = history[0]
    assert first.validation_loss is not None and first.calibration_error is not None
    assert first.generalization_gap == pytest.approx(first.validation_loss - first.training_loss)
    assert [quality.name for quality in first.decisions] == ["positive", "negative"]
    assert first.graphs_per_second > 0


def test_final_weights_are_the_best_validation_weights():
    validation = dataset(2, 8)
    session = new_session(dataset(1, 16), validation)
    history = session.run(8)
    best = min(history, key=lambda report: report.validation_loss)
    assert session.best_epoch == best.epoch
    session.use_best_weights()
    assert evaluate(session.trainer.model, validation).loss == pytest.approx(best.validation_loss, rel=1e-6)


def test_best_weights_end_the_training(tmp_path):
    session = new_session(dataset(1, 8), dataset(2, 4))
    session.run(2)
    session.use_best_weights()
    assert not session.is_resumable
    with pytest.raises(RuntimeError):
        session.run(4)
    with pytest.raises(ValueError):
        save_checkpoint(tmp_path / "c.npz", session)


def test_early_stopping_after_patience_epochs_without_improvement():
    training = dataset(1, 16)
    shifted = [TrainingExample(e.graph, 1.0 - e.labels) for e in dataset(2, 8)]
    session = new_session(training, shifted, patience=2)
    history = session.run(50)
    assert session.stopped_early
    assert len(history) < 50
    assert history[-1].epoch - session.best_epoch == 2


def test_no_validation_means_no_early_stopping():
    session = new_session(dataset(1, 8), (), patience=1)
    history = session.run(3)
    assert len(history) == 3 and not session.stopped_early
    assert history[0].validation_loss is None and history[0].decisions == ()


def test_resumed_training_is_bit_identical_to_uninterrupted_training(tmp_path):
    training, validation = dataset(1, 16), dataset(2, 8)
    uninterrupted = new_session(training, validation)
    uninterrupted.run(6)

    first_half = new_session(training, validation)
    first_half.run(3)
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, first_half, {"note": "halfway"})

    resumed = new_session(training, validation)
    assert load_checkpoint(path, resumed) == {"note": "halfway"}
    resumed.run(6)

    for name, value in uninterrupted.trainer.model.state_dict().items():
        np.testing.assert_array_equal(value, resumed.trainer.model.state_dict()[name])
    assert [r.to_dict() | {"seconds": 0, "graphs_per_second": 0} for r in resumed.history] == [
        r.to_dict() | {"seconds": 0, "graphs_per_second": 0} for r in uninterrupted.history
    ]


def test_checkpoint_writer_saves_on_interval_and_at_the_end(tmp_path):
    path = tmp_path / "run.npz"
    session = new_session(dataset(1, 8), dataset(2, 4), observers=[CheckpointWriter(path, every=2)])
    session.run(3)
    restored = new_session(dataset(1, 8), dataset(2, 4))
    load_checkpoint(path, restored)
    assert restored.epoch == 3 and len(restored.history) == 3
    assert not (tmp_path / "run.npz.partial").exists()


def test_checkpoint_refuses_a_different_model(tmp_path):
    path = tmp_path / "checkpoint.npz"
    session = new_session(dataset(1, 8), ())
    session.run(1)
    save_checkpoint(path, session)
    model = GSDModel(GSDMConfig(feature_size=3, relation_count=1, decision_count=2, hidden_size=7, seed=4))
    other = TrainingSession(Trainer(model, Adam(model.parameters())), dataset(1, 8))
    with pytest.raises(ValueError):
        load_checkpoint(path, other)


def test_epoch_report_round_trips_through_json():
    report = new_session(dataset(1, 8), dataset(2, 4)).run(1)[0]
    assert EpochReport.from_dict(json.loads(json.dumps(report.to_dict()))) == report


def test_decision_quality_counts():
    logits = np.array([[1.0, -1.0], [1.0, -1.0], [-1.0, -1.0]])
    labels = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    first, second = decision_quality(logits, labels, ["a", "b"])
    assert (first.precision, first.recall, first.f1, first.positives) == (0.5, 0.5, 0.5, 2)
    assert (second.precision, second.recall, second.f1, second.positives) == (None, None, None, 0)

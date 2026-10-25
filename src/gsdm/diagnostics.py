# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Is the model learning patterns or memorizing examples? Heuristic, documented rules (ADR-0009, item 9)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from gsdm.session import EpochReport

MINIMUM_EPOCHS = 3
RISE_SINCE_BEST = 0.10
TRAINING_DROP_SINCE_BEST = 0.10
EPOCHS_PAST_BEST = 2
GAP_RATIO = 3.0
NEAR_PERFECT_TRAINING_LOSS = 0.05
STALL_WINDOW = 5
STALL_IMPROVEMENT = 0.01


class LearningStatus(Enum):
    LEARNING = "learning"
    MEMORIZING = "memorizing"
    STALLED = "stalled"
    NO_VALIDATION = "no-validation"
    TOO_SHORT = "too-short"


@dataclass(frozen=True)
class LearningDiagnosis:
    status: LearningStatus
    best_epoch: int | None
    evidence: str


def diagnose(history: Sequence[EpochReport]) -> LearningDiagnosis:
    validated = [report for report in history if report.validation_loss is not None]
    if not validated:
        return LearningDiagnosis(LearningStatus.NO_VALIDATION, None, "no validation set: memorization cannot be detected")
    best = min(validated, key=lambda report: report.validation_loss)
    if len(validated) < MINIMUM_EPOCHS:
        return LearningDiagnosis(LearningStatus.TOO_SHORT, best.epoch, f"only {len(validated)} validated epochs")
    return _memorization(validated, best) or _stall(validated, best) or _learning(validated, best)


def _memorization(history: Sequence[EpochReport], best: EpochReport) -> LearningDiagnosis | None:
    last = history[-1]
    validation_rise = last.validation_loss / best.validation_loss - 1.0
    training_drop = 1.0 - last.training_loss / best.training_loss if best.training_loss > 0 else 0.0
    if (
        last.epoch - best.epoch >= EPOCHS_PAST_BEST
        and validation_rise > RISE_SINCE_BEST
        and training_drop > TRAINING_DROP_SINCE_BEST
    ):
        return LearningDiagnosis(
            LearningStatus.MEMORIZING,
            best.epoch,
            f"validation loss rose {validation_rise:.0%} since epoch {best.epoch} "
            f"while training loss fell {training_drop:.0%}",
        )
    if last.training_loss < NEAR_PERFECT_TRAINING_LOSS and last.validation_loss > GAP_RATIO * last.training_loss:
        return LearningDiagnosis(
            LearningStatus.MEMORIZING,
            best.epoch,
            f"near-perfect training loss ({last.training_loss:.4f}) but validation loss is "
            f"{last.validation_loss / last.training_loss:.1f}x higher",
        )
    return None


def _stall(history: Sequence[EpochReport], best: EpochReport) -> LearningDiagnosis | None:
    if len(history) < STALL_WINDOW:
        return None
    window = history[-STALL_WINDOW:]
    validation_gain = _relative_gain(window[0].validation_loss, min(r.validation_loss for r in window))
    training_gain = _relative_gain(window[0].training_loss, min(r.training_loss for r in window))
    if validation_gain < STALL_IMPROVEMENT and training_gain < STALL_IMPROVEMENT:
        return LearningDiagnosis(
            LearningStatus.STALLED,
            best.epoch,
            f"less than {STALL_IMPROVEMENT:.0%} improvement in the last {STALL_WINDOW} epochs on both sets",
        )
    return None


def _learning(history: Sequence[EpochReport], best: EpochReport) -> LearningDiagnosis:
    first = history[0]
    gain = _relative_gain(first.validation_loss, best.validation_loss)
    return LearningDiagnosis(
        LearningStatus.LEARNING,
        best.epoch,
        f"validation loss improved {gain:.0%} from epoch {first.epoch} to its best at epoch {best.epoch}",
    )


def _relative_gain(start: float, end: float) -> float:
    return (start - end) / start if start > 0 else 0.0

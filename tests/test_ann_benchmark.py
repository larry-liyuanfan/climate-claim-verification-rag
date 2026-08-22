from __future__ import annotations

import numpy as np
import pytest

from climate_rag.ann_benchmark import recall_at_k


def test_recall_at_k_uses_exact_row_positions() -> None:
    exact = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
    candidate = np.asarray([[1, 3, 9], [4, 8, 9]], dtype=np.int64)

    assert recall_at_k(exact, candidate, 3) == pytest.approx([2 / 3, 1 / 3])


def test_recall_at_k_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="fewer than k"):
        recall_at_k(np.asarray([[1]]), np.asarray([[1]]), 2)

import sys
import types

import numpy as np
import pytest
import torch

from instinct_nav_onboard.dynamic_mask import Yolo11Segmentation, resize_binary_mask


def test_resize_binary_mask_preserves_binary_values():
    mask = np.asarray([[0.0, 1.0], [0.6, 0.4]], dtype=np.float32)
    resized = resize_binary_mask(mask, (4, 4))
    assert resized.shape == (4, 4)
    assert set(np.unique(resized)) == {0.0, 1.0}


def test_yolo11_seg_checks_local_weights_before_importing_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="YOLO11-seg weights"):
        Yolo11Segmentation(tmp_path / "missing-yolo11s-seg.pt")


def test_yolo11_seg_unions_instance_masks_and_forwards_options(tmp_path, monkeypatch):
    weights = tmp_path / "yolo11s-seg.pt"
    weights.touch()
    calls = {}

    class FakeModel:
        def __init__(self, path, task):
            calls["init"] = (path, task)

        def predict(self, **kwargs):
            calls["predict"] = kwargs
            masks = torch.tensor(
                [
                    [[1, 0], [0, 0]],
                    [[0, 0], [0, 1]],
                ],
                dtype=torch.float32,
            )
            return [types.SimpleNamespace(masks=types.SimpleNamespace(data=masks))]

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeModel))
    provider = Yolo11Segmentation(weights, class_ids=(0, 2), device="cpu", half=True)
    result = provider.predict(np.zeros((2, 2, 3), dtype=np.uint8))

    assert calls["init"] == (str(weights.resolve()), "segment")
    assert calls["predict"]["classes"] == [0, 2]
    assert calls["predict"]["quantize"] is None
    assert calls["predict"]["retina_masks"] is True
    np.testing.assert_array_equal(result, np.asarray([[1, 0], [0, 1]], dtype=np.float32))

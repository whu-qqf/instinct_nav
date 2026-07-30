from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = (
    ROOT
    / "source/instinct_nav/instinct_nav/navigation/mdp/navigation/actions"
    / "instinct_frozen_backend.py"
)
OFFICIAL_EXPORTED = ROOT / "data&model/checkpoints/parkour_onboard_preview_stair/exported"


def _load_backend_class():
    spec = importlib.util.spec_from_file_location("instinct_frozen_backend_test", BACKEND_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.OfficialOnnxFrozenPolicyBackend


def test_official_pytorch_backend_matches_fixed_batch_onnx_exports():
    backend_class = _load_backend_class()
    backend = backend_class(
        2,
        29,
        "cpu",
        OFFICIAL_EXPORTED / "actor.onnx",
        OFFICIAL_EXPORTED / "0-depth_encoder.onnx",
    )
    observations = np.random.default_rng(42).normal(
        0.0, 0.5, size=(2, backend.expected_obs_dim())
    ).astype(np.float32)

    depth_session = ort.InferenceSession(
        str(OFFICIAL_EXPORTED / "0-depth_encoder.onnx"), providers=["CPUExecutionProvider"]
    )
    actor_session = ort.InferenceSession(
        str(OFFICIAL_EXPORTED / "actor.onnx"), providers=["CPUExecutionProvider"]
    )
    expected = []
    for sample in observations:
        proprio = sample[: backend.proprio_dim][None]
        depth = sample[backend.proprio_dim :].reshape(1, *backend.depth_shape)
        latent = depth_session.run(None, {"input": depth})[0]
        expected.append(
            actor_session.run(None, {"input": np.concatenate((proprio, latent), axis=1)})[0][0]
        )

    actual = backend.compute_raw_actions(torch.from_numpy(observations)).numpy()
    np.testing.assert_allclose(actual, np.stack(expected), rtol=1.0e-4, atol=1.0e-5)
    assert all(not parameter.requires_grad for parameter in backend.actor.parameters())
    assert all(not parameter.requires_grad for parameter in backend.depth_encoder.parameters())

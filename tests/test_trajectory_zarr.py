from pathlib import Path

import numpy as np
import pytest

from open_physical_sim.logging_zarr import TrajectoryWriter, _HAS_ZARR, _FORMAT


@pytest.mark.skipif(not _HAS_ZARR, reason="zarr not installed")
def test_episodic_layout(tmp_path: Path):
    store = tmp_path / "run.zarr"
    w = TrajectoryWriter(store)
    w.start_episode("ep0")
    for t in range(3):
        w.append(
            obs={"proprio": np.zeros(4, dtype=np.float32)},
            action=np.array([0.1, 0.2], dtype=np.float32),
            reward=0.0,
            terminated=(t == 2),
            truncated=False,
            timestamp=float(t) * 0.02,
        )
    assert w._ep_id is None  # auto-flushed on terminated
    import zarr

    root = zarr.open_group(str(store), mode="r")
    assert root.attrs["format"] == _FORMAT
    ep = root["episodes"]["ep0"]
    assert ep["actions"].shape == (3, 2)
    assert ep["obs"]["proprio"].shape == (3, 4)
    assert ep["timestamps"].shape == (3,)
    assert "step_000000" not in root

from pathlib import Path
import numpy as np
import pytest
from open_physical_sim.logging_zarr import TrajectoryWriter, _HAS_ZARR


@pytest.mark.skipif(not _HAS_ZARR, reason="zarr missing")
def test_info_success_always_written(tmp_path: Path):
    store = tmp_path / "run.zarr"
    w = TrajectoryWriter(store)
    w.start_episode("ep0")
    w.append(obs={"proprio": np.zeros(2)}, action=np.ones(2), reward=0.0, terminated=True)
    import zarr
    ep = zarr.open_group(str(store), mode="r")["episodes"]["ep0"]
    assert "success" in ep["info"]
    assert ep["info"]["success"].shape == (1,)

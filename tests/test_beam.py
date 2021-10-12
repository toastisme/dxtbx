from pathlib import Path

from dxtbx.model.beam import MonoBeamFactory


def test_beam():
    image = str(Path(__file__).parent / "phi_scan_001.cbf")
    assert MonoBeamFactory.imgCIF(image)

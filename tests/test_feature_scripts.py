from pathlib import Path


def test_phase3_scripts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "04_extract_psd_matrices.py").exists()
    assert (root / "scripts" / "05_extract_fc_matrices.py").exists()
    assert (root / "scripts" / "06_build_handcrafted_features.py").exists()


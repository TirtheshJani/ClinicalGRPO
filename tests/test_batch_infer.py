import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent.parent))


def _df(tmp_path, n=2):
    df = pd.DataFrame({
        "discharge_summary": [f"Note {i}." for i in range(n)],
        "icd10_codes": [["E11.9"]] * n,
    })
    p = tmp_path / "in.parquet"
    df.to_parquet(p)
    return p


def test_output_has_predicted_codes_column(tmp_path):
    from unittest.mock import MagicMock
    from scripts.batch_infer import run_batch_infer
    mock_pred = MagicMock()
    mock_pred.predict_batch.return_value = [["E11.9"], ["I10"]]
    out = tmp_path / "out.parquet"
    run_batch_infer(mock_pred, _df(tmp_path), out)
    result = pd.read_parquet(out)
    assert "predicted_codes" in result.columns
    assert len(result) == 2


def test_output_preserves_existing_columns(tmp_path):
    from unittest.mock import MagicMock
    from scripts.batch_infer import run_batch_infer
    df = pd.DataFrame({"discharge_summary": ["Note."], "icd10_codes": [["I10"]], "subject_id": [99]})
    p = tmp_path / "in.parquet"
    df.to_parquet(p)
    mock_pred = MagicMock()
    mock_pred.predict_batch.return_value = [["I10"]]
    out = tmp_path / "out.parquet"
    run_batch_infer(mock_pred, p, out)
    result = pd.read_parquet(out)
    assert "subject_id" in result.columns
    assert result["subject_id"].iloc[0] == 99


def test_batch_infer_cli_help():
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/batch_infer.py", "--help"],
        capture_output=True, text=True, cwd="/home/user/ClinicalGRPO"
    )
    assert r.returncode == 0
    assert "adapter" in r.stdout.lower()

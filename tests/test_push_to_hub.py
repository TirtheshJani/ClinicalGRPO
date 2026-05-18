# tests/test_push_to_hub.py
import sys
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cli_help():
    import subprocess
    r = subprocess.run(
        [sys.executable, "scripts/push_to_hub.py", "--help"],
        capture_output=True, text=True, cwd="/home/user/ClinicalGRPO"
    )
    assert r.returncode == 0
    assert "repo" in r.stdout.lower()


def test_push_calls_upload_folder(tmp_path):
    from unittest.mock import patch, MagicMock
    from scripts.push_to_hub import push_adapter

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{}')

    mock_api = MagicMock()
    with patch("scripts.push_to_hub.HfApi", return_value=mock_api):
        push_adapter(str(adapter_dir), "user/test-repo", private=True)

    mock_api.create_repo.assert_called_once_with(
        repo_id="user/test-repo", private=True, exist_ok=True, repo_type="model"
    )
    call_kw = mock_api.upload_folder.call_args.kwargs
    assert call_kw["repo_id"] == "user/test-repo"
    assert call_kw["folder_path"] == str(adapter_dir)


def test_push_raises_on_missing_adapter(tmp_path):
    from scripts.push_to_hub import push_adapter
    with pytest.raises(FileNotFoundError):
        push_adapter(str(tmp_path / "nonexistent"), "user/repo")

from pathlib import Path
import subprocess
import sys


def test_operational_scripts_do_not_start_work_on_import():
    root = Path(__file__).resolve().parents[1]
    code = "import sys; sys.argv = ['import-check', '--unexpected']; import run_paper_neural_session; import verify_paper_runtime; print('import-safe')"
    result = subprocess.run([sys.executable, '-c', code], cwd=root,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'import-safe'

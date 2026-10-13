import os
import subprocess
import sys
import psutil
import pytest
from lab.process import bounded_run


def test_timeout_stops_experiment_descendant(tmp_path):
    pid_file = tmp_path / 'child.pid'
    script = (
        'import subprocess,sys,time,pathlib; '
        'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
        'pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)'
    )
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_run([sys.executable, '-c', script, str(pid_file)], cwd=tmp_path, timeout=2)
    assert pid_file.exists(), 'Fixture child must actually start'
    assert not psutil.pid_exists(int(pid_file.read_text()))
    assert psutil.Process(os.getpid()).is_running()

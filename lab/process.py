"""Terminate only descendants of an experiment owned by this launcher."""
import subprocess
import psutil


def stop_owned(process):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    _, alive = psutil.wait_procs(children, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def bounded_run(command, *, cwd, timeout):
    process = subprocess.Popen(command, cwd=cwd)
    try:
        return process.wait(timeout=timeout)
    except BaseException:
        stop_owned(process)
        raise

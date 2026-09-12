"""Optional rest and temperature hysteresis between CUDA batches."""
import subprocess
import time
from pathlib import Path
import os


def gpu_temperature():
    exe = str(Path('C:/Windows/System32/nvidia-smi.exe')) if os.name == 'nt' else 'nvidia-smi'
    result = subprocess.run([exe, '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    return int(result.stdout.splitlines()[0].strip())


class CoolingLimit(MemoryError):
    pass


class Cooling:
    def __init__(self, config, device):
        self.rest = config.get('batch_rest_seconds', 0) if device == 'cuda' else 0
        self.enabled = device == 'cuda' and (bool(self.rest) or config.get('thermal_control', False))
        self.pause = config.get('cooling_pause_c', 73)
        self.resume = config.get('cooling_resume_c', 60)
        self.last_check = -float('inf')

    def before(self):
        if not self.enabled or time.monotonic() - self.last_check < 5:
            return
        started = time.monotonic()
        temperature = gpu_temperature()
        if temperature >= self.pause:
            print(f'Cooling pause at {temperature} C; waiting for <= {self.resume} C', flush=True)
            while temperature > self.resume:
                if time.monotonic() - started >= 600:
                    raise CoolingLimit('Cooling wait exceeded 600 seconds')
                time.sleep(5)
                temperature = gpu_temperature()
            print(f'Cooling resumed at {temperature} C', flush=True)
        self.last_check = time.monotonic()

    def after(self):
        if self.rest:
            import torch
            torch.cuda.synchronize()
            time.sleep(self.rest)

"""Prepare the pinned upstream checkout; never reset an existing checkout or install packages."""
import argparse
from pathlib import Path
import site
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = '52de18ae6f1ccc7426eb70895c0e43e1552ad493'
UPSTREAM_URL = 'https://github.com/beir-cellar/beir.git'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register', action='store_true',
                        help='Register the checkout in this active virtual environment')
    args = parser.parse_args()
    if args.register and sys.prefix == sys.base_prefix:
        parser.error('--register requires a virtual environment, not system Python')
    checkout = ROOT / 'upstream/beir'
    if checkout.exists():
        commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
        changes = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True)
        if commit != UPSTREAM_COMMIT or changes:
            raise RuntimeError('Existing upstream differs or has changes; refusing to modify it')
    else:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', UPSTREAM_URL, str(checkout)], check=True)
        subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', UPSTREAM_COMMIT], check=True)
    if args.register:
        candidates = [Path(p) for p in site.getsitepackages() if Path(p).is_relative_to(Path(sys.prefix))]
        if not candidates:
            raise RuntimeError('No site-packages directory in the active virtual environment')
        target = candidates[0] / 'beir_paper_source.pth'
        target.write_text(str(checkout.resolve()) + '\n', encoding='utf-8')
        print('Registered pinned upstream in the active virtual environment')
    print('Upstream:', UPSTREAM_COMMIT, checkout)


if __name__ == '__main__':
    main()

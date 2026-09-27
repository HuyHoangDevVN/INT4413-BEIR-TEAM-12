"""Read-only runtime/source verification for the owned paper container."""
import hashlib
import argparse
from pathlib import Path
import subprocess
from lab.cache import atomic_json

def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=root / 'reports/paper_reproduction/runtime_verification.json')
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(root / 'reports')
    evidence = {}
    for name in ['main.py', 'config.py']:
        actual = subprocess.check_output(['docker', 'exec', 'beir-paper-20211021', 'cat', '/home/' + name], text=True)
        expected = (root / 'upstream/beir/examples/beir-pyserini' / name).read_text(encoding='utf-8')
        assert actual.replace('\r\n', '\n') == expected.replace('\r\n', '\n')
        evidence[name] = dict(matches_upstream=True, normalized_sha256=hashlib.sha256(actual.replace('\r\n', '\n').encode()).hexdigest())
    evidence['java'] = subprocess.check_output(['docker', 'exec', 'beir-paper-20211021', 'java', '-version'], stderr=subprocess.STDOUT, text=True)
    assert '11.0.11' in evidence['java']
    evidence['pyserini'] = subprocess.check_output(['docker', 'exec', 'beir-paper-20211021', 'python', '-c', "import pkg_resources; print(pkg_resources.get_distribution('pyserini'))"], text=True).strip()
    assert evidence['pyserini'] == 'pyserini 0.12.0'
    atomic_json(args.output, evidence)
    print(evidence)


if __name__ == '__main__':
    main()

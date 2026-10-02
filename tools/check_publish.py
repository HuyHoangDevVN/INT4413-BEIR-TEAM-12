"""Check staged publication contents without displaying matched secrets."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {'data', 'cache', 'runs', 'reports', 'report_handoff', 'upstream', '.venv', '.venv-paper', '.ssh'}
PATTERN = re.compile(rb'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{25,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    errors = []
    count = 0
    total = 0
    for name in filter(None, names):
        path = Path(name)
        if path.parts[0] in BLOCKED or path.name.startswith('.env') or path.suffix in {'.pem', '.key', '.pfx'}:
            errors.append('Excluded artifact is tracked: ' + name)
            continue
        blob = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        count += 1
        total += len(blob)
        if len(blob) > 5 * 1024 * 1024:
            errors.append('Large file: ' + name)
        if PATTERN.search(blob):
            errors.append('Credential pattern in file: ' + name)
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'Publication check passed: {count} files, {total} source bytes; no excluded artifacts or credential-pattern findings')


if __name__ == '__main__':
    main()

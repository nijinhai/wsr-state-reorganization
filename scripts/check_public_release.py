"""Inspect the allowlisted public tree; report filenames, never secret values."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.md', '.py', '.json', '.csv', '.toml', '.yml', '.yaml', '.txt'}
SECRET_PATTERNS = [r'gh[pousr]_[A-Za-z0-9]{20,}', r'github_pat_[A-Za-z0-9_]{30,}',
                   r'hf_[A-Za-z0-9]{20,}', r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----']


def files():
    allowed_dirs = ['wsr', 'tests', 'docs', 'results', 'third_party', 'scripts', '.github']
    paths = [p for p in ROOT.iterdir() if p.is_file() and not p.name.startswith('.env')]
    for name in allowed_dirs:
        paths += [p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    return paths


def main():
    errors, checked = [], 0
    for p in files():
        if p.suffix in {'.pt', '.pth', '.bin', '.safetensors', '.parquet', '.arrow'}:
            errors.append(f'Undistributable generated asset: {p.relative_to(ROOT)}')
        if p.stat().st_size > 5*1024*1024:
            errors.append(f'Unexpected large file: {p.relative_to(ROOT)}')
        if p.suffix not in TEXT_SUFFIXES:
            continue
        text = p.read_text()
        checked += 1
        if p.name != 'check_public_release.py':
            for pattern in SECRET_PATTERNS:
                if re.search(pattern, text):
                    errors.append(f'Credential-like content: {p.relative_to(ROOT)}')
            if re.search(r'/(?:home|mnt/[a-z])/[^\s)\]"\']+', text):
                errors.append(f'Local absolute path: {p.relative_to(ROOT)}')
        if p.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', text):
                target = target.split('#')[0]
                if not target or '://' in target or target.startswith('mailto:'):
                    continue
                if not (p.parent/target).exists():
                    errors.append(f'Broken link: {p.relative_to(ROOT)} -> {target}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'PASS: inspected {checked} public text files; links/assets/credential patterns checked.')


if __name__ == '__main__':
    main()

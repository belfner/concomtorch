#!/usr/bin/env python3
"""
Move newly built wheels into the public-served wheelhouse and regenerate PyG-style HTML indexes.

For each (torch_minor, cuda) pair present in the wheelhouse, emits an HTML page listing every
wheel that matches, suitable for `pip install --find-links` consumption.
"""
from __future__ import annotations

import argparse
import html
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from plan import (
    WheelKey,
    parse_wheel,
)

INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>
"""


def collect(serve_root: Path) -> dict[tuple[str, str], list[Path]]:
    """
    Group wheels in serve_root/files/ by (torch_minor, cuda).
    """
    files_dir = serve_root / 'files'
    if not files_dir.is_dir():
        return {}
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for wheel in sorted(files_dir.glob('*.whl')):
        key = parse_wheel(wheel.name)
        if key is None:
            continue
        groups[(key.torch_minor, key.cuda)].append(wheel)
    return groups


def write_torch_cuda_page(serve_root: Path, torch_minor: str, cuda: str, wheels: list[Path]) -> Path:
    """
    Render one HTML page listing wheels for a specific (torch_minor, cuda) pair.
    """
    out_path = serve_root / f'torch-{torch_minor}+{cuda}.html'
    links = '\n'.join(
        f'<a href="files/{html.escape(w.name)}">{html.escape(w.name)}</a><br>'
        for w in wheels
    )
    title = f'concomtorch wheels for torch {torch_minor} + {cuda}'
    out_path.write_text(INDEX_TEMPLATE.format(title=title, body=links), encoding='utf-8')
    return out_path


def write_root_index(serve_root: Path, groups: dict[tuple[str, str], list[Path]]) -> Path:
    """
    Render the top-level index linking to every (torch_minor, cuda) page.
    """
    out_path = serve_root / 'index.html'
    rows = []
    for (torch_minor, cuda), _wheels in sorted(groups.items()):
        page = f'torch-{torch_minor}+{cuda}.html'
        rows.append(f'<a href="{html.escape(page)}">{html.escape(f"torch {torch_minor} + {cuda}")}</a><br>')
    body = '\n'.join(rows) if len(rows) > 0 else '<em>(no wheels published yet)</em>'
    out_path.write_text(
        INDEX_TEMPLATE.format(title='concomtorch wheel index', body=body),
        encoding='utf-8',
    )
    return out_path


def move_new_wheels(source: Path, dest_files: Path) -> list[Path]:
    """
    Move every .whl in source into dest_files, returning the list of moved paths.

    Existing files at the destination are overwritten so rebuilds replace prior artifacts.
    """
    if not source.is_dir():
        return []
    dest_files.mkdir(parents=True, exist_ok=True)
    moved = []
    for wheel in source.glob('*.whl'):
        target = dest_files / wheel.name
        shutil.move(str(wheel), str(target))
        moved.append(target)
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'wheelhouse',
                        help='Local build output directory.')
    parser.add_argument('--serve-root', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'public',
                        help='Public-served root, with files/ and <torch+cuda>.html pages.')
    args = parser.parse_args()

    moved = move_new_wheels(args.source, args.serve_root / 'files')
    print(f'Moved {len(moved)} wheels into {args.serve_root / "files"}')

    groups = collect(args.serve_root)
    for (torch_minor, cuda), wheels in groups.items():
        page = write_torch_cuda_page(args.serve_root, torch_minor, cuda, wheels)
        print(f'Wrote {page} ({len(wheels)} wheels)')

    root = write_root_index(args.serve_root, groups)
    print(f'Wrote {root}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

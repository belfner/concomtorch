#!/usr/bin/env python3
"""
Move newly built wheels into the public-served root and regenerate the PEP 503 index tree.

The serve root exposes a two-layer index: ``<cuda>/<torch_tag>/``. The first layer is the
cuda variant, the second is the torch minor (e.g. ``cu126/torch2_6/``). Each
``<cuda>/<torch_tag>/`` directory is a PEP 503 simple index that pip consumes via
``--extra-index-url``. The user selects the cuda and torch minor that match their installed
PyTorch and passes that two-layer URL directly; the wheel for that bucket is then the only
``concomtorch`` candidate, so a bare ``pip install concomtorch`` resolves it.

Layout::

    public/
      index.html                           landing page listing every cuda variant
      cu126/
        index.html                         browsable page listing torch channels
        torch2_6/
          index.html                       PEP 503 root index (lists 'concomtorch/')
          concomtorch/
            index.html                     PEP 503 project page (lists wheel files)
        torch2_7/
          ...
      cu128/
        ...
      files/
        concomtorch-0.1.0+cu126torch2.6.1-cp310-cp310-manylinux_2_28_x86_64.whl
        ...
"""
from __future__ import annotations

import argparse
import html
import shutil
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from plan import parse_wheel
from plan import torch_minor

PROJECT_NAME = 'concomtorch'

PROJECT_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="pypi:repository-version" content="1.0">
<title>Links for {project}</title>
</head>
<body>
<h1>Links for {project}</h1>
{body}
</body>
</html>
"""

TORCH_ROOT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="pypi:repository-version" content="1.0">
<title>concomtorch wheel channel: {cuda} torch {torch}</title>
</head>
<body>
<h1>concomtorch wheel channel: {cuda} torch {torch}</h1>
<a href="{project}/">{project}/</a><br>
</body>
</html>
"""

CUDA_INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>concomtorch wheel channels: {cuda}</title>
</head>
<body>
<h1>concomtorch wheel channels: {cuda}</h1>
<p>Pick the torch channel that matches your installed torch minor and pass that
two-layer URL as <code>--extra-index-url</code> to pip.</p>
{body}
</body>
</html>
"""

LANDING_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>concomtorch wheel index</title>
</head>
<body>
<h1>concomtorch wheel index</h1>
<p>Pick the cuda variant and torch minor that match your installed PyTorch, append
both to the URL of this page, and pass that as <code>--extra-index-url</code> to pip.
For example, if this page is served at <code>https://wheels.example.com/</code> and you
run CUDA 12.6 with torch 2.6:</p>
<pre>pip install concomtorch --extra-index-url https://wheels.example.com/cu126/torch2_6/</pre>
<h2>CUDA variants</h2>
{body}
</body>
</html>
"""


def torch_dir_name(minor: str) -> str:
    """
    Map a torch minor like ``2.6`` to its index directory name ``torch2_6``.

    The dotted minor cannot be a path segment verbatim because a leading-dot or
    dotted directory is awkward to serve and browse, so the dot is replaced with an
    underscore and a ``torch`` prefix disambiguates it from the cuda layer.

    Parameters
    ----------
    minor : str
        Torch minor version, e.g. ``2.6``.

    Returns
    -------
    str
        Directory name, e.g. ``torch2_6``.
    """
    return f'torch{minor.replace(".", "_")}'


def _minor_sort_key(minor: str) -> tuple[int, ...]:
    """
    Sort key that orders torch minors numerically (so ``2.10`` follows ``2.9``).

    Parameters
    ----------
    minor : str
        Torch minor version, e.g. ``2.6``.

    Returns
    -------
    tuple[int, ...]
        Integer tuple of the dotted components.
    """
    return tuple(int(part) for part in minor.split('.'))


def collect(serve_root: Path) -> dict[tuple[str, str], list[Path]]:
    """
    Group wheels in ``serve_root/files/`` by ``(cuda, torch_minor)``.

    Parameters
    ----------
    serve_root : Path
        Public serve root containing a ``files/`` subdirectory of wheels.

    Returns
    -------
    dict[tuple[str, str], list[Path]]
        Mapping from ``(cuda, torch_minor)`` (e.g. ``('cu126', '2.6')``) to the wheels for
        that bucket, sorted by filename for stable index output.
    """
    return collect_from(serve_root / 'files')


def collect_from(files_dir: Path) -> dict[tuple[str, str], list[Path]]:
    """
    Group wheels in an arbitrary directory by ``(cuda, torch_minor)``.

    Parameters
    ----------
    files_dir : Path
        Directory of wheel files to group.

    Returns
    -------
    dict[tuple[str, str], list[Path]]
        Mapping from ``(cuda, torch_minor)`` to the wheels for that bucket, sorted by
        filename.
    """
    if not files_dir.is_dir():
        return {}
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for wheel in sorted(files_dir.glob('*.whl')):
        key = parse_wheel(wheel.name)
        if key is None:
            continue
        groups[(key.cuda, torch_minor(key.torch))].append(wheel)
    return dict(groups)


def _write_project_page(
    serve_root: Path,
    cuda: str,
    minor: str,
    wheels: list[Path],
    wheel_href: Callable[[Path], str] | str,
) -> Path:
    """
    Render the PEP 503 project page for one ``(cuda, torch_minor)`` bucket.

    Parameters
    ----------
    serve_root : Path
        Public serve root.
    cuda : str
        Cuda variant tag (e.g. ``cu126``).
    minor : str
        Torch minor version (e.g. ``2.6``).
    wheels : list[Path]
        Wheel paths belonging to this bucket.
    wheel_href : Callable[[Path], str] or str
        Per-wheel URL builder. When a string is passed it is treated as a directory prefix
        and concatenated with each wheel filename (the local-serve layout where wheels live
        in ``<serve_root>/files/``). When a callable is passed it is invoked once per wheel
        and must return the full ``href`` URL; this form is used when wheels are hosted on
        per-(cuda, torch_minor) GitHub Releases.

    Returns
    -------
    Path
        Path of the written ``<cuda>/<torch_tag>/<project>/index.html``.
    """
    out_dir = serve_root / cuda / torch_dir_name(minor) / PROJECT_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'index.html'

    if isinstance(wheel_href, str):
        prefix = wheel_href.rstrip('/')

        def href_for(w: Path) -> str:
            return f'{prefix}/{w.name}'
    else:
        href_for = wheel_href

    links = '\n'.join(
        f'<a href="{html.escape(href_for(w))}">{html.escape(w.name)}</a><br>'
        for w in wheels
    )
    out_path.write_text(
        PROJECT_PAGE_TEMPLATE.format(project=PROJECT_NAME, body=links),
        encoding='utf-8',
    )
    return out_path


def _write_torch_root(serve_root: Path, cuda: str, minor: str) -> Path:
    """
    Render the PEP 503 root index for one ``(cuda, torch_minor)`` bucket.

    Parameters
    ----------
    serve_root : Path
        Public serve root.
    cuda : str
        Cuda variant tag.
    minor : str
        Torch minor version.

    Returns
    -------
    Path
        Path of the written ``<cuda>/<torch_tag>/index.html``.
    """
    out_dir = serve_root / cuda / torch_dir_name(minor)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'index.html'
    out_path.write_text(
        TORCH_ROOT_TEMPLATE.format(
            cuda=html.escape(cuda),
            torch=html.escape(minor),
            project=PROJECT_NAME,
        ),
        encoding='utf-8',
    )
    return out_path


def _write_cuda_index(serve_root: Path, cuda: str, minors: list[str]) -> Path:
    """
    Render the browsable page for one cuda variant listing its torch channels.

    Parameters
    ----------
    serve_root : Path
        Public serve root.
    cuda : str
        Cuda variant tag.
    minors : list[str]
        Torch minors available for this cuda variant.

    Returns
    -------
    Path
        Path of the written ``<cuda>/index.html``.
    """
    out_dir = serve_root / cuda
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'index.html'
    rows = [
        f'<a href="{html.escape(torch_dir_name(m))}/">torch {html.escape(m)}</a><br>'
        for m in sorted(minors, key=_minor_sort_key)
    ]
    out_path.write_text(
        CUDA_INDEX_TEMPLATE.format(cuda=html.escape(cuda), body='\n'.join(rows)),
        encoding='utf-8',
    )
    return out_path


def _write_landing(serve_root: Path, cudas: list[str]) -> Path:
    """
    Render the human-facing landing page at the serve root listing every cuda variant.

    Parameters
    ----------
    serve_root : Path
        Public serve root.
    cudas : list[str]
        Cuda variants present in the wheelhouse.

    Returns
    -------
    Path
        Path of the written ``index.html``.
    """
    out_path = serve_root / 'index.html'
    if len(cudas) == 0:
        body = '<em>(no wheels published yet)</em>'
    else:
        rows = [
            f'<a href="{html.escape(cuda)}/">{html.escape(cuda)}/</a><br>'
            for cuda in sorted(cudas)
        ]
        body = '\n'.join(rows)
    out_path.write_text(LANDING_TEMPLATE.format(body=body), encoding='utf-8')
    return out_path


def render_index_tree(
    serve_root: Path,
    groups: dict[tuple[str, str], list[Path]],
    wheel_href: Callable[[Path], str] | str = '../../../files',
) -> None:
    """
    Write the full two-layer PEP 503 index tree for ``groups`` into ``serve_root``.

    For each ``(cuda, torch_minor)`` bucket this writes the project page and its PEP 503
    root, then one browsable index per cuda variant and a landing page over all variants.

    Parameters
    ----------
    serve_root : Path
        Output root (``public/`` for local serve, the gh-pages worktree for github-pages).
    groups : dict[tuple[str, str], list[Path]]
        Mapping from ``(cuda, torch_minor)`` to the wheels for that bucket.
    wheel_href : Callable[[Path], str] or str
        Per-wheel URL builder passed through to each project page. The string default is
        the local-serve relative path from ``<cuda>/<torch_tag>/<project>/`` up to
        ``<serve_root>/files/``. Pass a callable for absolute hosts (e.g. GitHub Releases).
    """
    minors_by_cuda: dict[str, list[str]] = defaultdict(list)
    for (cuda, minor), wheels in sorted(groups.items()):
        _write_project_page(serve_root, cuda, minor, wheels, wheel_href)
        _write_torch_root(serve_root, cuda, minor)
        minors_by_cuda[cuda].append(minor)
        print(f'Rendered {cuda}/{torch_dir_name(minor)} with {len(wheels)} wheel(s).', flush=True)

    for cuda, minors in sorted(minors_by_cuda.items()):
        _write_cuda_index(serve_root, cuda, minors)

    _write_landing(serve_root, list(minors_by_cuda.keys()))


def move_new_wheels(source: Path, dest_files: Path) -> list[Path]:
    """
    Move every ``.whl`` in ``source`` into ``dest_files``, returning the list of moved paths.

    Existing files at the destination are overwritten so rebuilds replace prior artifacts.

    Parameters
    ----------
    source : Path
        Local build output directory.
    dest_files : Path
        Public ``files/`` directory.

    Returns
    -------
    list[Path]
        Destination paths of moved wheels.
    """
    if not source.is_dir():
        return []
    dest_files.mkdir(parents=True, exist_ok=True)
    moved = []
    # Every wheel reaching `source` has already passed the in-container pytest
    # gate: cibuildwheel runs CIBW_TEST_COMMAND_LINUX after auditwheel-repair and
    # before moving the wheel to /output and copying it back to the host, so a
    # failing suite aborts the build before the wheel can land here. Verification
    # is upstream by construction; this mover must not add a second gate.
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
                        help='Public-served root, with files/ and the two-layer PEP 503 tree.')
    parser.add_argument('--files-base-url', default='../../../files',
                        help='URL prefix used in <a href> for each wheel. Defaults to the '
                             "local-serve relative path '../../../files' (from "
                             '<cuda>/<torch_tag>/<project>/ up to <serve-root>/files/). Pass '
                             'an absolute URL to render an index whose wheels live on a '
                             'different host. For per-(cuda, torch_minor) GitHub Releases, '
                             'use ci/release.py which constructs a per-wheel URL builder.')
    parser.add_argument('--skip-move', action='store_true',
                        help='Skip moving wheels into <serve-root>/files/. Use when wheels '
                             'are hosted elsewhere (e.g. GitHub Releases) and only the HTML '
                             'index is written here.')
    args = parser.parse_args()

    if not args.skip_move:
        moved = move_new_wheels(args.source, args.serve_root / 'files')
        print(f'Moved {len(moved)} wheels into {args.serve_root / "files"}')
        groups = collect(args.serve_root)
    else:
        groups = collect_from(args.source)

    render_index_tree(args.serve_root, groups, args.files_base_url)
    print(f'Wrote index tree under {args.serve_root}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

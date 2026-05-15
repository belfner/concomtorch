#!/usr/bin/env python3
"""
Move newly built wheels into the public-served root and regenerate per-cuda PEP 503 indexes.

The serve root mirrors PyTorch's distribution layout: one channel per cuda variant, each a
PEP 503 simple index that pip consumes via ``--extra-index-url``. Users select the cuda
channel matching their installed torch; pip resolves the torch minor automatically from the
``Requires-Dist`` baked into each wheel.

Layout::

    public/
      index.html                         landing page documenting all channels
      cu121/
        index.html                       PEP 503 root index (lists 'concomtorch/')
        concomtorch/
          index.html                     PEP 503 project page (lists wheel files)
      cu124/
        ...
      files/
        concomtorch-0.1.0+cu121torch2.4-cp310-cp310-manylinux_2_28_x86_64.whl
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

from plan import (
    WheelKey,
    parse_wheel,
)

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

CHANNEL_ROOT_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="pypi:repository-version" content="1.0">
<title>concomtorch wheel channel: {cuda}</title>
</head>
<body>
<h1>concomtorch wheel channel: {cuda}</h1>
<a href="{project}/">{project}/</a><br>
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
<p>Pick the channel that matches your installed torch CUDA build, append it to the URL of
this page, and pass that as <code>--extra-index-url</code> to pip. For example, if this
page is served at <code>https://wheels.example.com/</code>:</p>
<pre>pip install concomtorch --extra-index-url https://wheels.example.com/cu121/</pre>
<h2>Channels</h2>
{body}
</body>
</html>
"""


def collect(serve_root: Path) -> dict[str, list[Path]]:
    """
    Group wheels in ``serve_root/files/`` by cuda variant.

    Parameters
    ----------
    serve_root : Path
        Public serve root containing a ``files/`` subdirectory of wheels.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from cuda variant (e.g. 'cu121') to the wheels for that variant, sorted by
        filename for stable index output.
    """
    return collect_from(serve_root / 'files')


def collect_from(files_dir: Path) -> dict[str, list[Path]]:
    """
    Group wheels in an arbitrary directory by cuda variant.

    Parameters
    ----------
    files_dir : Path
        Directory of wheel files to group.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from cuda variant to the wheels for that variant, sorted by filename.
    """
    if not files_dir.is_dir():
        return {}
    groups: dict[str, list[Path]] = defaultdict(list)
    for wheel in sorted(files_dir.glob('*.whl')):
        key = parse_wheel(wheel.name)
        if key is None:
            continue
        groups[key.cuda].append(wheel)
    return dict(groups)


def write_channel_project_page(
    serve_root: Path,
    cuda: str,
    wheels: list[Path],
    wheel_href: Callable[[Path], str] | str = '../../files',
) -> Path:
    """
    Render the PEP 503 project page for one cuda channel.

    Parameters
    ----------
    serve_root : Path
        Public serve root.
    cuda : str
        Cuda variant tag (e.g. 'cu121').
    wheels : list[Path]
        Wheel paths belonging to this channel.
    wheel_href : Callable[[Path], str] or str
        Per-wheel URL builder. When a string is passed it is treated as a directory prefix
        and concatenated with each wheel filename (the local-serve layout where wheels live
        alongside the HTML in ``<serve_root>/files/``). When a callable is passed it is
        invoked once per wheel and must return the full ``href`` URL; this form is used
        when wheels are hosted on different URLs per (cuda, torch_minor), such as separate
        GitHub Releases.

    Returns
    -------
    Path
        Path of the written ``<cuda>/<project>/index.html``.
    """
    out_dir = serve_root / cuda / PROJECT_NAME
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


def write_channel_root(serve_root: Path, cuda: str) -> Path:
    """
    Render the PEP 503 root index for one cuda channel (lists the single 'concomtorch/' project).

    Parameters
    ----------
    serve_root : Path
        Public serve root.
    cuda : str
        Cuda variant tag.

    Returns
    -------
    Path
        Path of the written ``<cuda>/index.html``.
    """
    out_dir = serve_root / cuda
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'index.html'
    out_path.write_text(
        CHANNEL_ROOT_TEMPLATE.format(cuda=html.escape(cuda), project=PROJECT_NAME),
        encoding='utf-8',
    )
    return out_path


def write_landing(serve_root: Path, channels: list[str]) -> Path:
    """
    Render the human-facing landing page at the serve root listing every cuda channel.

    Parameters
    ----------
    serve_root : Path
        Public serve root.
    channels : list[str]
        Cuda variants present in the wheelhouse.

    Returns
    -------
    Path
        Path of the written ``index.html``.
    """
    out_path = serve_root / 'index.html'
    if len(channels) == 0:
        body = '<em>(no wheels published yet)</em>'
    else:
        rows = [
            f'<a href="{html.escape(cuda)}/">{html.escape(cuda)}/</a><br>'
            for cuda in sorted(channels)
        ]
        body = '\n'.join(rows)
    out_path.write_text(LANDING_TEMPLATE.format(body=body), encoding='utf-8')
    return out_path


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
                        help='Public-served root, with files/ and per-cuda PEP 503 indexes.')
    parser.add_argument('--files-base-url', default='../../files',
                        help='URL prefix used in <a href> for each wheel. Defaults to the '
                             "local-serve relative path '../../files'. Pass an absolute URL "
                             '(e.g. a single shared download base) to render an index whose '
                             'wheels live on a different host. For per-(cuda, torch_minor) '
                             'GitHub Releases, use ci/release.py which constructs a per-wheel '
                             'URL builder.')
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

    for cuda, wheels in sorted(groups.items()):
        page = write_channel_project_page(args.serve_root, cuda, wheels, args.files_base_url)
        root = write_channel_root(args.serve_root, cuda)
        print(f'Wrote {root} and {page} ({len(wheels)} wheels)')

    landing = write_landing(args.serve_root, list(groups.keys()))
    print(f'Wrote {landing}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Publish built wheels to GitHub Releases and push the per-cuda PEP 503 index to GitHub Pages.

Wheels are partitioned across one GitHub Release per (cuda, torch_minor) combination, with
tag ``<prefix>-<cuda>-torch<minor>`` (e.g. ``wheels-cu124-torch2.6``). This keeps each release
small enough to stay well within GitHub's per-release asset limits and makes the set of assets
in a given release stable across rebuilds.

Flow:
    1. Resolve the GitHub repo slug (owner/name) from ``git remote get-url origin``.
    2. Group wheels in the wheelhouse by per-(cuda, torch_minor) release tag.
    3. For each bucket, ensure the release exists and upload any new wheels.
    4. Render the per-cuda PEP 503 indexes; each ``<a href>`` points at the release that holds
       that specific wheel,
       ``https://github.com/<owner>/<repo>/releases/download/<prefix>-<cuda>-torch<minor>/<wheel>``.
    5. Sync the rendered tree into a checkout of the ``gh-pages`` branch and push.

Authentication relies on the ``gh`` CLI being logged in. Git pushes use whichever credential
helper the build user has configured for the repo (token, SSH key, or gh auth git-credential).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from plan import (
    parse_wheel,
    torch_minor,
)
from publish import (
    collect_from,
    write_channel_project_page,
    write_channel_root,
    write_landing,
)

CI_DIR = Path(__file__).resolve().parent
REPO_ROOT = CI_DIR.parent
GITHUB_REMOTE_RE = re.compile(
    r'(?:git@(?:github\.com|github-[a-z0-9._-]+):|https://github\.com/)'
    r'(?P<owner>[^/]+)/(?P<name>[^/.\s]+?)(?:\.git)?$'
)


def load_dotenv(path: Path) -> dict[str, str]:
    """
    Load ``KEY=value`` lines from ``path`` into ``os.environ``.

    Values set in the file overwrite any pre-existing process environment so the file is
    authoritative for this repo. Used to source ``GH_TOKEN`` for the per-repo personal GitHub
    account that owns this clone, without globally switching the active ``gh`` keyring user.

    The parser handles plain ``KEY=value``, surrounding single/double quotes, an optional
    ``export`` prefix, comments (lines starting with ``#``), and blank lines. Anything more
    elaborate (variable interpolation, multi-line values, escapes) is intentionally not
    supported; if the file is missing this is a no-op.

    Parameters
    ----------
    path : Path
        Path to the .env file.

    Returns
    -------
    dict[str, str]
        Mapping of keys that were applied to ``os.environ``.
    """
    if not path.is_file():
        return {}
    applied: dict[str, str] = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line == '' or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[len('export '):].lstrip()
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        os.environ[key] = value
        applied[key] = value
    return applied


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    """
    Subprocess wrapper that echoes the command before executing.
    """
    print('>>', ' '.join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=capture, text=True)


def detect_repo_slug() -> tuple[str, str]:
    """
    Return (owner, name) parsed from ``git remote get-url origin``.

    Raises
    ------
    RuntimeError
        When the origin URL does not point at a GitHub repository.
    """
    result = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    url = result.stdout.strip()
    match = GITHUB_REMOTE_RE.search(url)
    if match is None:
        raise RuntimeError(f'origin remote {url!r} does not look like a GitHub repository')
    return match.group('owner'), match.group('name')


def tag_for_wheel(wheel: Path, prefix: str) -> str:
    """
    Build the per-(cuda, torch_minor) release tag for one wheel.

    The release tag is structured as ``<prefix>-<cuda>-torch<torch_minor>`` so each
    (cuda, torch_minor) combination has its own dedicated GitHub Release. This keeps any
    one release small enough to stay within GitHub's per-release asset limit and lets the
    release accumulate multiple patch builds (e.g. ``+cu124torch2.6.0`` and
    ``+cu124torch2.6.1``) under a single stable tag; pip selects the highest version by PEP
    440 ordering of the local segment at install time.

    Parameters
    ----------
    wheel : Path
        Wheel path. Only the filename is consulted.
    prefix : str
        Tag prefix, e.g. 'wheels'.

    Returns
    -------
    str
        Release tag like 'wheels-cu124-torch2.6'.

    Raises
    ------
    ValueError
        When the filename does not match the expected
        '+cu{N}torch{X.Y.Z}' local version pattern.
    """
    key = parse_wheel(wheel.name)
    if key is None:
        raise ValueError(f'unrecognized wheel filename: {wheel.name}')
    return f'{prefix}-{key.cuda}-torch{torch_minor(key.torch)}'


def ensure_release(tag: str, owner: str, name: str) -> None:
    """
    Create the release with the given tag if it does not yet exist.

    Parameters
    ----------
    tag : str
        Release tag, e.g. 'wheels-cu124-torch2.6'.
    owner : str
        Repo owner.
    name : str
        Repo name.
    """
    check = subprocess.run(
        ['gh', 'release', 'view', tag, '--repo', f'{owner}/{name}'],
        capture_output=True, text=True,
    )
    if check.returncode == 0:
        print(f'Release {tag} already exists.', flush=True)
        return
    run([
        'gh', 'release', 'create', tag,
        '--repo', f'{owner}/{name}',
        '--title', f'concomtorch wheels ({tag})',
        '--notes', f'Auto-generated release holding wheels for the {tag} combination. '
                   'Consumed by the concomtorch GitHub Pages PEP 503 index.',
    ])


def list_release_assets(tag: str, owner: str, name: str) -> set[str]:
    """
    Return the set of asset filenames currently attached to the release.
    """
    result = run(
        ['gh', 'release', 'view', tag, '--repo', f'{owner}/{name}', '--json', 'assets'],
        capture=True,
    )
    data = json.loads(result.stdout)
    return {asset['name'] for asset in data.get('assets', [])}


def upload_wheels(wheelhouse: Path, tag_prefix: str, owner: str, name: str) -> list[Path]:
    """
    Group wheels in ``wheelhouse`` by per-(cuda, torch_minor) release tag and upload each.

    Wheels are partitioned into release buckets using :func:`tag_for_wheel`. For each bucket,
    the release is created on demand and wheels whose filenames are not already attached are
    uploaded. Wheels whose filenames are already release assets are skipped: pip clients cache
    by URL, so a same-name-different-bytes rebuild would not be observed by consumers and is
    treated as suspect. ``gh release upload`` is invoked without ``--clobber`` so that any
    filter or pagination bug surfaces as a loud upload error rather than a silent overwrite.

    Wheels with filenames that do not match the expected ``+cu{N}torch{X.Y}`` local-version
    pattern are skipped with a log entry.

    Parameters
    ----------
    wheelhouse : Path
        Directory of wheels to upload.
    tag_prefix : str
        Prefix used to build the per-bucket release tag (e.g. ``'wheels'`` produces
        ``'wheels-cu124-torch2.6'``).
    owner : str
        Repo owner.
    name : str
        Repo name.

    Returns
    -------
    list[Path]
        Wheels that were uploaded this invocation, across all release buckets.
    """
    if not wheelhouse.is_dir():
        return []
    local = sorted(wheelhouse.glob('*.whl'))
    buckets: dict[str, list[Path]] = defaultdict(list)
    unparsable: list[Path] = []
    for wheel in local:
        try:
            tag = tag_for_wheel(wheel, tag_prefix)
        except ValueError:
            unparsable.append(wheel)
            continue
        buckets[tag].append(wheel)

    if len(unparsable) > 0:
        print(
            f'Skipping {len(unparsable)} wheel(s) whose filenames do not match the expected '
            "'+cu{N}torch{X.Y}' local version pattern.",
            flush=True,
        )
        for w in unparsable:
            print(f'  - unparsable filename: {w.name}', flush=True)

    uploaded: list[Path] = []
    for tag in sorted(buckets):
        wheels = buckets[tag]
        ensure_release(tag, owner, name)
        existing = list_release_assets(tag, owner, name)
        new = [w for w in wheels if w.name not in existing]
        skipped = [w for w in wheels if w.name in existing]
        if len(skipped) > 0:
            print(
                f'[{tag}] Skipping {len(skipped)} wheel(s) already on the release. '
                'To replace a published wheel, bump the version (or delete the asset on '
                'GitHub) and rerun.',
                flush=True,
            )
            for w in skipped:
                print(f'  - already on release: {w.name}', flush=True)
        if len(new) == 0:
            print(
                f'[{tag}] No new wheels to upload (release already has {len(existing)} assets).',
                flush=True,
            )
            continue
        print(f'[{tag}] Uploading {len(new)} new wheel(s) to {owner}/{name}.', flush=True)
        run([
            'gh', 'release', 'upload', tag,
            '--repo', f'{owner}/{name}',
            *[str(w) for w in new],
        ])
        uploaded.extend(new)
    return uploaded


def render_pages(
    pages_dir: Path,
    wheelhouse: Path,
    tag_prefix: str,
    owner: str,
    name: str,
) -> None:
    """
    Write the PEP 503 tree into ``pages_dir`` with hrefs pointing at GitHub Release assets.

    Each wheel's href resolves to the release that holds it: a wheel built for cu124/torch2.6
    becomes ``https://github.com/<owner>/<name>/releases/download/<prefix>-cu124-torch2.6/<wheel>``.
    The per-wheel URL is computed via :func:`tag_for_wheel` so the channel index correctly
    fans out to one release per (cuda, torch_minor) bucket.

    Parameters
    ----------
    pages_dir : Path
        Output directory that will be synced to the ``gh-pages`` branch.
    wheelhouse : Path
        Directory of wheels used to enumerate what should appear in each channel index.
    tag_prefix : str
        Prefix used to build the per-bucket release tag.
    owner : str
        Repo owner.
    name : str
        Repo name.
    """
    pages_dir.mkdir(parents=True, exist_ok=True)
    base_download_url = f'https://github.com/{owner}/{name}/releases/download'

    def wheel_href(wheel: Path) -> str:
        tag = tag_for_wheel(wheel, tag_prefix)
        return f'{base_download_url}/{tag}/{wheel.name}'

    groups = collect_from(wheelhouse)
    for cuda, wheels in sorted(groups.items()):
        write_channel_project_page(pages_dir, cuda, wheels, wheel_href)
        write_channel_root(pages_dir, cuda)
        print(f'Rendered {cuda} channel with {len(wheels)} wheel(s).', flush=True)
    write_landing(pages_dir, list(groups.keys()))


def deploy_pages(pages_dir: Path, owner: str, name: str, branch: str = 'gh-pages') -> None:
    """
    Sync ``pages_dir`` into a checkout of the gh-pages branch and push to origin.

    Uses a git worktree under ``.git/concomtorch-gh-pages`` so the host working tree is not
    disturbed. The branch is created as an orphan if it does not yet exist.

    Parameters
    ----------
    pages_dir : Path
        Source tree to publish.
    owner : str
        Repo owner (for log output only).
    name : str
        Repo name (for log output only).
    branch : str
        Target branch on origin.
    """
    git_dir = subprocess.check_output(
        ['git', 'rev-parse', '--git-dir'], cwd=REPO_ROOT, text=True,
    ).strip()
    worktree = Path(git_dir).resolve() / 'concomtorch-gh-pages'

    remote_has_branch = subprocess.run(
        ['git', 'ls-remote', '--exit-code', '--heads', 'origin', branch],
        cwd=REPO_ROOT, capture_output=True,
    ).returncode == 0

    if worktree.exists():
        run(['git', 'worktree', 'remove', '--force', str(worktree)], cwd=REPO_ROOT)

    if remote_has_branch:
        run(['git', 'fetch', 'origin', branch], cwd=REPO_ROOT)
        run(['git', 'worktree', 'add', str(worktree), f'origin/{branch}'], cwd=REPO_ROOT)
        run(['git', 'switch', '-C', branch], cwd=worktree)
    else:
        run(['git', 'worktree', 'add', '--detach', str(worktree), 'HEAD'], cwd=REPO_ROOT)
        run(['git', 'switch', '--orphan', branch], cwd=worktree)
        run(['git', 'rm', '-rf', '.'], cwd=worktree)

    for entry in worktree.iterdir():
        if entry.name == '.git':
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()

    for entry in pages_dir.iterdir():
        target = worktree / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)

    (worktree / '.nojekyll').write_text('', encoding='utf-8')

    run(['git', 'add', '-A'], cwd=worktree)
    status = subprocess.run(
        ['git', 'status', '--porcelain'], cwd=worktree, capture_output=True, text=True,
    )
    if status.stdout.strip() == '':
        print(f'No changes to publish to {owner}/{name}@{branch}.', flush=True)
    else:
        run([
            'git', '-c', 'user.email=concomtorch-ci@localhost',
            '-c', 'user.name=concomtorch CI',
            'commit', '-m', 'Update wheel index',
        ], cwd=worktree)
        run(['git', 'push', 'origin', branch], cwd=worktree)

    run(['git', 'worktree', 'remove', '--force', str(worktree)], cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', type=Path, default=REPO_ROOT / 'wheelhouse',
                        help='Directory of wheels to upload and index.')
    parser.add_argument('--pages-dir', type=Path, default=REPO_ROOT / 'pages',
                        help='Local output directory for the gh-pages tree.')
    parser.add_argument('--tag-prefix', default='wheels',
                        help='Prefix for per-(cuda, torch_minor) release tags. Each combination '
                             "gets its own release named '<prefix>-<cuda>-torch<minor>', e.g. "
                             "'wheels-cu124-torch2.6'.")
    parser.add_argument('--branch', default='gh-pages',
                        help='Branch to push the rendered index to.')
    parser.add_argument('--skip-upload', action='store_true',
                        help='Skip the gh release upload step.')
    parser.add_argument('--skip-deploy', action='store_true',
                        help='Render pages locally but do not push to gh-pages.')
    parser.add_argument('--repo', default=None,
                        help="GitHub repo slug 'owner/name'. Defaults to parsing the origin "
                             'remote with git remote get-url origin.')
    parser.add_argument('--dotenv', type=Path, default=REPO_ROOT / '.env',
                        help='Path to a KEY=value file whose entries are exported into the '
                             "process environment before any 'gh' invocation. Used to supply "
                             'GH_TOKEN scoped to this repo without changing the global gh '
                             'keyring user. Missing file is a no-op.')
    args = parser.parse_args()

    applied = load_dotenv(args.dotenv)
    if 'GH_TOKEN' in applied:
        print(f'Loaded GH_TOKEN from {args.dotenv}.', flush=True)

    if args.repo is not None:
        if '/' not in args.repo:
            raise SystemExit(f"--repo must be in 'owner/name' form, got {args.repo!r}")
        owner, name = args.repo.split('/', 1)
    else:
        owner, name = detect_repo_slug()
    print(f'Repo: {owner}/{name}', flush=True)

    if not args.skip_upload:
        upload_wheels(args.wheelhouse, args.tag_prefix, owner, name)

    if args.pages_dir.exists():
        shutil.rmtree(args.pages_dir)
    render_pages(args.pages_dir, args.wheelhouse, args.tag_prefix, owner, name)

    if not args.skip_deploy:
        deploy_pages(args.pages_dir, owner, name, args.branch)

    return 0


if __name__ == '__main__':
    sys.exit(main())

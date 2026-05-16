#!/usr/bin/env python3
"""
Publish built wheels to GitHub Releases and push the two-layer PEP 503 index tree to GitHub Pages.

Wheels are partitioned across one GitHub Release per (cuda, torch_minor) combination, with
tag ``<prefix>-<cuda>-torch<minor>`` (e.g. ``wheels-cu124-torch2.6``). This keeps each release
small enough to stay well within GitHub's per-release asset limits and makes the set of assets
in a given release stable across rebuilds.

Flow:
    1. Resolve the GitHub repo slug (owner/name) from ``git remote get-url origin``.
    2. Group wheels in the wheelhouse by per-(cuda, torch_minor) release tag.
    3. For each bucket, ensure the release exists and upload any new wheels.
    4. Render the two-layer PEP 503 index tree (per-(cuda, torch_minor) roots under
       ``<cuda>/<torch_tag>/``); each ``<a href>`` points at the release that holds that
       specific wheel,
       ``https://github.com/<owner>/<repo>/releases/download/<prefix>-<cuda>-torch<minor>/<wheel>``.
    5. Sync the rendered tree into a checkout of the ``gh-pages`` branch and push.

Authentication: both ``gh`` (release upload) and ``git push`` (Pages) read ``GH_TOKEN``.
``ci/release.py`` sources ``GH_TOKEN`` from a repo-local ``.env`` (see ``--dotenv``) so a
per-repo token scopes the run while the global ``gh`` keyring user stays as-is, and feeds
that token to the Pages push through a transient ``GIT_ASKPASS`` helper so it stays out of
argv. With ``GH_TOKEN`` unset, ``gh`` uses its logged-in keyring and ``git push`` uses the
``origin`` remote with whichever credential helper the build user has configured.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from loguru import logger

from logging_setup import setup_logging
from plan import (
    parse_wheel,
    torch_minor,
)
from publish import (
    collect_from,
    render_index_tree,
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
    ``export`` prefix, comments (lines starting with ``#``), and blank lines. A missing file
    yields an empty mapping.

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
    logger.info('>> ' + ' '.join(cmd))
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
    # A credential-bearing HTTPS remote (https://<token>@github.com/owner/repo)
    # carries a secret in the URL userinfo. Strip it before matching and before
    # the URL can reach an exception message or the log.
    safe_url = re.sub(r'(https://)[^/@]*@', r'\1', url)
    match = GITHUB_REMOTE_RE.search(safe_url)
    if match is None:
        raise RuntimeError(f'origin remote {safe_url!r} does not look like a GitHub repository')
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
        logger.info(f'Release {tag} already exists.')
        return
    run([
        'gh', 'release', 'create', tag,
        '--repo', f'{owner}/{name}',
        '--title', f'concomtorch wheels ({tag})',
        '--notes', f'Auto-generated release holding wheels for the {tag} combination. '
                   'Consumed by the concomtorch GitHub Pages PEP 503 index.',
    ])


def list_release_assets(tag: str, owner: str, name: str) -> dict[str, int]:
    """
    Return a mapping of asset filename to byte size for the release.

    The size is carried so callers can distinguish a fully-uploaded asset from a
    zero-byte placeholder left by an interrupted ``gh release upload``.

    Parameters
    ----------
    tag : str
        Release tag.
    owner : str
        Repo owner.
    name : str
        Repo name.

    Returns
    -------
    dict[str, int]
        Asset filename to size in bytes.
    """
    result = run(
        ['gh', 'release', 'view', tag, '--repo', f'{owner}/{name}', '--json', 'assets'],
        capture=True,
    )
    data = json.loads(result.stdout)
    return {asset['name']: int(asset.get('size', 0)) for asset in data.get('assets', [])}


def upload_wheels(wheelhouse: Path, tag_prefix: str, owner: str, name: str) -> list[Path]:
    """
    Group wheels in ``wheelhouse`` by per-(cuda, torch_minor) release tag and upload each.

    Wheels are partitioned into release buckets using :func:`tag_for_wheel`. For each bucket,
    the release is created on demand and wheels whose filenames are not already attached are
    uploaded. Fully-uploaded assets (size > 0) of the same name are skipped: pip clients cache
    by URL, so a same-name-different-bytes rebuild would not be observed by consumers and is
    treated as suspect. Brand-new uploads run without ``--clobber`` so any filter or
    pagination bug surfaces as a loud upload error rather than a silent overwrite. A
    zero-byte asset is the signature of an interrupted prior upload and would be a permanent
    dead link in the index; such assets are re-uploaded with ``--clobber`` to repair them.

    A wheel whose filename does not match the expected ``+cu{N}torch{X.Y.Z}`` local-version
    pattern indicates a setup.py metadata regression that would ship a silently-incomplete
    release, so it raises :class:`RuntimeError`.

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

    Raises
    ------
    RuntimeError
        When any wheel filename does not match the expected local-version pattern.
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
        names = ', '.join(w.name for w in unparsable)
        raise RuntimeError(
            f"{len(unparsable)} wheel(s) in {wheelhouse} do not match the expected "
            f"'+cu{{N}}torch{{X.Y.Z}}' local version pattern, which indicates a setup.py "
            f'metadata regression: {names}'
        )

    uploaded: list[Path] = []
    for tag in sorted(buckets):
        wheels = buckets[tag]
        ensure_release(tag, owner, name)
        existing = list_release_assets(tag, owner, name)
        complete = {n for n, size in existing.items() if size > 0}
        fresh = [w for w in wheels if w.name not in existing]
        repair = [w for w in wheels if w.name in existing and w.name not in complete]
        skipped = [w for w in wheels if w.name in complete]
        if len(skipped) > 0:
            logger.info(
                f'[{tag}] Skipping {len(skipped)} wheel(s) already on the release. '
                'To replace a published wheel, bump the version (or delete the asset on '
                'GitHub) and rerun.'
            )
            for w in skipped:
                logger.info(f'  - already on release: {w.name}')
        if len(fresh) == 0 and len(repair) == 0:
            logger.info(
                f'[{tag}] No new wheels to upload (release already has {len(existing)} assets).'
            )
            continue
        if len(fresh) > 0:
            logger.info(f'[{tag}] Uploading {len(fresh)} new wheel(s) to {owner}/{name}.')
            run([
                'gh', 'release', 'upload', tag,
                '--repo', f'{owner}/{name}',
                *[str(w) for w in fresh],
            ])
            uploaded.extend(fresh)
        if len(repair) > 0:
            logger.warning(
                f'[{tag}] Re-uploading {len(repair)} zero-byte asset(s) left by an '
                f'interrupted prior upload.'
            )
            run([
                'gh', 'release', 'upload', tag,
                '--repo', f'{owner}/{name}', '--clobber',
                *[str(w) for w in repair],
            ])
            uploaded.extend(repair)
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

    Each wheel's href resolves to the release that holds it: a wheel built for cu126/torch2.6
    becomes ``https://github.com/<owner>/<name>/releases/download/<prefix>-cu126-torch2.6/<wheel>``.
    The per-wheel URL is computed via :func:`tag_for_wheel` so each (cuda, torch_minor)
    project page points at the matching release.

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
    render_index_tree(pages_dir, groups, wheel_href)


@contextlib.contextmanager
def deploy_lock(git_dir: Path) -> Iterator[None]:
    """
    Serialize gh-pages deploys via an exclusive file lock under the git dir.

    Two overlapping ticks would otherwise race on the same worktree path and on the
    remote branch. The lock is held for the whole worktree lifecycle so a slow deploy
    blocks rather than corrupts a concurrent one.

    Parameters
    ----------
    git_dir : Path
        Absolute path to the repository git directory.

    Yields
    ------
    None
        With the exclusive lock held.
    """
    lock_path = git_dir / 'concomtorch-gh-pages.lock'
    with open(lock_path, 'w', encoding='utf-8') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def push_branch(worktree: Path, branch: str, owner: str, name: str) -> None:
    """
    Push ``branch`` to GitHub, using ``GH_TOKEN`` for auth when it is set.

    ``gh`` reads ``GH_TOKEN`` but ``git push`` does not, so a service user with only
    a token in the environment authenticates the asset upload yet fails the Pages
    push. When ``GH_TOKEN`` is present the push targets the plain
    ``https://github.com/<owner>/<name>.git`` URL and the token is supplied through a
    transient ``GIT_ASKPASS`` helper that reads it from the environment, so the secret
    never appears in argv (and therefore never in a ``CalledProcessError`` traceback or
    the command echo). Otherwise the push uses the ``origin`` remote and whatever
    credential helper the user has configured.

    Parameters
    ----------
    worktree : Path
        Worktree directory to push from.
    branch : str
        Branch to push.
    owner : str
        Repo owner.
    name : str
        Repo name.

    Raises
    ------
    RuntimeError
        When the token-authenticated push fails (with a message that excludes the
        token and git's stderr).
    """
    token = os.environ.get('GH_TOKEN', '').strip()
    if token == '':
        run(['git', 'push', 'origin', branch], cwd=worktree)
        return

    url = f'https://x-access-token@github.com/{owner}/{name}.git'
    askpass = tempfile.NamedTemporaryFile(
        mode='w', suffix='.sh', delete=False, encoding='utf-8',
    )
    try:
        askpass.write('#!/bin/sh\nexec printf %s "$GH_TOKEN"\n')
        askpass.close()
        os.chmod(askpass.name, 0o700)
        env = os.environ.copy()
        env['GIT_ASKPASS'] = askpass.name
        env['GIT_TERMINAL_PROMPT'] = '0'
        logger.info(f'>> git push https://github.com/{owner}/{name}.git {branch} (token auth)')
        proc = subprocess.run(
            ['git', 'push', url, branch],
            cwd=worktree, env=env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f'git push to {owner}/{name}@{branch} failed (exit {proc.returncode}); '
                "git's output is suppressed here to avoid leaking the token-bearing URL"
            )
    finally:
        os.unlink(askpass.name)


def deploy_pages(pages_dir: Path, owner: str, name: str, branch: str = 'gh-pages') -> None:
    """
    Sync ``pages_dir`` into a checkout of the gh-pages branch and push to origin.

    Uses a dedicated git worktree under ``.git/concomtorch-gh-pages`` for the deploy, leaving
    the host working tree as-is. The branch is created as an orphan on first deploy. The
    worktree is created and removed inside a single exclusive deploy lock, and removal runs
    in a ``finally`` so the worktree is always cleaned up, including after a failed push.

    Parameters
    ----------
    pages_dir : Path
        Source tree to publish.
    owner : str
        Repo owner.
    name : str
        Repo name.
    branch : str
        Target branch on origin.
    """
    git_dir = Path(subprocess.check_output(
        ['git', 'rev-parse', '--absolute-git-dir'], cwd=REPO_ROOT, text=True,
    ).strip())
    worktree = git_dir / 'concomtorch-gh-pages'

    with deploy_lock(git_dir):
        remote_has_branch = subprocess.run(
            ['git', 'ls-remote', '--exit-code', '--heads', 'origin', branch],
            cwd=REPO_ROOT, capture_output=True,
        ).returncode == 0

        if worktree.exists():
            run(['git', 'worktree', 'remove', '--force', str(worktree)], cwd=REPO_ROOT)

        try:
            if remote_has_branch:
                run(['git', 'fetch', 'origin', branch], cwd=REPO_ROOT)
                run(['git', 'worktree', 'add', str(worktree), f'origin/{branch}'], cwd=REPO_ROOT)
                run(['git', 'switch', '-C', branch], cwd=worktree)
            else:
                run(['git', 'worktree', 'add', '--detach', str(worktree), 'HEAD'], cwd=REPO_ROOT)
                run(['git', 'switch', '--orphan', branch], cwd=worktree)
                # The orphan index may be empty, in which case 'git rm -rf .' exits with
                # a pathspec error. The physical worktree.iterdir() sweep below removes the
                # carried-over files regardless, so a non-empty index is the only case that
                # needs unstaging here; tolerate the empty-index exit.
                logger.info('>> git rm -rf . (tolerating empty orphan index)')
                subprocess.run(['git', 'rm', '-rf', '.'], cwd=worktree)

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
                logger.info(f'No changes to publish to {owner}/{name}@{branch}.')
            else:
                run([
                    'git', '-c', 'user.email=concomtorch-ci@localhost',
                    '-c', 'user.name=concomtorch CI',
                    'commit', '-m', 'Update wheel index',
                ], cwd=worktree)
                push_branch(worktree, branch, owner, name)
        finally:
            if worktree.exists():
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

    setup_logging('release')

    applied = load_dotenv(args.dotenv)
    if 'GH_TOKEN' in applied:
        logger.info(f'Loaded GH_TOKEN from {args.dotenv}.')

    if args.repo is not None:
        if '/' not in args.repo:
            raise SystemExit(f"--repo must be in 'owner/name' form, got {args.repo!r}")
        owner, name = args.repo.split('/', 1)
    else:
        owner, name = detect_repo_slug()
    logger.info(f'Repo: {owner}/{name}')

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

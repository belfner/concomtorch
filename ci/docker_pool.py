#!/usr/bin/env python3
"""
Manage the cache of manylinux+CUDA build images.

Each cuda variant maps to one image tag of the form `concomtorch-manylinux:{cuda}`. The torch
version is not baked into the image because torch is installed at cibuildwheel time. The pool
builds missing images in parallel up to a cap, and evicts least-recently-used images when the
resident count exceeds another cap.
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

CUDA_TAG_RE = re.compile(r'^cu(?P<major>\d{1,2})(?P<minor>\d)$')

IMAGE_NAMESPACE = 'concomtorch-manylinux'
DEFAULT_MANYLINUX_IMAGE = 'quay.io/pypa/manylinux_2_28_x86_64'


def cuda_tag_parts(cuda_variant: str) -> tuple[str, str]:
    """
    Derive the CUDA version strings the Dockerfile needs from a cuXYZ tag.

    The PyTorch cuda tag scheme is uniform: the trailing digit is the CUDA minor and the
    preceding digits are the major (e.g. ``cu126`` -> 12.6, ``cu132`` -> 13.2), so both the
    dotted version used for ``CUDA_HOME`` and the NVIDIA package suffix are pure functions of
    the tag and require no per-variant table.

    Parameters
    ----------
    cuda_variant : str
        CUDA variant tag, e.g. ``cu126``.

    Returns
    -------
    tuple of (str, str)
        ``(major_minor_dotted, pkg_suffix)``, e.g. ``('12.6', '12-6')``.

    Raises
    ------
    ValueError
        If the tag does not match the ``cu<major><minor>`` scheme.
    """
    m = CUDA_TAG_RE.match(cuda_variant)
    if m is None:
        raise ValueError(f'Malformed cuda variant tag: {cuda_variant!r} (expected cu<major><minor>)')
    major = m.group('major')
    minor = m.group('minor')
    return f'{major}.{minor}', f'{major}-{minor}'


def image_tag(cuda_variant: str) -> str:
    """
    Return the canonical image tag for a cuda variant.
    """
    return f'{IMAGE_NAMESPACE}:{cuda_variant}'


@dataclass
class ImageInfo:
    tag: str
    cuda_variant: str
    image_id: str
    created: datetime
    size_bytes: int


def run(cmd: list[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """
    Subprocess helper that logs the command, respects check, and optionally captures output.
    """
    print('>>', ' '.join(shlex.quote(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def list_resident() -> list[ImageInfo]:
    """
    Return ImageInfo for every image whose tag matches IMAGE_NAMESPACE.

    Uses `docker images --format json` for a stable parse.
    """
    result = subprocess.run(
        ['docker', 'images', '--format', '{{json .}}', f'{IMAGE_NAMESPACE}'],
        check=True, capture_output=True, text=True,
    )
    out: list[ImageInfo] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line == '':
            continue
        row = json.loads(line)
        tag = f'{row["Repository"]}:{row["Tag"]}'
        cuda_variant = row['Tag']
        out.append(ImageInfo(
            tag=tag,
            cuda_variant=cuda_variant,
            image_id=row['ID'],
            created=parse_docker_created(row.get('CreatedAt', '')),
            size_bytes=size_to_bytes(row.get('Size', '0B')),
        ))
    return out


def parse_docker_created(text: str) -> datetime:
    """
    Parse the `docker images` CreatedAt field. Falls back to epoch on unrecognized formats.
    """
    if text == '':
        return datetime.fromtimestamp(0)
    # Docker emits forms like '2026-04-21 12:34:56 +0000 UTC'.
    cleaned = text.replace(' UTC', '').strip()
    for fmt in ('%Y-%m-%d %H:%M:%S %z', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return datetime.fromtimestamp(0)


def size_to_bytes(text: str) -> int:
    """
    Convert a docker size string like '12.3GB' to bytes.
    """
    text = text.strip().upper()
    units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3, 'TB': 1024**4}
    for suffix, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(suffix):
            try:
                return int(float(text[:-len(suffix)]) * mult)
            except ValueError:
                return 0
    return 0


def dockerfile_text(cuda_variant: str, manylinux_image: str = DEFAULT_MANYLINUX_IMAGE) -> str:
    """
    Render the Dockerfile for one cuda variant.

    The host GCC that nvcc accepts is a property of the installed CUDA toolkit,
    encoded in ``$CUDA_HOME/include/crt/host_config.h`` as the ``#if __GNUC__ > N``
    guard. The toolkit is installed first, the cap ``N`` is read from that header,
    and the highest ``gcc-toolset-K`` (``K`` from the cap down to 9) that the RHEL8
    repos offer is installed and exposed through a stable
    ``/opt/rh/gcc-toolset-active`` symlink. A final layer compiles a trivial CUDA
    translation unit with that toolset so an unusable pairing fails the image build
    rather than every wheel build.

    Parameters
    ----------
    cuda_variant : str
        CUDA variant tag, e.g. ``cu126``.
    manylinux_image : str
        Base manylinux image the Dockerfile derives from.

    Returns
    -------
    str
        The rendered Dockerfile contents.
    """
    cuda_major_minor, cuda_pkg_suffix = cuda_tag_parts(cuda_variant)
    return f"""\
# syntax=docker/dockerfile:1
ARG BASE_IMAGE={manylinux_image}
FROM ${{BASE_IMAGE}}

RUN yum -y install dnf-plugins-core curl git && \\
    yum -y config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \\
    yum -y clean all && rm -rf /var/cache/yum

RUN yum -y install cuda-toolkit-{cuda_pkg_suffix} && \\
    yum -y clean all && rm -rf /var/cache/yum

ENV CUDA_HOME=/usr/local/cuda-{cuda_major_minor}
ENV PATH=$CUDA_HOME/bin:$PATH

RUN cap="$(grep -oP '__GNUC__ > \\K[0-9]+' "$CUDA_HOME/include/crt/host_config.h" | head -n1)" && \\
    if [ -z "$cap" ]; then echo "could not read GCC cap from host_config.h" >&2; exit 1; fi && \\
    chosen="" && \\
    for n in $(seq "$cap" -1 9); do \\
        if yum -y install "gcc-toolset-$n-gcc" "gcc-toolset-$n-gcc-c++" "gcc-toolset-$n-binutils"; then chosen="$n"; break; fi; \\
    done && \\
    if [ -z "$chosen" ]; then echo "no gcc-toolset at or below GCC $cap is available" >&2; exit 1; fi && \\
    ln -s "/opt/rh/gcc-toolset-$chosen" /opt/rh/gcc-toolset-active && \\
    yum -y clean all && rm -rf /var/cache/yum

ENV PATH=/opt/rh/gcc-toolset-active/root/usr/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rh/gcc-toolset-active/root/usr/lib64:/opt/rh/gcc-toolset-active/root/usr/lib:$LD_LIBRARY_PATH

RUN printf '__global__ void smoke_kernel() {{}}\\n' > /tmp/smoke.cu && \\
    nvcc -ccbin "$(command -v gcc)" -c -o /tmp/smoke.o /tmp/smoke.cu && \\
    rm -f /tmp/smoke.o /tmp/smoke.cu
"""


def build_image(cuda_variant: str, manylinux_image: str = DEFAULT_MANYLINUX_IMAGE) -> None:
    """
    Build the image for one cuda variant. Raises on failure.
    """
    tag = image_tag(cuda_variant)
    text = dockerfile_text(cuda_variant, manylinux_image)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / 'Dockerfile').write_text(text, encoding='utf-8')
        run([
            'docker', 'build',
            '-t', tag,
            '--build-arg', f'BASE_IMAGE={manylinux_image}',
            str(tmp_path),
        ])


def delete_image(tag: str) -> bool:
    """
    Delete one image. Returns True on success.
    """
    result = subprocess.run(['docker', 'rmi', tag], capture_output=True, text=True)
    if result.returncode == 0:
        print(f'Deleted image {tag}', flush=True)
        return True
    print(f'Failed to delete {tag}: {result.stderr.strip()}', flush=True)
    return False


def ensure_images_parallel(
    cuda_variants: list[str],
    max_parallel: int,
    manylinux_image: str = DEFAULT_MANYLINUX_IMAGE,
) -> tuple[list[str], list[str]]:
    """
    Build any missing images for the requested cuda variants concurrently.

    Parameters
    ----------
    cuda_variants : list[str]
        Variants that the upcoming wheel build phase needs.
    max_parallel : int
        Cap on concurrent `docker build` invocations.
    manylinux_image : str
        Base image for the Dockerfile.

    Returns
    -------
    tuple of (list[str], list[str])
        (built_or_already_present, failed_cuda_variants).
    """
    present = {info.cuda_variant for info in list_resident()}
    missing = [cv for cv in cuda_variants if cv not in present]

    if len(missing) == 0:
        print(f'All {len(cuda_variants)} images already resident.', flush=True)
        return list(cuda_variants), []

    print(f'Building {len(missing)} image(s) with up to {max_parallel} in parallel: {missing}',
          flush=True)

    success: list[str] = list(present & set(cuda_variants))
    failed: list[str] = []
    print_lock = threading.Lock()

    def worker(cv: str) -> tuple[str, bool, str]:
        try:
            build_image(cv, manylinux_image)
            with print_lock:
                print(f'Built {image_tag(cv)}', flush=True)
            return (cv, True, '')
        except subprocess.CalledProcessError as exc:
            return (cv, False, str(exc))

    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = [pool.submit(worker, cv) for cv in missing]
        for fut in as_completed(futures):
            cv, ok, err = fut.result()
            if ok:
                success.append(cv)
            else:
                failed.append(cv)
                print(f'Image build failed for {cv}: {err}', flush=True)

    return success, failed


def evict_lru(max_resident: int, keep: set[str]) -> list[str]:
    """
    Trim the image cache to at most max_resident images.

    Images whose cuda variant is in `keep` are never evicted. Among evictable images, the oldest
    by `created` is deleted first.

    Parameters
    ----------
    max_resident : int
        Maximum total images of namespace IMAGE_NAMESPACE to keep on disk.
    keep : set[str]
        Cuda variants that must remain regardless of age.

    Returns
    -------
    list[str]
        Tags of deleted images.
    """
    resident = list_resident()
    if len(resident) <= max_resident:
        return []

    keepers = [info for info in resident if info.cuda_variant in keep]
    evictable = [info for info in resident if info.cuda_variant not in keep]
    evictable.sort(key=lambda i: i.created)  # oldest first

    to_delete = max(0, len(resident) - max_resident)
    if to_delete > len(evictable):
        print(
            f'Cannot evict to {max_resident}: only {len(evictable)} images outside the keep set.',
            flush=True,
        )
        to_delete = len(evictable)

    deleted = []
    for info in evictable[:to_delete]:
        if delete_image(info.tag):
            deleted.append(info.tag)
    _ = keepers
    return deleted


def main() -> int:
    """
    Inspect the image pool from the command line.
    """
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list', help='List resident images.')

    p_ensure = sub.add_parser('ensure', help='Build the requested images in parallel.')
    p_ensure.add_argument('--max-parallel', type=int, default=2)
    p_ensure.add_argument('cuda', nargs='+')

    p_evict = sub.add_parser('evict', help='Trim to at most --max-resident images.')
    p_evict.add_argument('--max-resident', type=int, required=True)
    p_evict.add_argument('--keep', nargs='*', default=[])

    args = parser.parse_args()

    if args.cmd == 'list':
        for info in list_resident():
            print(f'{info.tag}\t{info.image_id}\t{info.size_bytes // 1024 // 1024} MB\t{info.created.isoformat()}')
        return 0

    if args.cmd == 'ensure':
        success, failed = ensure_images_parallel(args.cuda, max_parallel=args.max_parallel)
        print(f'success: {success}')
        print(f'failed: {failed}')
        return 0 if len(failed) == 0 else 1

    if args.cmd == 'evict':
        deleted = evict_lru(args.max_resident, keep=set(args.keep))
        print(f'deleted: {deleted}')
        return 0

    return 2


if __name__ == '__main__':
    sys.exit(main())

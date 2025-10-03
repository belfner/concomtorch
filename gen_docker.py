#!/usr/bin/env python3
# build_linux_wheels.py
# Python 3.10
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

CUDA_MATRIX = {
    # CUDA variant  -> (CUDA_MAJOR_MINOR, CUDA_PKG_SUFFIX)
    # Suffix is used in yum package name "cuda-toolkit-12-1"
    "cu121": ("12.1", "12-1"),
    "cu124": ("12.4", "12-4"),
    "cu126": ("12.6", "12-6"),
    "cu127": ("12.7", "12-7"),
    "cu128": ("12.8", "12-8"),
    "cu129": ("12.9", "12-9"),
    "cu130": ("13.0", "13-0"),
}

DEFAULT_MANYLINUX_IMAGE = "quay.io/pypa/manylinux_2_28_x86_64"


def normalize_abi(tag: str) -> str:
    """
    Accept 'cp310', '310', 'cp39', '39' and return 'cp310' / 'cp39'.
    """
    t = tag.lower().strip()
    if t.startswith("cp"):
        return t
    # allow '39' or '310'
    if t.isdigit():
        return f"cp{t}"
    raise ValueError(f"Unrecognized Python ABI: {tag}")


def normalize_abis(tags: list[str]) -> list[str]:
    return [normalize_abi(t) for t in tags]


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print(">>", " ".join(shlex.quote(c) for c in cmd))
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None, env=env)


def make_dockerfile_text(
        manylinux_image: str,
        cuda_major_minor: str,
        cuda_pkg_suffix: str,
) -> str:
    """
    Produce a Dockerfile that:
      - Starts from manylinux_2_28 (AlmaLinux 8 / EL8)
      - Installs CUDA Toolkit from NVIDIA's EL8 repo (nvcc, headers)
      - Leaves Python tool install to runtime (cibuildwheel is installed at container run)
    """
    # We install minimal utilities and the CUDA toolkit
    # Notes:
    #  - manylinux_2_28 has yum/dnf; NVIDIA provides a repo for RHEL8 (works on AlmaLinux 8)
    #  - This gets us /usr/local/cuda-{MAJOR.MINOR} with nvcc
    return f"""\
# syntax=docker/dockerfile:1
ARG BASE_IMAGE={manylinux_image}
FROM ${{BASE_IMAGE}}

# Install basics + add NVIDIA CUDA repository for EL8
RUN yum -y install dnf-plugins-core curl git && \\
    yum -y config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \\
    yum -y clean all && rm -rf /var/cache/yum

# Install CUDA toolkit matching the chosen CUDA variant (for nvcc + headers)
# E.g. cuda-toolkit-12-1, cuda-toolkit-12-4, etc.
RUN yum -y install cuda-toolkit-{cuda_pkg_suffix} && \\
    yum -y clean all && rm -rf /var/cache/yum

# Expose CUDA_HOME and PATH (nvcc)
ENV CUDA_HOME=/usr/local/cuda-{cuda_major_minor}
ENV PATH=$CUDA_HOME/bin:$PATH
"""
    # cibuildwheel and torch are installed at runtime to allow matrix flexibility


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build manylinux wheels inside a CUDA-enabled manylinux container."
    )
    p.add_argument(
        "--torch",
        dest="torch_version",
        required=True,
        help="PyTorch version spec, e.g. 2.4.1 or '2.4.*'",
    )
    p.add_argument(
        "--cuda",
        dest="cuda_variant",
        required=True,
        choices=sorted(CUDA_MATRIX.keys()),
        help="CUDA variant (matches PyTorch wheel channel), e.g. cu121, cu124, cu126",
    )
    p.add_argument(
        "--py",
        dest="py_abis",
        nargs="+",
        required=True,
        help="One or more CPython ABI tags to build, e.g. cp39 cp310 cp311 cp312",
    )
    p.add_argument(
        "--image-tag",
        dest="image_tag",
        default=None,
        help="Tag to assign to the built Docker image (default auto-generated).",
    )
    p.add_argument(
        "--manylinux-image",
        dest="manylinux_image",
        default=DEFAULT_MANYLINUX_IMAGE,
        help=f"Base image (default: {DEFAULT_MANYLINUX_IMAGE})",
    )
    p.add_argument(
        "--project-dir",
        dest="project_dir",
        default=".",
        help="Path to your project directory (defaults to current directory).",
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default="wheelhouse",
        help="Directory on the host to receive built wheels (default: wheelhouse).",
    )
    return p.parse_args()


def compute_cibw_build_pattern(py_abis: list[str]) -> str:
    abis = normalize_abis(py_abis)
    parts = [f"{abi}-manylinux_x86_64" for abi in abis]
    return " ".join(parts)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    if args.cuda_variant not in CUDA_MATRIX:
        print(f"Unsupported CUDA variant: {args.cuda_variant}", file=sys.stderr)
        sys.exit(2)
    cuda_major_minor, cuda_pkg_suffix = CUDA_MATRIX[args.cuda_variant]

    project_dir = Path(args.project_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dirs(output_dir)

    image_tag = (
            args.image_tag
            or f"concomtorch-manylinux-{args.cuda_variant}:torch-{args.torch_version}"
    )

    dockerfile_text = make_dockerfile_text(
        manylinux_image=args.manylinux_image,
        cuda_major_minor=cuda_major_minor,
        cuda_pkg_suffix=cuda_pkg_suffix,
    )

    # Write Dockerfile to a temp directory for the build context
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        dockerfile_path = tmpdir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_text, encoding="utf-8")

        # Build the image
        run(
            [
                "docker",
                "build",
                "-t",
                image_tag,
                "--build-arg",
                f"BASE_IMAGE={args.manylinux_image}",
                str(tmpdir),
            ]
        )

    # Normalize ABIs once
    norm_abis = normalize_abis(args.py_abis)

    # After building `image_tag`
    cibw_env = os.environ.copy()
    cibw_env.update({
        "CIBW_MANYLINUX_X86_64_IMAGE": image_tag,
        "CIBW_BUILD": compute_cibw_build_pattern(norm_abis),
        "CIBW_BUILD_FRONTEND": "pip; args: --no-build-isolation",
        "CIBW_SKIP": "*musllinux*",
        "CIBW_BEFORE_BUILD": (
            f'rm -rf .venv venv build dist *.egg-info wheelhouse || true && '
            f'python -m pip install --upgrade pip && '
            f'python -m pip install "torch=={args.torch_version}+{args.cuda_variant}" '
            f'--index-url https://download.pytorch.org/whl/{args.cuda_variant}/ && '
            'python -m pip install ninja numpy'
        ),
        "CIBW_REPAIR_WHEEL_COMMAND_LINUX": (
            'auditwheel repair -w {dest_dir} {wheel} '
            '--exclude libtorch.so '
            '--exclude libtorch_cpu.so '
            '--exclude libtorch_cuda.so '
            '--exclude libtorch_python.so '
            '--exclude libc10.so '
            '--exclude libc10_cuda.so '
            '--exclude libtorch_global_deps.so '
            '--exclude libcaffe2_nvrtc.so '
            '--exclude libtorch_cuda_linalg.so '
            '--exclude libshm.so'
        ),
        "CIBW_ENVIRONMENT": (
            f'CUDA_HOME=/usr/local/cuda-{cuda_major_minor} '
            f'LD_LIBRARY_PATH=/usr/local/cuda-{cuda_major_minor}/lib64:$LD_LIBRARY_PATH '
            f'AUDITWHEEL_PLAT="manylinux_2_28_x86_64"'
        ),
        "CIBW_BUILD_VERBOSITY": '2',
        # Use Docker-managed volume for pip cache (persists across builds, no permission issues)
        "CIBW_CONTAINER_ENGINE": "docker; create_args: -v concomtorch-pip-cache:/root/.cache/pip",
    })

    run(
        [sys.executable, "-m", "cibuildwheel", "--platform", "linux", "--output-dir", str(output_dir),
         str(project_dir)],
        env=cibw_env,
    )

    print("\n✅ Wheels written to:", output_dir)


if __name__ == "__main__":
    main()

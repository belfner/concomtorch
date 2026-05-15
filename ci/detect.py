#!/usr/bin/env python3
"""
Enumerate (torch_version, cuda_variant, py_abi) combinations that PyTorch publishes and the
local matrix.yaml allows.

Uses the torch-wheel-index package to query download.pytorch.org. The output is the WANTED set,
not the build plan: ci/plan.py subtracts what already exists in the wheelhouse.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import Path

import yaml
from packaging.version import Version


@dataclass(frozen=True)
class Combo:
    torch: str
    cuda: str
    py: str

    def key(self) -> tuple[str, str, str]:
        return (self.torch, self.cuda, self.py)


def cu_tag_to_dotted(cu_tag: str) -> str:
    """
    Translate a PyTorch cuXYZ tag to the dotted form torch-wheel-index uses.

    Parameters
    ----------
    cu_tag : str
        e.g. 'cu121'.

    Returns
    -------
    str
        e.g. '12.1'.
    """
    if not cu_tag.startswith('cu'):
        raise ValueError(f'Expected cu### tag, got {cu_tag!r}')
    digits = cu_tag[2:]
    return f'{digits[:-1]}.{digits[-1]}'


def dotted_to_cu_tag(dotted: str) -> str:
    """
    Translate '12.1' to 'cu121'.
    """
    return 'cu' + dotted.replace('.', '')


def cp_to_dotted(cp: str) -> str:
    """
    Translate 'cp310' to '3.10'.
    """
    if not cp.startswith('cp'):
        raise ValueError(f'Expected cp### tag, got {cp!r}')
    digits = cp[2:]
    return f'{digits[:1]}.{digits[1:]}'


def dotted_to_cp(dotted: str) -> str:
    """
    Translate '3.10' to 'cp310'.
    """
    return 'cp' + dotted.replace('.', '')


def load_matrix(path: Path) -> dict:
    """
    Load and validate the build matrix YAML.

    Parameters
    ----------
    path : Path
        Path to matrix.yaml.

    Returns
    -------
    dict
        Parsed matrix configuration.
    """
    with path.open() as fh:
        data = yaml.safe_load(fh)
    required = {'torch_min', 'python_abis', 'cuda_variants'}
    missing = required - set(data.keys())
    if len(missing) > 0:
        raise ValueError(f'matrix.yaml missing required keys: {sorted(missing)}')
    data.setdefault('exclude', [])
    return data


def fetch_catalog() -> list[dict]:
    """
    Run `torch-wheel-index find --compute-type cuda --platform linux` and parse the catalog.

    Returns
    -------
    list[dict]
        Catalog rows.
    """
    result = subprocess.run(
        ['torch-wheel-index', 'find', '--format', 'json',
         '--compute-type', 'cuda', '--platform', 'linux'],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def is_excluded(torch_ver: str, cuda: str, exclude_rules: list[dict]) -> bool:
    """
    Check whether a (torch, cuda) pair is excluded by any rule.

    Each rule may contain `cuda` (string equality required), `torch_min` (skip when
    torch >= value) and/or `torch_max` (skip when torch <= value).

    Parameters
    ----------
    torch_ver : str
        Torch version, e.g. '2.4.1'.
    cuda : str
        CUDA variant tag, e.g. 'cu121'.
    exclude_rules : list[dict]
        Rules from matrix.yaml.

    Returns
    -------
    bool
        True if this pair should be skipped.
    """
    tv = Version(torch_ver)
    for rule in exclude_rules:
        if rule.get('cuda') != cuda:
            continue
        if 'torch_min' in rule and tv < Version(rule['torch_min']):
            continue
        if 'torch_max' in rule and tv > Version(rule['torch_max']):
            continue
        return True
    return False


def enumerate_wanted(matrix: dict, catalog: list[dict]) -> list[Combo]:
    """
    Intersect the catalog with matrix.yaml constraints.

    Parameters
    ----------
    matrix : dict
        Parsed matrix.yaml.
    catalog : list[dict]
        Output of `torch-wheel-index find`.

    Returns
    -------
    list[Combo]
        Sorted list of valid combinations.
    """
    wanted_cuda = set(matrix['cuda_variants'])
    wanted_py = set(matrix['python_abis'])
    torch_min = Version(matrix['torch_min'])

    seen: set[tuple[str, str, str]] = set()
    out: list[Combo] = []
    for row in catalog:
        torch_ver = row['version']
        cuda_dotted = row['compute_version']
        py_dotted = row['python_version']

        cuda = dotted_to_cu_tag(cuda_dotted)
        py = dotted_to_cp(py_dotted)

        if cuda not in wanted_cuda or py not in wanted_py:
            continue
        if Version(torch_ver) < torch_min:
            continue
        if is_excluded(torch_ver, cuda, matrix['exclude']):
            continue
        combo = Combo(torch=torch_ver, cuda=cuda, py=py)
        if combo.key() in seen:
            continue
        seen.add(combo.key())
        out.append(combo)

    out.sort(key=lambda c: (Version(c.torch), c.cuda, c.py))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path,
                        default=Path(__file__).parent / 'matrix.yaml')
    parser.add_argument('--format', choices=['json', 'text'], default='json')
    args = parser.parse_args()

    matrix = load_matrix(args.matrix)
    catalog = fetch_catalog()
    wanted = enumerate_wanted(matrix, catalog)

    if args.format == 'json':
        json.dump([asdict(c) for c in wanted], sys.stdout, indent=2)
        sys.stdout.write('\n')
    else:
        for c in wanted:
            print(f'{c.torch}\t{c.cuda}\t{c.py}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

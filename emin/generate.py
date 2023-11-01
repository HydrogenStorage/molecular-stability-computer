"""Generate potential molecules using Surge"""
from tempfile import TemporaryDirectory
from subprocess import Popen, PIPE
from typing import Iterable
from pathlib import Path
import heapq

import numpy as np

surge_path = Path(__file__).parent / '../bin/surge'


def generate_molecules_with_surge(formula: str, to_avoid: Iterable[int] = frozenset(range(1, 10))) -> Iterable[str]:
    """Generate molecules with the same formula using Surge

    Args:
        formula: Target formula
        to_avoid: Substructures to avoid (See ``-B`` tag of Surge).
            Avoid all by default
    Yields:
        SMILES strings
    """

    with TemporaryDirectory(prefix='surge') as tmp:
        error_file = Path(tmp) / 'stderr'
        command = [str(surge_path), '-S', '-B' + ','.join(map(str, to_avoid)), formula]
        with open(error_file, 'w') as fe, Popen(command, stderr=fe, stdout=PIPE) as proc:
            for line in proc.stdout:
                yield line.decode().strip()
        if proc.returncode != 0:
            raise ValueError(f'Command: {" ".join(command)}\nSTDERR: {error_file.read_text()}')


def get_random_selection_with_surge(formula: str, to_select: int | float, seed: int = 1, **kwargs) -> tuple[list[str], int]:
    """Get only the top fraction of molecules generated with surge

    Keyword arguments are passed to :meth:`generate_molecules_with_surge`

    Args:
        formula: Molecular formula to generate
        to_select: Maximum number or fraction to return
        seed: Random number seed
    Returns:
        - List of top molecules
        - Total generated
    """

    get_fraction = to_select < 1

    # Make the generator for smiles and random numbers
    mol_gen = generate_molecules_with_surge(formula, **kwargs)
    rng = np.random.default_rng(seed)

    # Start the list
    output = []
    count = 0
    for smiles in mol_gen:
        count += 1
        score = rng.random()
        if len(output) < (count * to_select if get_fraction else to_select):
            heapq.heappush(output, (score, smiles))
        else:
            heapq.heappushpop(output, (score, smiles))
    return output, count

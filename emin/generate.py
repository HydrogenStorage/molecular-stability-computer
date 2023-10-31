"""Generate potential molecules using Surge"""
from tempfile import TemporaryDirectory
from subprocess import Popen, PIPE
from typing import Iterable
from pathlib import Path

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
                yield line.strip()
        if proc.returncode != 0:
            raise ValueError(f'Command: {" ".join(command)}\nSTDERR: {error_file.read_text()}')

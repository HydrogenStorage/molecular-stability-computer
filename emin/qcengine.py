"""Compute the stability of a molecule using QCEngine"""
from qcelemental.models import OptimizationResult, OptimizationInput, Molecule, AtomicResult, AtomicInput
from qcelemental.models.procedures import QCInputSpecification
from rdkit.Chem import AllChem
from rdkit import Chem
import qcengine as qcng


def get_qcengine_spec(level_name: str) -> tuple[str, QCInputSpecification]:
    """Get a specification for a certain accuracy level

    Args:
        level_name: Name of the accuracy level
    Returns:
        - Name of the code to use
        - Input specification, if applicable
    """

    level_name = level_name.lower()
    if level_name == 'xtb':
        return 'xtb', QCInputSpecification(
            driver='gradient',
            model={'method': 'GFN2-xTB'},
            keywords={"accuracy": 0.05}
        )
    else:
        raise ValueError(f'No such specification level: {level_name}')


# Adapted from ExaMol: https://github.com/exalearn/ExaMol/blob/main/examol/simulate/initialize.py
def generate_xyz(smiles: str) -> str:
    """Generate the XYZ coordinates for a molecule

    Args:
        smiles: SMILES of molecule
    Returns:
        XYZ coordinates for the molecule
    """

    # Generate 3D coordinates for the molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)

    # Save geometry as 3D coordinates
    xyz = f"{mol.GetNumAtoms()}\n"
    xyz += smiles + "\n"
    conf = mol.GetConformer()
    for i, a in enumerate(mol.GetAtoms()):
        s = a.GetSymbol()
        c = conf.GetAtomPosition(i)
        xyz += f"{s} {c[0]} {c[1]} {c[2]}\n"

    return xyz


def evaluate_mmff94(smiles: str, relax: bool) -> float:
    """Generate a structure then compute the energy with the MMFF94 forcefield

    Args:
        smiles: SMILES string of the molecule
        relax: Whether to run relaxation
    Returns:
        Energy of the molecule
    """
    # Generate 3D coordinates for the molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=1)

    # Make the forcefield
    ffprops = AllChem.MMFFGetMoleculeProperties(mol)
    ff = AllChem.MMFFGetMoleculeForceField(mol, ffprops)

    if relax:
        ff.Minimize()
    return ff.CalcEnergy() / 627.509


def relax_molecule(xyz: str, code: str, spec: QCInputSpecification) -> OptimizationResult:
    """Perform a structural optimization of a molecule

    Args:
        xyz: Initial geometry of molecule to be evaluated
        code: Software to use for energy calculation
        spec: Specification for the runtime
    Returns:
        Optimization result
    """

    molecule = Molecule.from_data(xyz, 'xyz')
    opt_input = OptimizationInput(
        input_specification=spec,
        initial_molecule=molecule,
        keywords={"program": code}
    )
    return qcng.compute_procedure(
        opt_input,
        'geometric'
    )


def compute_energy(xyz: str, code: str, spec: QCInputSpecification) -> AtomicResult:
    """Perform a single-point energy evaluation

    Args:
        xyz: Initial geometry of molecule to be evaluated
        code: Software to use for energy calculation
        spec: Specification for the runtime
    Returns:
        Energy result
    """
    molecule = Molecule.from_data(xyz, 'xyz')
    eng_input = AtomicInput(
        molecule=molecule,
        driver='energy',
        model=spec.model,
        keywords=spec.keywords,
    )
    return qcng.compute(eng_input, program=code)

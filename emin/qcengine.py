"""Compute the stability of a molecule using QCEngine"""
from qcelemental.models import OptimizationResult, OptimizationInput, Molecule
from qcelemental.models.procedures import QCInputSpecification
from rdkit.Chem import AllChem
from rdkit import Chem
import qcengine as qcng


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

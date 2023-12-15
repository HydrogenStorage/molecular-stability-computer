"""Source molecules from PubChem"""
from functools import lru_cache

from rdkit import Chem
import requests


def get_molecules_from_pubchem(formula: str,
                               neutral_only: bool = True,
                               ignore_isotopes: bool = True) -> list[str]:
    """Get molecules from PubChem that share the same formula

    Args:
        formula: Chemical formula
        neutral_only: Only return neutral molecules
        ignore_isotopes: Skip molecules where isotopes are specified
    Returns:
        List of SMILES strings of molecules known in PubChem
    """

    # Download the file all in one chunk
    rep = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{formula}/property/InChI/TXT')

    # Process the molecules
    output = set()
    molecules = rep.content.decode().split()
    for inchi in molecules:
        if neutral_only and '/q' in inchi:
            continue
        if ignore_isotopes and '/i' in inchi:
            continue
        output.add(inchi)

    # Convert to SMILES and return
    output_smiles = []
    for inchi in output:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)
            if '.' not in smiles:  # Skip molecules broken into two parts
                output_smiles.append(smiles)
    return output_smiles


@lru_cache
def get_inchi_keys_from_pubchem(formula: str, retries: int = 5) -> set[str]:
    """Get the InChI keys of all molecules in PubChem with a certain formula

    Args:
        formula: Formula in question
        retries: How many retries to attempt
    Returns:
        List of unique InChI keys
    """
    for _ in range(retries):
        try:
            res = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{formula}/property/InChIKey/TXT')
            assert res.status_code == 200, f'PubChem failed with error code: {res.status_code}'
            return set(res.content.decode().split())
        except (ConnectionError, requests.ReadTimeout):
            continue
    else:
        raise ValueError('Connection to PubChem failed')

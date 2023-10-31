"""Source molecules from PubChem"""
import requests


def get_molecules_from_pubchem(formula: str, neutral_only: bool = True, ignore_isotopes: bool = True) -> list[str]:
    """Get molecules from PubChem that share the same formula

    Args:
        formula: Chemical formula
        neutral_only: Only return neutral molecules
        ignore_isotopes: Skip molecules where isotopes are specified
    Returns:
        List of InChI strings of molecules known in PubChem
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
    return list(output)

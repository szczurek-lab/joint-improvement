"""Validity check utilities for molecular properties."""

try:
    from rdkit import Chem, RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    Chem = None
    RDLogger = None
    _RDKIT_AVAILABLE = False


def calculate_validity(smiles: str) -> bool:
    """
    Check if a SMILES string corresponds to a valid molecule.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    bool
        True if the SMILES string corresponds to a valid, non-empty molecule.

    Raises
    ------
    ImportError
        If RDKit is not installed.

    Notes
    -----
    This function uses RDKit's MolFromSmiles method to parse the SMILES string.
    It checks if the molecule is valid (non-empty) and has at least one atom.

    RDKit is an optional dependency. Install it with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for validity check. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )
    try:
        mol = Chem.MolFromSmiles(smiles)
        return smiles != "" and mol is not None and mol.GetNumAtoms() > 0
    except:  # noqa: E722
        return False

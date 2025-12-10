"""Docking score calculation utilities for molecular properties.

This module provides a simple interface for calculating docking scores using
QuickVina2. The implementation follows the Saturn oracle component pattern
for consistency with the broader codebase.
"""

from __future__ import annotations

try:
    import numpy as np
    from rdkit import Chem, RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    Chem = None
    RDLogger = None
    np = None

try:
    from joint_improvement.utils.chemistry.docking_utils.quickvina2 import DockingVina

    _DOCKING_AVAILABLE = True
except ImportError as e:
    _DOCKING_AVAILABLE = False
    DockingVina = None
    _DOCKING_IMPORT_ERROR = str(e)


# Singleton instance for docking oracle
_docking_instance: DockingVina | None = None
_docking_target: str | None = None


def calculate_docking(smiles: str, target: str = "fa7") -> float:
    """Calculate the docking score of a molecule using QuickVina2.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    target : str, optional
        Target protein for docking. Supported targets: 'fa7', 'parp1', '5ht1b', 'jak2', 'braf'.
        Default is 'fa7'.

    Returns
    -------
    float
        Docking score from QuickVina2. Returns NaN if the molecule cannot be
        parsed or if an error occurs during docking calculation.

    Raises
    ------
    ImportError
        If RDKit or docking utilities are not installed.

    Notes
    -----
    This function uses QuickVina2 to perform molecular docking and calculate
    binding scores. The docking setup requires:
    - QuickVina2 binary (qvina02) to be available
    - Receptor PDBQT files for the specified target
    - OpenBabel for 3D structure generation

    Docking scores are typically negative values where lower (more negative)
    values indicate better binding affinity. The score represents the predicted
    binding free energy in kcal/mol.

    The function uses a singleton pattern to avoid reinitializing the docking
    setup on every call. If the target changes, a new instance will be created.

    Supported targets and their box configurations:
    - fa7: (10.131, 41.879, 32.097) center, (20.673, 20.198, 21.362) size
    - parp1: (26.413, 11.282, 27.238) center, (18.521, 17.479, 19.995) size
    - 5ht1b: (-26.602, 5.277, 17.898) center, (22.5, 22.5, 22.5) size
    - jak2: (114.758, 65.496, 11.345) center, (19.033, 17.929, 20.283) size
    - braf: (84.194, 6.949, -7.081) center, (22.032, 19.211, 14.106) size
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for docking calculation. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    if not _DOCKING_AVAILABLE:
        error_msg = "Docking utilities are required for docking calculation. Ensure docking_utils module is available."
        if _DOCKING_IMPORT_ERROR:
            error_msg += f"\nImport error: {_DOCKING_IMPORT_ERROR}"
        raise ImportError(error_msg)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")

        # Use singleton pattern, but recreate if target changes
        global _docking_instance, _docking_target
        if _docking_instance is None or _docking_target != target:
            _docking_instance = DockingVina(target)
            _docking_target = target

        # DockingVina.predict expects a list of SMILES
        scores = _docking_instance.predict([smiles])
        if len(scores) == 0:
            return float("nan")

        return float(scores[0])

    except Exception:
        return float("nan")


def calculate_docking_batch(smiles_list: list[str], target: str) -> list[float]:
    """Calculate docking scores for a batch of molecules using QuickVina2.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES string representations of molecules.
    target : str, optional
        Target protein for docking. Supported targets: 'fa7', 'parp1', '5ht1b', 'jak2', 'braf'.
        Default is 'fa7'.

    Returns
    -------
    list[float]
        List of docking scores from QuickVina2. Returns NaN for molecules that
        cannot be parsed or if an error occurs during docking calculation.

    Raises
    ------
    ImportError
        If RDKit or docking utilities are not installed.

    Notes
    -----
    This function is more efficient than calling calculate_docking multiple times
    as it processes all molecules in a single batch. See calculate_docking for
    more details on docking scores and requirements.
    """
    if not _RDKIT_AVAILABLE:
        raise ImportError(
            "RDKit is required for docking calculation. "
            "Install it with: pip install rdkit "
            "or conda install -c conda-forge rdkit"
        )

    if not _DOCKING_AVAILABLE:
        error_msg = "Docking utilities are required for docking calculation. Ensure docking_utils module is available."
        if _DOCKING_IMPORT_ERROR:
            error_msg += f"\nImport error: {_DOCKING_IMPORT_ERROR}"
        raise ImportError(error_msg)

    try:
        # Use singleton pattern, but recreate if target changes
        global _docking_instance, _docking_target
        if _docking_instance is None or _docking_target != target:
            _docking_instance = DockingVina(target)
            _docking_target = target

        # DockingVina.predict expects a list of SMILES
        scores = _docking_instance.predict(smiles_list)
        return [float(score) if score is not None else float("nan") for score in scores]

    except Exception:
        return [float("nan")] * len(smiles_list)

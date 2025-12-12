"""Docking score calculation utilities for molecular properties."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    from rdkit import Chem, RDLogger

    # Suppress RDKit warnings
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    _RDKIT_AVAILABLE = True
except ImportError:
    _RDKIT_AVAILABLE = False
    Chem = None
    RDLogger = None

try:
    from joint_improvement.utils.chemistry.docking_utils.quickvina2 import DockingVina
    from joint_improvement.utils.chemistry.docking_utils.quickvina2_gpu import (
        DockingVinaGPU,
    )

    _DOCKING_AVAILABLE = True
    _DOCKING_IMPORT_ERROR: str | None = None
except ImportError as e:
    _DOCKING_AVAILABLE = False
    DockingVina = None
    DockingVinaGPU = None
    _DOCKING_IMPORT_ERROR = str(e)


# Docking score thresholds for each target (lower scores indicate better binding)
TARGET_DOCKING_THRESHOLDS: dict[str, float] = {
    "braf": -10.3,
    "parp1": -10.0,
    "fa7": -8.5,
    "jak2": -9.1,
    "5ht1b": -8.7845,
}

# Singleton instances for docking oracle (one per device)
_docking_instances: dict[str, DockingVina] = {}
_docking_targets: dict[str, str] = {}


def calculate_docking(smiles: str, target: str = "fa7", device: str = "cpu") -> float:
    """Calculate the docking score of a molecule.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    target : str, optional
        Target protein for docking. Supported targets: 'fa7', 'parp1', '5ht1b', 'jak2', 'braf'.
        Default is 'fa7'.
    device : str, optional
        Device to use for docking. Options: 'cpu' (default) or 'gpu'.
        Note: GPU support requires QuickVina2-GPU with additional setup.

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
    RDKit and docking utilities are optional dependencies. Install RDKit with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``.
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

    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return float("nan")

        global _docking_instances, _docking_targets
        instance_key = f"{device}_{target}"
        if instance_key not in _docking_instances:
            if device == "gpu":
                if DockingVinaGPU is None:
                    raise ImportError("DockingVinaGPU is unavailable.")
                _docking_instances[instance_key] = DockingVinaGPU(target)
            else:
                if DockingVina is None:
                    raise ImportError("DockingVina is unavailable.")
                _docking_instances[instance_key] = DockingVina(target)
            _docking_targets[instance_key] = target

        scores = _docking_instances[instance_key].predict([smiles])
        if scores is None or len(scores) == 0:
            return float("nan")

        return float(scores[0])
    except Exception:
        return float("nan")


def calculate_docking_batch(
    smiles_list: list[str] | NDArray[np.str_], target: str, device: str = "cpu"
) -> NDArray[np.float64]:
    """Calculate docking scores for a batch of molecules.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES string representations of molecules.
    target : str
        Target protein for docking. Supported targets: 'fa7', 'parp1', '5ht1b', 'jak2', 'braf'.
    device : str, optional
        Device to use for docking. Options: 'cpu' (default) or 'gpu'.
        Note: GPU support requires QuickVina2-GPU with additional setup.

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
    RDKit and docking utilities are optional dependencies. Install RDKit with:
    ``pip install rdkit`` or ``conda install -c conda-forge rdkit``.
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

    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")

    vectorized_docking = np.vectorize(
        lambda smi: calculate_docking(smi, target=target, device=device),
        otypes=[float],
    )
    return vectorized_docking(smiles_list)

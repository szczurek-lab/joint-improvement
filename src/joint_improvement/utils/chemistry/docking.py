"""Docking score calculation utilities for molecular properties.

This module provides a MolStitch-compatible implementation for calculating docking
scores using QuickVina2 CPU. The workflow follows MolStitch's approach for
reproducibility.

Workflow Overview:
* Input canonical SMILES
* Generate 3D structure using OpenBabel --gen3D
* Convert to PDBQT using Pybel
* Dock using QuickVina2 CPU (qvina02)
* Parse docking score from stdout

References
----------
- MolStitch: https://github.com/MolecularTeam/MolStitch/blob/main/evaluators/dock/qvina2.py
- QuickVina2: https://github.com/QVina/qvina
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem, RDLogger

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

try:
    import pybel

    _PYBEL_AVAILABLE = True
except ImportError:
    _PYBEL_AVAILABLE = False
    pybel = None

try:
    from loguru import logger

    _LOGGER_AVAILABLE = True
except ImportError:
    _LOGGER_AVAILABLE = False
    logger = None


class DockingScore:
    """QuickVina2 CPU-based docking score calculator (MolStitch-compatible).

    This class provides a MolStitch-compatible interface for calculating docking
    scores using QuickVina2 CPU. The implementation follows MolStitch's approach
    for reproducibility.

    Attributes
    ----------
    _instance : Optional[DockingScore]
        Singleton instance of the DockingScore class.
    binary : str
        Path to QuickVina2 binary executable (qvina02).
    receptor : str
        Path to receptor PDBQT file.
    box_center : tuple[float, float, float]
        Center coordinates of the docking box.
    box_size : tuple[float, float, float]
        Size of the docking box in Angstroms.
    exhaustiveness : int
        Exhaustiveness parameter for docking search.
    num_modes : int
        Number of docking modes to generate.
    timeout_gen3d : int
        Timeout in seconds for 3D generation.
    timeout_dock : int
        Timeout in seconds for docking.
    """

    _instance: DockingScore | None = None
    binary: str | None = None
    receptor: str | None = None
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    exhaustiveness: int = 1
    num_modes: int = 10
    timeout_gen3d: int = 30
    timeout_dock: int = 100

    def __init__(
        self,
        binary: str | None = None,
        receptor: str | None = None,
        box_center: tuple[float, float, float] | None = None,
        box_size: tuple[float, float, float] | None = None,
        exhaustiveness: int = 1,
        num_modes: int = 10,
        timeout_gen3d: int = 30,
        timeout_dock: int = 100,
    ) -> None:
        """Initialize QuickVina2 CPU docking score calculator (MolStitch-compatible).

        Parameters
        ----------
        binary : str, optional
            Path to QuickVina2 binary executable (qvina02). If None, will attempt
            to load from QUICKVINA2_BINARY environment variable.
        receptor : str, optional
            Path to receptor PDBQT file. If None, will attempt to load from
            QUICKVINA2_RECEPTOR environment variable.
        box_center : tuple[float, float, float], optional
            Center coordinates of the docking box. If None, will attempt to load
            from QUICKVINA2_BOX_CENTER environment variable or use default.
        box_size : tuple[float, float, float], optional
            Size of the docking box in Angstroms. If None, uses default (20, 20, 20).
        exhaustiveness : int, default=1
            Exhaustiveness parameter for docking search.
        num_modes : int, default=10
            Number of docking modes to generate.
        timeout_gen3d : int, default=30
            Timeout in seconds for 3D generation.
        timeout_dock : int, default=100
            Timeout in seconds for docking.

        Raises
        ------
        ValueError
            If required parameters are not provided or invalid.
        FileNotFoundError
            If binary or receptor files don't exist.
        """
        # Determine binary path
        if binary is None:
            binary = os.environ.get("QUICKVINA2_BINARY")
        if binary is None:
            raise ValueError(
                "QuickVina2 binary path not provided. Set QUICKVINA2_BINARY "
                "environment variable or provide binary argument."
            )

        binary_path = Path(binary)
        if not binary_path.exists():
            raise FileNotFoundError(f"QuickVina2 binary not found: {binary_path}")

        # Ensure binary is executable
        subprocess.run(["chmod", "u+x", str(binary_path)], check=False)
        self.binary = str(binary_path)

        # Determine receptor path
        if receptor is None:
            receptor = os.environ.get("QUICKVINA2_RECEPTOR")
        if receptor is None:
            raise ValueError(
                "Receptor PDBQT file path not provided. Set QUICKVINA2_RECEPTOR "
                "environment variable or provide receptor argument."
            )

        receptor_path = Path(receptor)
        if not receptor_path.exists():
            raise FileNotFoundError(f"Receptor PDBQT file not found: {receptor_path}")
        if not receptor.endswith(".pdbqt"):
            raise ValueError("Receptor file must have .pdbqt extension")
        self.receptor = str(receptor_path)

        # Setup docking box
        if box_center is None:
            box_center_str = os.environ.get("QUICKVINA2_BOX_CENTER")
            if box_center_str:
                try:
                    box_center = tuple(float(x) for x in box_center_str.split(","))  # type: ignore[assignment]
                except ValueError:
                    raise ValueError("QUICKVINA2_BOX_CENTER must be comma-separated floats") from None

        if box_center is None:
            raise ValueError(
                "Box center not provided. Set QUICKVINA2_BOX_CENTER environment variable "
                "or provide box_center argument."
            )

        self.box_center = box_center
        self.box_size = box_size if box_size is not None else (20.0, 20.0, 20.0)
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.timeout_gen3d = timeout_gen3d
        self.timeout_dock = timeout_dock

        if _LOGGER_AVAILABLE:
            logger.info("QuickVina2 CPU docking score calculator initialized")
            logger.info(f"Binary: {self.binary}")
            logger.info(f"Receptor: {self.receptor}")
            logger.info(f"Box center: {self.box_center}, size: {self.box_size}")

    @classmethod
    def get_instance(
        cls,
        binary: str | None = None,
        receptor: str | None = None,
        box_center: tuple[float, float, float] | None = None,
        box_size: tuple[float, float, float] | None = None,
        exhaustiveness: int = 1,
        num_modes: int = 10,
        timeout_gen3d: int = 30,
        timeout_dock: int = 100,
    ) -> DockingScore:
        """Get or create singleton instance of DockingScore.

        Parameters
        ----------
        binary : str, optional
            Path to QuickVina2 binary executable.
        receptor : str, optional
            Path to receptor PDBQT file.
        box_center : tuple[float, float, float], optional
            Center coordinates of the docking box.
        box_size : tuple[float, float, float], optional
            Size of the docking box in Angstroms.
        exhaustiveness : int, default=1
            Exhaustiveness parameter.
        num_modes : int, default=10
            Number of docking modes.
        timeout_gen3d : int, default=30
            Timeout for 3D generation.
        timeout_dock : int, default=100
            Timeout for docking.

        Returns
        -------
        DockingScore
            Singleton instance of DockingScore class.
        """
        if cls._instance is None:
            cls._instance = cls(
                binary=binary,
                receptor=receptor,
                box_center=box_center,
                box_size=box_size,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                timeout_gen3d=timeout_gen3d,
                timeout_dock=timeout_dock,
            )
        return cls._instance

    def _gen_3d(self, smi: str, ligand_mol_file: str) -> None:
        """Generate initial 3D conformation from SMILES using OpenBabel (MolStitch approach).

        Parameters
        ----------
        smi : str
            SMILES string.
        ligand_mol_file : str
            Output MOL file path.
        """
        run_line = f"obabel -:{smi} --gen3D -O {ligand_mol_file}"
        subprocess.check_output(
            run_line.split(),
            stderr=subprocess.STDOUT,
            timeout=self.timeout_gen3d,
            universal_newlines=True,
        )

    def _docking(
        self,
        ligand_mol_file: str,
        ligand_pdbqt_file: str,
        docking_pdbqt_file: str,
    ) -> list[float]:
        """Run docking using QuickVina2 CPU (MolStitch approach).

        Parameters
        ----------
        ligand_mol_file : str
            Input MOL file path.
        ligand_pdbqt_file : str
            Output PDBQT file path for ligand.
        docking_pdbqt_file : str
            Output PDBQT file path for docking results.

        Returns
        -------
        list[float]
            List of affinity scores (docking scores).
        """
        # Convert MOL to PDBQT using Pybel
        if _PYBEL_AVAILABLE:
            ms = list(pybel.readfile("mol", ligand_mol_file))
            if not ms:
                return []
            m = ms[0]
            m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        else:
            # Fallback to OpenBabel CLI
            result = subprocess.run(
                ["obabel", ligand_mol_file, "-opdbqt", "-O", ligand_pdbqt_file],
                capture_output=True,
                check=False,
            )
            if result.returncode != 0:
                return []

        # Run QuickVina2 docking
        run_line = (
            f"{self.binary} --receptor {self.receptor} --ligand {ligand_pdbqt_file} "
            f"--out {docking_pdbqt_file} "
            f"--center_x {self.box_center[0]} --center_y {self.box_center[1]} --center_z {self.box_center[2]} "
            f"--size_x {self.box_size[0]} --size_y {self.box_size[1]} --size_z {self.box_size[2]} "
            f"--num_modes {self.num_modes} --exhaustiveness {self.exhaustiveness}"
        )

        output_bytes: bytes = subprocess.check_output(  # type: ignore[assignment]
            run_line.split(),
            stderr=subprocess.STDOUT,
            timeout=self.timeout_dock,
        )
        output_str: str = output_bytes.decode("utf-8")

        # Parse affinity list from stdout (MolStitch approach)
        result_lines = output_str.split("\n")
        check_result = False
        affinity_list = []

        for result_line in result_lines:
            if result_line.startswith("-----+"):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith("Writing output"):
                break
            if result_line.startswith("Refine time"):
                break
            lis = result_line.strip().split()
            if not lis or not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list.append(affinity)

        return affinity_list

    @classmethod
    def calculate(cls, mol: Chem.Mol) -> float:
        """Calculate docking score for a molecule using QuickVina2 CPU (MolStitch approach).

        This method uses a singleton instance to avoid reinitializing the
        docking setup on every call. The workflow follows MolStitch's protocol:
        1. Convert to canonical SMILES
        2. Generate 3D structure using OpenBabel --gen3D
        3. Convert to PDBQT using Pybel
        4. Run QuickVina2 CPU docking
        5. Parse docking score from stdout

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.

        Returns
        -------
        float
            Docking score from QuickVina2. Returns NaN if the molecule cannot
            be processed or if docking fails.

        Notes
        -----
        Docking scores are typically negative values where lower (more negative)
        values indicate better binding affinity. The score represents the predicted
        binding free energy in kcal/mol.
        """
        try:
            # Get or create singleton instance
            if cls._instance is None:
                cls._instance = cls.get_instance()

            # Convert to canonical SMILES (MolStitch approach)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            if not canonical_smiles:
                return float("nan")

            # Create temporary directory
            temp_dir = tempfile.mkdtemp()

            try:
                ligand_mol_file = os.path.join(temp_dir, "ligand.mol")
                ligand_pdbqt_file = os.path.join(temp_dir, "ligand.pdbqt")
                docking_pdbqt_file = os.path.join(temp_dir, "dock.pdbqt")

                # Generate 3D structure using OpenBabel (MolStitch approach)
                try:
                    cls._instance._gen_3d(canonical_smiles, ligand_mol_file)
                except Exception:
                    return float("nan")

                # Run docking
                try:
                    affinity_list = cls._instance._docking(
                        ligand_mol_file,
                        ligand_pdbqt_file,
                        docking_pdbqt_file,
                    )
                except Exception:
                    return float("nan")

                # Return first affinity (best score) or NaN if empty
                if len(affinity_list) == 0:
                    return float("nan")

                return float(affinity_list[0])

            finally:
                # Cleanup temporary directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            if _LOGGER_AVAILABLE and logger is not None:
                logger.error(f"Error calculating docking score: {e}")
            return float("nan")


def calculate_docking(smiles: str) -> float:
    """Calculate the docking score of a molecule using QuickVina2 CPU (MolStitch-compatible).

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    float
        Docking score from QuickVina2. Returns NaN if the molecule cannot
        be parsed or if an error occurs during docking calculation.

    Notes
    -----
    This function uses QuickVina2 CPU to perform molecular docking and calculate
    binding scores. The docking setup must be initialized by setting the following
    environment variables:
    - QUICKVINA2_BINARY: Path to QuickVina2 binary executable (qvina02)
    - QUICKVINA2_RECEPTOR: Path to receptor PDBQT file
    - QUICKVINA2_BOX_CENTER: Comma-separated box center coordinates (e.g., "10.131,41.879,32.097")

    Alternatively, call DockingScore.get_instance() with explicit paths.

    The workflow follows MolStitch's protocol:
    1. Convert SMILES to canonical form
    2. Generate 3D structure using OpenBabel --gen3D
    3. Convert to PDBQT using Pybel
    4. Run QuickVina2 CPU docking
    5. Parse docking score from stdout

    Docking scores are typically negative values where lower (more negative)
    values indicate better binding affinity. The score represents the predicted
    binding free energy in kcal/mol.

    Requires:
    - QuickVina2 binary (https://github.com/QVina/qvina)
    - OpenBabel (for 3D generation and PDBQT conversion)
    - Pybel (optional, for better PDBQT conversion)
    - RDKit (for molecular processing)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    return DockingScore.calculate(mol)

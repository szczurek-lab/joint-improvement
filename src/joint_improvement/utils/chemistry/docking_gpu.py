"""GPU docking score calculation utilities for molecular properties.

This module provides a QuickVina2-GPU-based implementation for calculating docking
scores. The workflow follows the protocol from https://arxiv.org/abs/2406.08506 Appendix C.

Workflow Overview:
* Input canonical SMILES
* Convert to RDKit Mol
* Protonate
* Generate 1 (lowest energy) conformer using RDKit ETKDG and minimize using RDKit UFF
* Convert conformer to PDBQT file
* Dock using QuickVina2-GPU version 2.1

References
----------
- QuickVina2-GPU: https://www.biorxiv.org/content/early/2023/11/05/2023.11.04.565429
- QuickVina2-GPU GitHub: https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
- ETKDG: https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654
- Workflow: https://arxiv.org/abs/2406.08506 Appendix C
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol

if TYPE_CHECKING:
    from collections.abc import Callable

# Suppress RDKit warnings
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

try:
    from loguru import logger

    _LOGGER_AVAILABLE = True
except ImportError:
    _LOGGER_AVAILABLE = False
    logger = None


class DockingScoreGPU:
    """QuickVina2-GPU-based docking score calculator.

    This class provides an interface for calculating docking scores using QuickVina2-GPU
    version 2.1. The implementation follows the protocol from the Saturn repository
    and the workflow described in https://arxiv.org/abs/2406.08506 Appendix C.

    Attributes
    ----------
    _instance : Optional[DockingScoreGPU]
        Singleton instance of the DockingScoreGPU class.
    binary : str
        Path to QuickVina2-GPU binary executable.
    receptor : str
        Path to receptor PDBQT file.
    reference_ligand : str
        Path to reference ligand PDB file for docking box setup.
    box_center : tuple[float, float, float]
        Center coordinates of the docking box.
    box_size : tuple[float, float, float]
        Size of the docking box in Angstroms.
    thread : int
        Exhaustiveness parameter (known as "Thread" in QuickVina2-GPU).
    force_field : callable
        Force field function for ligand energy minimization (UFF or MMFF94).
    output_dir : str
        Directory to save docking results.
    """

    _instance: DockingScoreGPU | None = None
    binary: str | None = None
    receptor: str | None = None
    reference_ligand: str | None = None
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    thread: int = 5000
    force_field: Callable[[Mol], int] | None = None  # type: ignore[valid-type, name-defined, misc]
    output_dir: str | None = None

    def __init__(
        self,
        binary: str | None = None,
        receptor: str | None = None,
        reference_ligand: str | None = None,
        box_size: tuple[float, float, float] | None = None,
        thread: int = 5000,
        force_field: str = "uff",
        output_dir: str | None = None,
    ) -> None:
        """Initialize QuickVina2-GPU docking score calculator.

        Parameters
        ----------
        binary : str, optional
            Path to QuickVina2-GPU binary executable. If None, will attempt to load
            from QUICKVINA2_GPU_BINARY environment variable.
        receptor : str, optional
            Path to receptor PDBQT file. If None, will attempt to load from
            QUICKVINA2_GPU_RECEPTOR environment variable.
        reference_ligand : str, optional
            Path to reference ligand PDB file for docking box setup. If None, will
            attempt to load from QUICKVINA2_GPU_REFERENCE_LIGAND environment variable.
        box_size : tuple[float, float, float], optional
            Size of the docking box in Angstroms. If None, uses default (20, 20, 20).
        thread : int, default=5000
            Exhaustiveness parameter (known as "Thread" in QuickVina2-GPU).
        force_field : str, default="uff"
            Force field for ligand energy minimization. Must be either "uff" or "mmff94".
        output_dir : str, optional
            Directory to save docking results. If None, will attempt to load from
            QUICKVINA2_GPU_OUTPUT_DIR environment variable or use a temporary directory.

        Raises
        ------
        ValueError
            If required parameters are not provided or invalid.
        FileNotFoundError
            If binary, receptor, or reference ligand files don't exist.
        """
        # Determine binary path
        if binary is None:
            binary = os.environ.get("QUICKVINA2_GPU_BINARY")
        if binary is None:
            raise ValueError(
                "QuickVina2-GPU binary path not provided. Set QUICKVINA2_GPU_BINARY "
                "environment variable or provide binary argument."
            )

        binary_path = Path(binary)
        if not binary_path.exists():
            raise FileNotFoundError(f"QuickVina2-GPU binary not found: {binary_path}")

        # Ensure binary is executable
        subprocess.run(["chmod", "u+x", str(binary_path)], check=False)
        self.binary = str(binary_path)

        # Determine receptor path
        if receptor is None:
            receptor = os.environ.get("QUICKVINA2_GPU_RECEPTOR")
        if receptor is None:
            raise ValueError(
                "Receptor PDBQT file path not provided. Set QUICKVINA2_GPU_RECEPTOR "
                "environment variable or provide receptor argument."
            )

        receptor_path = Path(receptor)
        if not receptor_path.exists():
            raise FileNotFoundError(f"Receptor PDBQT file not found: {receptor_path}")
        if not receptor.endswith(".pdbqt"):
            raise ValueError("Receptor file must have .pdbqt extension")
        self.receptor = str(receptor_path)

        # Determine reference ligand path
        if reference_ligand is None:
            reference_ligand = os.environ.get("QUICKVINA2_GPU_REFERENCE_LIGAND")
        if reference_ligand is None:
            raise ValueError(
                "Reference ligand PDB file path not provided. Set QUICKVINA2_GPU_REFERENCE_LIGAND "
                "environment variable or provide reference_ligand argument."
            )

        reference_ligand_path = Path(reference_ligand)
        if not reference_ligand_path.exists():
            raise FileNotFoundError(f"Reference ligand PDB file not found: {reference_ligand_path}")
        if not reference_ligand.endswith(".pdb"):
            raise ValueError("Reference ligand file must have .pdb extension")
        self.reference_ligand = str(reference_ligand_path)

        # Setup force field
        force_field_id = force_field.lower()
        if force_field_id not in ["uff", "mmff94"]:
            raise ValueError("force_field must be either 'uff' or 'mmff94'")
            self.force_field = AllChem.UFFOptimizeMolecule if force_field_id == "uff" else AllChem.MMFFOptimizeMolecule  # type: ignore[attr-defined]

        # Setup docking box based on reference ligand
        self._setup_docking_box()

        # Set box size
        self.box_size = box_size if box_size is not None else (20.0, 20.0, 20.0)

        # Set thread (exhaustiveness)
        self.thread = thread

        # Setup output directory
        if output_dir is None:
            output_dir = os.environ.get("QUICKVINA2_GPU_OUTPUT_DIR")
        if output_dir is None:
            # Use temporary directory if not specified
            output_dir = tempfile.mkdtemp(prefix="quickvina2_gpu_")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        if _LOGGER_AVAILABLE:
            logger.info("QuickVina2-GPU docking score calculator initialized")
            logger.info(f"Binary: {self.binary}")
            logger.info(f"Receptor: {self.receptor}")
            logger.info(f"Reference ligand: {self.reference_ligand}")
            logger.info(f"Box center: {self.box_center}, size: {self.box_size}")
            logger.info(f"Output directory: {self.output_dir}")

    def _setup_docking_box(self) -> None:
        """Setup the docking box for the target receptor based on the reference ligand.

        Follows the protocol from https://arxiv.org/abs/2406.08506 Appendix C:
        * Centroids are the average position of the reference ligand atoms
        * Box size is set to 20 Å x 20 Å x 20 Å (common default)
        """
        # Get the average coordinates of the reference ligand atoms
        ref_mol = Chem.MolFromPDBFile(self.reference_ligand)
        if ref_mol is None:
            raise ValueError(f"Failed to read reference ligand from {self.reference_ligand}")

        ref_conformer = ref_mol.GetConformer()
        ref_coords = ref_conformer.GetPositions()
        ref_center = tuple(np.mean(ref_coords, axis=0))

        # Set the box center
        self.box_center = ref_center

    @classmethod
    def get_instance(
        cls,
        binary: str | None = None,
        receptor: str | None = None,
        reference_ligand: str | None = None,
        box_size: tuple[float, float, float] | None = None,
        thread: int = 5000,
        force_field: str = "uff",
        output_dir: str | None = None,
    ) -> DockingScoreGPU:
        """Get or create singleton instance of DockingScoreGPU.

        Parameters
        ----------
        binary : str, optional
            Path to QuickVina2-GPU binary executable.
        receptor : str, optional
            Path to receptor PDBQT file.
        reference_ligand : str, optional
            Path to reference ligand PDB file.
        box_size : tuple[float, float, float], optional
            Size of the docking box in Angstroms.
        thread : int, default=5000
            Exhaustiveness parameter.
        force_field : str, default="uff"
            Force field for minimization ("uff" or "mmff94").
        output_dir : str, optional
            Directory to save docking results.

        Returns
        -------
        DockingScoreGPU
            Singleton instance of DockingScoreGPU class.
        """
        if cls._instance is None:
            cls._instance = cls(
                binary=binary,
                receptor=receptor,
                reference_ligand=reference_ligand,
                box_size=box_size,
                thread=thread,
                force_field=force_field,
                output_dir=output_dir,
            )
        return cls._instance

    def _compute_property(
        self,
        mols: list[Mol],
        oracle_calls: int = 0,
    ) -> np.ndarray:
        """Execute QuickVina2-GPU-2.1 as a subprocess for a batch of molecules.

        Parameters
        ----------
        mols : list[Mol]
            List of RDKit molecule objects to dock.
        oracle_calls : int, default=0
            Oracle call counter for organizing output directories.

        Returns
        -------
        np.ndarray
            Array of docking scores (one per molecule).
        """
        # 1. Make temporary files to store the input and output
        temp_input_sdf_dir = tempfile.mkdtemp()
        temp_input_pdbqt_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()

        # 2. Convert RDKit Mols to *canonical* SMILES
        canonical_smiles = [Chem.MolToSmiles(mol, canonical=True) for mol in mols]

        # 3. Convert back to RDKit Mols
        # NOTE: This is likely redundant but is done to match the original workflow
        mols = [Chem.MolFromSmiles(smiles) for smiles in canonical_smiles]

        # Filter out None molecules
        valid_indices = [i for i, mol in enumerate(mols) if mol is not None]
        mols = [mols[i] for i in valid_indices]

        if len(mols) == 0:
            # Cleanup and return NaN scores
            shutil.rmtree(temp_input_sdf_dir, ignore_errors=True)
            shutil.rmtree(temp_input_pdbqt_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            return np.full(len(canonical_smiles), float("nan"))

        # 4. Protonate Mols
        mols = [Chem.AddHs(mol) for mol in mols]

        # 5. Generate 1 (lowest energy) conformer using RDKit ETKDG and force-field minimize
        valid_mol_indices = []
        for idx, mol in enumerate(mols):
            # Skip molecules that fail to embed
            try:
                # Generate conformer with ETKDG
                result = AllChem.EmbedMolecule(mol, ETversion=2, randomSeed=0)  # type: ignore[attr-defined]
                if result == -1:
                    continue  # Failed to embed
                # Minimize conformer
                if self.force_field is not None:
                    self.force_field(mol)  # type: ignore[misc]
                valid_mol_indices.append(idx)
                # Write out the minimized conformer in SDF format
                sdf_file = os.path.join(temp_input_sdf_dir, f"ligand_{len(valid_mol_indices)}.sdf")
                writer = Chem.SDWriter(sdf_file)
                writer.write(mol)
                writer.flush()
                writer.close()
                # Convert SDF to PDBQT with OpenBabel
                pdbqt_file = os.path.join(temp_input_pdbqt_dir, f"ligand_{len(valid_mol_indices)}.pdbqt")
                subprocess.run(
                    ["obabel", sdf_file, "-opdbqt", "-O", pdbqt_file],
                    capture_output=True,
                    check=False,
                )
            except Exception:
                continue

        if len(valid_mol_indices) == 0:
            # Cleanup and return NaN scores
            shutil.rmtree(temp_input_sdf_dir, ignore_errors=True)
            shutil.rmtree(temp_input_pdbqt_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            return np.full(len(canonical_smiles), float("nan"))

        # 6. Run QuickVina2-GPU-2.1
        # Subprocess call should occur in the directory of the binary due to needing to read kernel files
        # Based on this GitHub issue: https://github.com/DeltaGroupNJUPT/Vina-GPU/issues/12

        # Change directory to the binary directory
        # Subprocess call should occur in the directory of the binary due to needing to read kernel files
        current_dir = os.getcwd()
        binary_abs_path = os.path.abspath(self.binary)
        binary_dir = os.path.dirname(binary_abs_path)
        binary_name = os.path.basename(binary_abs_path)

        try:
            if binary_dir:
                os.chdir(binary_dir)
                binary_path = f"./{binary_name}"
            else:
                # Binary is in root directory (unlikely but handle it)
                binary_path = binary_abs_path

            output = subprocess.run(
                [
                    binary_path,
                    "--receptor",
                    self.receptor,
                    "--ligand_directory",
                    temp_input_pdbqt_dir,
                    "--output_directory",
                    temp_output_dir,
                    # Exhaustiveness
                    "--thread",
                    str(self.thread),
                    # Docking box
                    "--center_x",
                    str(self.box_center[0]),
                    "--center_y",
                    str(self.box_center[1]),
                    "--center_z",
                    str(self.box_center[2]),
                    "--size_x",
                    str(self.box_size[0]),
                    "--size_y",
                    str(self.box_size[1]),
                    "--size_z",
                    str(self.box_size[2]),
                    # Output only the lowest docking score pose
                    "--num_modes",
                    "1",
                    # Fix seed
                    "--seed",
                    "0",
                ],
                capture_output=True,
            )

            if output.returncode != 0:
                if _LOGGER_AVAILABLE:
                    logger.warning(f"QuickVina2-GPU failed: {output.stderr.decode()}")
        finally:
            # Change back to the original directory
            os.chdir(current_dir)

        # 7. Copy and save the docking output
        if self.output_dir:
            results_dir = os.path.join(self.output_dir, f"results_{oracle_calls}")
            if os.path.exists(results_dir):
                shutil.rmtree(results_dir)
            shutil.copytree(temp_output_dir, results_dir)

        # 8. Parse the docking scores
        docking_scores = np.full(len(canonical_smiles), float("nan"))

        if os.path.exists(temp_output_dir):
            docked_output = os.listdir(temp_output_dir)
            # NOTE: docking_scores is populated below - ligands that failed to embed
            # would already have an assigned score = NaN
            for file in docked_output:
                try:
                    # Extract ligand index from filename (e.g., "ligand_1.pdbqt" -> 1)
                    ligand_idx_str = file.split("_")[1].split(".")[0]
                    ligand_idx = int(ligand_idx_str) - 1  # Convert to 0-based index
                    if ligand_idx >= len(valid_mol_indices):
                        continue
                    original_idx = valid_indices[valid_mol_indices[ligand_idx]]

                    with open(os.path.join(temp_output_dir, file)) as f:
                        for line in f.readlines():
                            # Extract the docking score
                            if "REMARK VINA RESULT" in line:
                                try:
                                    docking_scores[original_idx] = float(line.split()[3])
                                except Exception:
                                    docking_scores[original_idx] = float("nan")
                                break
                except Exception:
                    continue

        # 9. Delete the temporary folders
        shutil.rmtree(temp_input_sdf_dir, ignore_errors=True)
        shutil.rmtree(temp_input_pdbqt_dir, ignore_errors=True)
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        return docking_scores

    @classmethod
    def calculate(cls, mol: Mol) -> float:
        """Calculate docking score for a molecule using QuickVina2-GPU.

        This method uses a singleton instance to avoid reinitializing the
        docking setup on every call. The workflow follows the protocol from
        https://arxiv.org/abs/2406.08506 Appendix C:
        1. Convert to canonical SMILES
        2. Convert to RDKit Mol
        3. Protonate
        4. Generate conformer using RDKit ETKDG
        5. Minimize using RDKit UFF or MMFF94
        6. Convert to PDBQT
        7. Run QuickVina2-GPU docking
        8. Parse docking score

        Parameters
        ----------
        mol : Chem.Mol
            RDKit molecule object.

        Returns
        -------
        float
            Docking score from QuickVina2-GPU. Returns NaN if the molecule cannot
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

            # Process single molecule as batch
            scores = cls._instance._compute_property([mol], oracle_calls=0)

            if len(scores) == 0:
                return float("nan")

            return float(scores[0])

        except Exception as e:
            if _LOGGER_AVAILABLE and logger is not None:
                logger.error(f"Error calculating docking score: {e}")
            return float("nan")

    @classmethod
    def calculate_batch(cls, mols: list[Mol], oracle_calls: int = 0) -> np.ndarray:
        """Calculate docking scores for a batch of molecules using QuickVina2-GPU.

        This method processes multiple molecules in a single QuickVina2-GPU run,
        which is more efficient than processing them individually.

        Parameters
        ----------
        mols : list[Mol]
            List of RDKit molecule objects.
        oracle_calls : int, default=0
            Oracle call counter for organizing output directories.

        Returns
        -------
        np.ndarray
            Array of docking scores (one per molecule). Returns NaN for molecules
            that cannot be processed or if docking fails.
        """
        try:
            # Get or create singleton instance
            if cls._instance is None:
                cls._instance = cls.get_instance()

            # Process batch
            scores = cls._instance._compute_property(mols, oracle_calls=oracle_calls)

            return scores

        except Exception as e:
            if _LOGGER_AVAILABLE and logger is not None:
                logger.error(f"Error calculating docking scores: {e}")
            return np.full(len(mols), float("nan"))


def calculate_docking_gpu(smiles: str) -> float:
    """Calculate the docking score of a molecule using QuickVina2-GPU.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.

    Returns
    -------
    float
        Docking score from QuickVina2-GPU. Returns NaN if the molecule cannot
        be parsed or if an error occurs during docking calculation.

    Notes
    -----
    This function uses QuickVina2-GPU to perform molecular docking and calculate
    binding scores. The docking setup must be initialized by setting the following
    environment variables:
    - QUICKVINA2_GPU_BINARY: Path to QuickVina2-GPU binary executable
    - QUICKVINA2_GPU_RECEPTOR: Path to receptor PDBQT file
    - QUICKVINA2_GPU_REFERENCE_LIGAND: Path to reference ligand PDB file
    - QUICKVINA2_GPU_OUTPUT_DIR: (Optional) Directory to save docking results

    Alternatively, call DockingScoreGPU.get_instance() with explicit paths.

    The workflow follows the protocol from https://arxiv.org/abs/2406.08506 Appendix C:
    1. Convert SMILES to canonical form
    2. Convert to RDKit Mol
    3. Protonate
    4. Generate conformer using RDKit ETKDG
    5. Minimize using RDKit UFF or MMFF94
    6. Convert to PDBQT
    7. Run QuickVina2-GPU docking
    8. Parse docking score

    Docking scores are typically negative values where lower (more negative)
    values indicate better binding affinity. The score represents the predicted
    binding free energy in kcal/mol.

    Requires:
    - QuickVina2-GPU binary (https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1)
    - OpenBabel (for PDBQT conversion)
    - RDKit (for molecular processing and conformer generation)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return float("nan")
    return DockingScoreGPU.calculate(mol)

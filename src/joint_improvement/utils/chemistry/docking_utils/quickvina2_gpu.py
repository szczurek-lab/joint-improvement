"""QuickVina2-GPU docking utilities.

This adapts the original QuickVina2-GPU v2.1 workflow so it can be used by the
`calculate_docking` helper in `joint_improvement.utils.chemistry.docking`. The
implementation mirrors the CPU `DockingVina` interface and keeps the minimal
RDKit/OpenBabel preparation steps described in https://arxiv.org/abs/2406.08506.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, Mol

from joint_improvement.utils.chemistry.docking_utils.dataclass import (
    OracleComponentParameters,
)
from joint_improvement.utils.chemistry.docking_utils.oracle_component import (
    OracleComponent,
)

# Target-specific boxes (centers in Å, sizes in Å)
_TARGET_BOXES: dict[str, dict[str, tuple[float, float, float]]] = {
    "fa7": {"center": (10.131, 41.879, 32.097), "size": (20.673, 20.198, 21.362)},
    "parp1": {"center": (26.413, 11.282, 27.238), "size": (18.521, 17.479, 19.995)},
    "5ht1b": {"center": (-26.602, 5.277, 17.898), "size": (22.5, 22.5, 22.5)},
    "jak2": {"center": (114.758, 65.496, 11.345), "size": (19.033, 17.929, 20.283)},
    "braf": {"center": (84.194, 6.949, -7.081), "size": (22.032, 19.211, 14.106)},
}


def _extract_index_from_name(filename: str) -> int | None:
    """Return zero-based ligand index from QuickVina2-GPU output filename."""
    for token in Path(filename).stem.split("_"):
        if token.isdigit():
            # input files are named ligand_<idx>, so convert to zero-based
            return int(token) - 1
    return None


class DockingVinaGPU:
    """GPU-accelerated docking wrapper compatible with `calculate_docking`."""

    def __init__(
        self,
        target: str,
        binary: str | None = None,
        thread: int = 5000,
        force_field: str = "uff",
        results_dir: str | None = None,
        num_modes: int = 1,
        seed: int = 0,
    ) -> None:
        module_dir = Path(__file__).parent
        self.binary = self._resolve_binary(binary)
        self.receptor = self._resolve_receptor(module_dir, target)
        self.box_center, self.box_size = self._resolve_box(target)
        force_field_id = force_field.lower()
        if force_field_id not in {"uff", "mmff94"}:
            raise ValueError("force_field must be either 'uff' or 'mmff94'")
        self.force_field = (
            AllChem.UFFOptimizeMolecule
            if force_field_id == "uff"
            else AllChem.MMFFOptimizeMolecule
        )
        self.thread = int(thread)
        self.results_dir = Path(results_dir) if results_dir else None
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        self.num_modes = int(num_modes)
        self.seed = int(seed)

    def predict(
        self, smiles_list: Sequence[str], oracle_calls: int | None = None
    ) -> list[float]:
        """Return docking scores for the provided SMILES strings."""
        docking_scores: list[float] = [99.9] * len(smiles_list)
        temp_input_sdf_dir = tempfile.mkdtemp()
        temp_input_pdbqt_dir = tempfile.mkdtemp()
        temp_output_dir = tempfile.mkdtemp()

        prepared_indices: list[int] = []

        try:
            for idx, smiles in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)
                if not self._prepare_conformer(mol, idx, temp_input_sdf_dir, temp_input_pdbqt_dir):
                    continue
                prepared_indices.append(idx)

            if not prepared_indices:
                return docking_scores

            self._run_docking(temp_input_pdbqt_dir, temp_output_dir)
            self._parse_outputs(
                temp_output_dir, docking_scores, expected_count=len(smiles_list)
            )

            if self.results_dir is not None and oracle_calls is not None:
                destination = self.results_dir / f"results_{oracle_calls}"
                shutil.copytree(temp_output_dir, destination, dirs_exist_ok=True)
        finally:
            shutil.rmtree(temp_input_sdf_dir, ignore_errors=True)
            shutil.rmtree(temp_input_pdbqt_dir, ignore_errors=True)
            shutil.rmtree(temp_output_dir, ignore_errors=True)

        return docking_scores

    def _resolve_binary(self, binary: str | None) -> str:
        """Validate and return the QuickVina2-GPU binary path."""
        candidate = binary or os.environ.get("QUICKVINA2_GPU_BINARY")
        if not candidate:
            raise ValueError(
                "Path to QuickVina2-GPU binary is required. "
                "Provide it via the `binary` argument or QUICKVINA2_GPU_BINARY env var."
            )
        binary_path = Path(candidate).expanduser().resolve()
        if not binary_path.exists():
            raise FileNotFoundError(f"QuickVina2-GPU binary not found at {binary_path}")
        binary_path.chmod(binary_path.stat().st_mode | 0o100)  # ensure user executable
        return str(binary_path)

    def _resolve_receptor(self, module_dir: Path, target: str) -> str:
        """Return receptor PDBQT path for the given target."""
        receptor_file = module_dir / f"{target}.pdbqt"
        if not receptor_file.exists():
            raise FileNotFoundError(
                f"Receptor file for target '{target}' not found at {receptor_file}"
            )
        return str(receptor_file.resolve())

    def _resolve_box(
        self, target: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Return docking box center and size for the requested target."""
        if target not in _TARGET_BOXES:
            raise ValueError(
                f"Unsupported target '{target}'. "
                f"Supported targets: {', '.join(sorted(_TARGET_BOXES))}"
            )
        data = _TARGET_BOXES[target]
        return data["center"], data["size"]

    def _prepare_conformer(
        self, mol: Mol, idx: int, sdf_dir: str, pdbqt_dir: str
    ) -> bool:
        """Generate a single conformer and convert it to PDBQT."""
        try:
            AllChem.EmbedMolecule(mol, ETversion=2, randomSeed=0)
            self.force_field(mol)
        except Exception:
            return False

        sdf_file = Path(sdf_dir) / f"ligand_{idx + 1}.sdf"
        pdbqt_file = Path(pdbqt_dir) / f"ligand_{idx + 1}.pdbqt"

        writer = Chem.SDWriter(str(sdf_file))
        writer.write(mol)
        writer.flush()
        writer.close()

        result = subprocess.run(
            ["obabel", str(sdf_file), "-opdbqt", "-O", str(pdbqt_file)],
            capture_output=True,
        )
        return result.returncode == 0 and pdbqt_file.exists()

    def _run_docking(self, ligand_dir: str, output_dir: str) -> None:
        """Invoke QuickVina2-GPU once for the prepared ligand directory."""
        binary_path = Path(self.binary)
        cmd = [
            binary_path.name,
            "--receptor",
            self.receptor,
            "--ligand_directory",
            ligand_dir,
            "--output_directory",
            output_dir,
            "--thread",
            str(self.thread),
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
            "--num_modes",
            str(self.num_modes),
            "--seed",
            str(self.seed),
        ]

        subprocess.run(
            cmd,
            cwd=str(binary_path.parent),
            check=False,
            capture_output=True,
        )

    def _parse_outputs(
        self, output_dir: str, docking_scores: list[float], expected_count: int
    ) -> None:
        """Parse docking scores from QuickVina2-GPU output files."""
        for file in Path(output_dir).iterdir():
            ligand_idx = _extract_index_from_name(file.name)
            if ligand_idx is None or ligand_idx < 0 or ligand_idx >= expected_count:
                continue
            score = self._extract_score_from_file(file)
            docking_scores[ligand_idx] = score

    def _extract_score_from_file(self, file_path: Path) -> float:
        """Extract the first docking score present in the output file."""
        try:
            with file_path.open("r") as handle:
                for line in handle:
                    if "REMARK VINA RESULT" in line:
                        tokens = line.split()
                        for token in tokens:
                            try:
                                return float(token)
                            except ValueError:
                                continue
        except Exception:
            pass
        return 99.9


class QuickVina2_GPU(OracleComponent):
    """OracleComponent wrapper around `DockingVinaGPU` for Saturn-style pipelines."""

    def __init__(self, parameters: OracleComponentParameters) -> None:
        super().__init__(parameters)
        specific = parameters.specific_parameters
        target = specific.get("target", "fa7")
        binary = specific.get("binary")
        thread = int(specific.get("thread", 5000))
        force_field = specific.get("force_field", "uff")
        results_dir = specific.get("results_dir")
        num_modes = int(specific.get("num_modes", 1))
        seed = int(specific.get("seed", 0))

        self.vina_gpu = DockingVinaGPU(
            target=target,
            binary=binary,
            thread=thread,
            force_field=force_field,
            results_dir=results_dir,
            num_modes=num_modes,
            seed=seed,
        )

    def __call__(
        self, mols: NDArray[Mol], oracle_calls: int
    ) -> NDArray[np.float64]:
        smiles = np.vectorize(Chem.MolToSmiles)(mols)
        scores = self.vina_gpu.predict(smiles.tolist(), oracle_calls=oracle_calls)
        return np.array(scores, dtype=float)
"""
Source: https://github.com/schwallergroup/saturn/blob/master/oracles/docking/quickvina2_gpu.py
Workflow based on: https://arxiv.org/abs/2406.08506 Appendix C

Workflow Overview:
* Input canonical SMILES
* Convert to RDKit Mol
* Protonate
* Generate 1 (lowest energy) conformer using RDKit ETKDG and minimize using RDKit UFF
* Convert conformer to PDBQT file
* Dock using QuickVina2-GPU version 2.1

# ETKDG: https://pubs.acs.org/doi/10.1021/acs.jcim.5b00654
"""
import os
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np
from joint_improvement.utils.chemistry.docking_utils.oracle_component import OracleComponent
from joint_improvement.utils.chemistry.docking_utils.dataclass import OracleComponentParameters
from rdkit import Chem
from rdkit.Chem import AllChem, Mol


class DockingVinaGPU(object):
    """GPU docking wrapper that provides SMILES-based interface compatible with DockingVina."""
    
    def __init__(self, target: str, gpu_binary: str | None = None, 
                 gpu_receptor: str | None = None, gpu_reference_ligand: str | None = None,
                 gpu_results_dir: str | None = None, gpu_thread: int = 5000):
        super().__init__()
        
        # Get the directory where docking_utils module is located
        module_dir = Path(__file__).parent
        
        # Set up GPU parameters - use provided values or try to infer from environment/target
        self.target = target
        
        # Try to get GPU binary from environment or use default location
        if gpu_binary is None:
            gpu_binary = os.environ.get("QUICKVINA2_GPU_BINARY", None)
        if gpu_binary is None or not os.path.exists(gpu_binary):
            raise ValueError(
                f"QuickVina2-GPU binary not found. Please provide gpu_binary parameter "
                f"or set QUICKVINA2_GPU_BINARY environment variable."
            )
        
        # Receptor file (same as CPU version)
        if gpu_receptor is None:
            gpu_receptor = str(module_dir / f'{target}.pdbqt')
        if not os.path.exists(gpu_receptor):
            raise ValueError(f"Receptor file not found: {gpu_receptor}")
        
        # Reference ligand - try to find it or use a default location
        if gpu_reference_ligand is None:
            # Try environment variable first
            gpu_reference_ligand = os.environ.get(f"QUICKVINA2_GPU_REF_{target.upper()}", None)
            # If not found, try common location
            if gpu_reference_ligand is None:
                ref_ligand_path = module_dir / f'{target}_ref.pdb'
                if ref_ligand_path.exists():
                    gpu_reference_ligand = str(ref_ligand_path)
        
        if gpu_reference_ligand is None or not os.path.exists(gpu_reference_ligand):
            raise ValueError(
                f"Reference ligand PDB file not found for target {target}. "
                f"Please provide gpu_reference_ligand parameter or set "
                f"QUICKVINA2_GPU_REF_{target.upper()} environment variable."
            )
        
        # Results directory
        if gpu_results_dir is None:
            gpu_results_dir = os.environ.get("QUICKVINA2_GPU_RESULTS_DIR", 
                                            str(Path(tempfile.gettempdir()) / "gpu_docking_results"))
        
        # Initialize QuickVina2_GPU
        params = OracleComponentParameters(
            name="quickvina2_gpu",
            reward_shaping_function_parameters={},
            specific_parameters={
                "binary": gpu_binary,
                "receptor": gpu_receptor,
                "reference_ligand": gpu_reference_ligand,
                "thread": gpu_thread,
                "results_dir": gpu_results_dir,
                "force_field": "uff",
            }
        )
        self.gpu_docker = QuickVina2_GPU(params)
        self.oracle_calls = 0
    
    def predict(self, smiles_list: list[str]) -> list[float]:
        """
        Predict docking scores for a list of SMILES strings.
        
        Parameters
        ----------
        smiles_list : list[str]
            List of SMILES strings
            
        Returns
        -------
        list[float]
            List of docking scores (or 99.9 for failures, matching DockingVina behavior)
        """
        # Convert SMILES to RDKit Mols
        mols = []
        valid_indices = []
        for idx, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)
                valid_indices.append(idx)
        
        if len(mols) == 0:
            return [99.9] * len(smiles_list)
        
        # Run GPU docking
        mols_array = np.array(mols)
        scores = self.gpu_docker(mols_array, oracle_calls=self.oracle_calls)
        self.oracle_calls += 1
        
        # Map scores back to original list (with 99.9 for invalid SMILES or failures)
        result = [99.9] * len(smiles_list)
        zero_count = 0
        for i, idx in enumerate(valid_indices):
            if i < len(scores):
                score = float(scores[i])
                # QuickVina2_GPU returns 0.0 for failures
                # Note: 0.0 could be a valid score (very poor binding), but typically indicates failure
                # We'll keep 0.0 as-is for now, but mark very small scores as potentially failed
                if score == 0.0:
                    zero_count += 1
                    # Keep 0.0 but it will be filtered out in comparison (as it's likely a failure)
                result[idx] = score
        
        # Debug: warn if all scores are 0.0 (likely indicates GPU docking failure)
        if zero_count == len(scores) and len(scores) > 0:
            import warnings
            warnings.warn(
                f"All GPU docking scores are 0.0. This likely indicates GPU docking failed. "
                f"Check GPU binary, receptor file, and reference ligand configuration.",
                UserWarning
            )
        
        return result


class QuickVina2_GPU(OracleComponent):
    """
    Executes QuickVina2-GPU version 2.1.

    References:
    1. https://www.biorxiv.org/content/early/2023/11/05/2023.11.04.565429
    2. https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1
    """
    def __init__(self, parameters: OracleComponentParameters):
        super().__init__(parameters)

        # QuickVina2-GPU-2.1 binary
        self.binary = self.parameters.specific_parameters.get("binary", None)
        assert self.binary is not None, "Please provide the path to the QuickVina2-GPU binary."
        # Ensure the binary is executable for the user
        subprocess.run(["chmod", "u+x", self.binary])

        # Force-field for ligand energy minimization
        force_field_id = self.parameters.specific_parameters.get("force_field", "uff").lower()
        assert force_field_id in ["uff", "mmff94"], "force_field must be either 'uff' or 'mmff94'."
        self.force_field = AllChem.UFFOptimizeMolecule if force_field_id == "uff" else AllChem.MMFFOptimizeMolecule

        # Receptor path
        self.receptor = self.parameters.specific_parameters.get("receptor", None)
        assert self.receptor is not None and self.receptor.endswith(".pdbqt"), "Please provide the path to the receptor PDBQT file."

        # Reference ligand path
        self.reference_ligand = self.parameters.specific_parameters.get("reference_ligand", None)
        assert self.reference_ligand is not None and self.reference_ligand.endswith(".pdb"), "Please provide the path to the reference ligand PDB file."

        # Exhaustiveness (known as "Thread" in QuickVina2-GPU)
        self.thread = self.parameters.specific_parameters.get("thread", 5000)  # Default to 5000 (same as source code default)

        # Setup docking box
        self._setup_docking_box()

        # Output directory
        output_dir = self.parameters.specific_parameters.get("results_dir", None)
        assert output_dir not in [None, ""], "Please provide the path to the output directory."
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __call__(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        return self._compute_property(mols, oracle_calls)
    
    def _compute_property(
        self, 
        mols: np.ndarray[Mol],
        oracle_calls: int
    ) -> np.ndarray[float]:
        """
        Execute QuickVina2-GPU-2.1 as a subprocess.
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

        # 4. Protonate Mols
        mols = [Chem.AddHs(mol) for mol in mols]

        # 5. Generate 1 (lowest energy) conformer using RDKit ETKDG and force-field (UFF or MMFF94) minimize
        for idx, mol in enumerate(mols):
        # Skip molecules that fail to embed
            try:
                # Generate conformer with ETKDG
                AllChem.EmbedMolecule(mol, ETversion=2, randomSeed=0)
                # Minimize conformer
                self.force_field(mol)
            except Exception:
                continue
            # Write out the minimized conformer in SDF format
            sdf_file = os.path.join(temp_input_sdf_dir, f"ligand_{idx+1}.sdf")
            writer = Chem.SDWriter(sdf_file)
            writer.write(mol)
            writer.flush()
            writer.close()
            # Convert SDF to PDBQT with OpenBabel
            pdbqt_file = os.path.join(temp_input_pdbqt_dir, f"ligand_{idx+1}.pdbqt")
            output = subprocess.run([
                "obabel",
                sdf_file,
                "-opdbqt",
                "-O",
                pdbqt_file
            ], capture_output=True)

        # 6. Run QuickVina2-GPU-2.1
        # TODO: Could be paralellized but GPU docking should be fast enough as large libraries are not expected
            
        # Subprocess call should occur in the directory of the binary due to needing to read kernel files
        # Based on this GitHub issue: https://github.com/DeltaGroupNJUPT/Vina-GPU/issues/12
            
        # Change directory to the binary directory
        current_dir = os.getcwd()
        os.chdir(os.path.dirname(self.binary))

        output = subprocess.run([
            "./QuickVina2-GPU-2-1",
            "--receptor", self.receptor,
            "--ligand_directory", temp_input_pdbqt_dir,
            "--output_directory", temp_output_dir,
            # Exhaustiveness
            "--thread", str(self.thread),
            # Docking box
            "--center_x", str(self.box_center[0]), "--center_y", str(self.box_center[1]), "--center_z", str(self.box_center[2]),
            "--size_x", str(self.box_size[0]), "--size_y", str(self.box_size[1]), "--size_z", str(self.box_size[2]),
            # Output only the lowest docking score pose
            "--num_modes", "1",
            # Fix seed
            # TODO: Expose parameter to user
            "--seed", "0"
        ], capture_output=True)

        # Change back to the original directory
        os.chdir(current_dir)

        # 7. Copy and save the docking output
        subprocess.run([
            "cp", 
            "-r", 
            temp_output_dir, 
            os.path.join(self.output_dir, f"results_{oracle_calls}")
        ])

        # 8. Parse the docking scores
        docking_scores = np.zeros(len(mols))
        docked_output = os.listdir(temp_output_dir)
        # NOTE: docking_scores is populated below - ligands that failed to embed would already have an assigned score = 0.0
        for file in docked_output:
            ligand_idx = int(file.split("_")[1]) - 1
            with open(os.path.join(temp_output_dir, file), "r") as f:
                for line in f.readlines():
                    # Extract the docking score
                    if "REMARK VINA RESULT" in line:
                        try:
                            docking_scores[ligand_idx] = (float(line.split()[3]))
                        except Exception:
                            docking_scores[ligand_idx] = 0.0
       
        # 9. Delete the temporary folders
        shutil.rmtree(temp_input_sdf_dir)
        shutil.rmtree(temp_input_pdbqt_dir)
        shutil.rmtree(temp_output_dir)

        return docking_scores

    def _setup_docking_box(self):
        """
        Setup the docking box for the target receptor based on the reference ligand.

        Follows the protocol from https://arxiv.org/abs/2406.08506 Appendix C:

        * Centroids are the average position of the reference ligand atoms

        * "Box sizes individually determined to encompass each target binding"
           Unclear how this is done - box size is set to 20 Å x 20 Å x 20 Å instead which is a common default
        """
        # Get the average coordinates of the reference ligand atoms
        ref_mol = Chem.MolFromPDBFile(self.reference_ligand)
        ref_conformer = ref_mol.GetConformer()
        ref_coords = ref_conformer.GetPositions()
        ref_center = tuple(np.mean(ref_coords, axis=0)) 

        # Set the box center and size
        self.box_center = ref_center  # Tuple[float, float, float]
        self.box_size = (20.0, 20.0, 20.0)

"""
Kaggle Checkpoint Manager for Multi-Session Training.

Handles automatic checkpoint persistence to Kaggle Datasets,
enabling seamless resume across multiple Kaggle notebook sessions.

Features:
- Upload checkpoints to Kaggle Dataset
- Download latest checkpoint at session start
- Version tracking to prevent overwrites
- Automatic dataset initialization
"""

import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime


class KaggleCheckpointManager:
    """
    Manages checkpoint persistence to Kaggle Datasets.

    Workflow:
        1. First session: Creates Kaggle Dataset, uploads checkpoints
        2. Subsequent sessions: Downloads latest checkpoint, resumes training

    Example:
        >>> manager = KaggleCheckpointManager(
        ...     dataset_slug="username/breast-cancer-mobo-checkpoints",
        ...     output_dir="results/botorch_mobo/run_001"
        ... )
        >>> # At session start
        >>> checkpoint = manager.download_latest_checkpoint()
        >>> # During training
        >>> manager.upload_checkpoint("results/.../checkpoint_iter10.pt")
    """

    def __init__(
        self,
        dataset_slug: str,
        output_dir: str,
        verbose: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize Kaggle checkpoint manager.

        Args:
            dataset_slug: Kaggle dataset slug (format: "username/dataset-name")
                         Example: "johndoe/breast-cancer-mobo-checkpoints"
            output_dir: Local directory for checkpoints
            verbose: Print detailed logs
            max_retries: Maximum upload/download retry attempts

        Raises:
            ValueError: If dataset_slug is invalid format
        """
        # Validate dataset slug format
        if '/' not in dataset_slug or dataset_slug.count('/') != 1:
            raise ValueError(
                f"Invalid dataset_slug: '{dataset_slug}'. "
                "Expected format: 'username/dataset-name'"
            )

        self.dataset_slug = dataset_slug
        self.username, self.dataset_name = dataset_slug.split('/')
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.max_retries = max_retries

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metadata tracking
        self.metadata_file = self.output_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()

        # Check if running on Kaggle
        self.is_kaggle = self._detect_kaggle_environment()

        if self.verbose:
            print(f"Kaggle Checkpoint Manager initialized:")
            print(f"  Dataset: {self.dataset_slug}")
            print(f"  Local dir: {self.output_dir}")
            print(f"  Running on Kaggle: {self.is_kaggle}")
            print(f"  Checkpoints tracked: {len(self.metadata.get('checkpoints', []))}")

    def _detect_kaggle_environment(self) -> bool:
        """Detect if running in Kaggle environment."""
        return os.path.exists('/kaggle/working')

    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata from JSON file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'last_upload': None, 'session_count': 0}

    def _save_metadata(self):
        """Save checkpoint metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _run_kaggle_command(self, command: List[str], retry: int = 0) -> bool:
        """
        Run a Kaggle API command with retry logic.

        Args:
            command: Command to run (as list)
            retry: Current retry attempt

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.verbose:
                print(f"  Running: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode == 0:
                if self.verbose:
                    print(f"  ✓ Success")
                return True
            else:
                error_msg = result.stderr.strip()
                print(f"  ✗ Command failed: {error_msg}")

                # Retry logic
                if retry < self.max_retries:
                    wait_time = 2 ** retry  # Exponential backoff
                    print(f"  Retrying in {wait_time} seconds... (attempt {retry+1}/{self.max_retries})")
                    time.sleep(wait_time)
                    return self._run_kaggle_command(command, retry + 1)

                return False

        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
            if retry < self.max_retries:
                wait_time = 2 ** retry
                print(f"  Retrying in {wait_time} seconds... (attempt {retry+1}/{self.max_retries})")
                time.sleep(wait_time)
                return self._run_kaggle_command(command, retry + 1)
            return False

    def dataset_exists(self) -> bool:
        """
        Check if the Kaggle dataset exists.

        Returns:
            True if dataset exists, False otherwise
        """
        command = ['kaggle', 'datasets', 'list', '-s', self.dataset_slug]
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode == 0:
            # Check if our dataset is in the output
            return self.dataset_slug in result.stdout
        return False

    def create_dataset(self) -> bool:
        """
        Create a new Kaggle dataset for checkpoints.

        Returns:
            True if created successfully, False otherwise
        """
        if self.verbose:
            print(f"\nCreating Kaggle dataset: {self.dataset_slug}")

        # Create dataset metadata file
        temp_dir = self.output_dir / "temp_dataset_init"
        temp_dir.mkdir(exist_ok=True)

        metadata = {
            "title": f"{self.dataset_name}",
            "id": f"{self.username}/{self.dataset_name}",
            "licenses": [{"name": "CC0-1.0"}]
        }

        metadata_path = temp_dir / "dataset-metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create a placeholder file
        placeholder = temp_dir / "README.md"
        with open(placeholder, 'w') as f:
            f.write("# Multi-Session Optimization Checkpoints\n\n")
            f.write("This dataset stores checkpoints for multi-session ")
            f.write("Bayesian optimization on Kaggle.\n\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Create dataset
        command = ['kaggle', 'datasets', 'create', '-p', str(temp_dir)]
        success = self._run_kaggle_command(command)

        # Cleanup
        shutil.rmtree(temp_dir)

        if success:
            print(f"  ✓ Dataset created: {self.dataset_slug}")
        return success

    def upload_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Upload checkpoint to Kaggle Dataset.

        Args:
            checkpoint_path: Path to checkpoint file to upload

        Returns:
            True if upload successful, False otherwise
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            return False

        if self.verbose:
            print(f"\nUploading checkpoint to Kaggle Dataset...")
            print(f"  File: {checkpoint_path.name}")
            print(f"  Size: {checkpoint_path.stat().st_size / (1024**2):.1f} MB")

        # Check if dataset exists, create if not
        if not self.dataset_exists():
            print(f"  Dataset '{self.dataset_slug}' not found. Creating...")
            if not self.create_dataset():
                print(f"  ✗ Failed to create dataset")
                return False
            # Wait for dataset creation to propagate
            time.sleep(5)

        # Prepare upload directory
        upload_dir = self.output_dir / "temp_upload"
        upload_dir.mkdir(exist_ok=True)

        # Copy checkpoint to upload directory
        upload_checkpoint = upload_dir / checkpoint_path.name
        shutil.copy2(checkpoint_path, upload_checkpoint)

        # Copy metadata file
        if self.metadata_file.exists():
            shutil.copy2(self.metadata_file, upload_dir / "checkpoint_metadata.json")

        # Update dataset metadata
        dataset_metadata = {
            "id": self.dataset_slug,
            "title": self.dataset_name,
            "licenses": [{"name": "CC0-1.0"}]
        }

        metadata_path = upload_dir / "dataset-metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)

        # Upload as new version
        command = [
            'kaggle', 'datasets', 'version',
            '-p', str(upload_dir),
            '-m', f"Checkpoint: {checkpoint_path.name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]

        success = self._run_kaggle_command(command)

        # Cleanup
        shutil.rmtree(upload_dir)

        if success:
            # Update local metadata
            self.metadata['checkpoints'].append({
                'filename': checkpoint_path.name,
                'upload_time': datetime.now().isoformat(),
                'size_mb': checkpoint_path.stat().st_size / (1024**2)
            })
            self.metadata['last_upload'] = datetime.now().isoformat()
            self._save_metadata()

            print(f"  ✓ Checkpoint uploaded successfully")
            print(f"  View at: https://www.kaggle.com/datasets/{self.dataset_slug}")

        return success

    def download_latest_checkpoint(self) -> Optional[Path]:
        """
        Download the latest checkpoint from Kaggle Dataset.

        Returns:
            Path to downloaded checkpoint file, or None if no checkpoint found
        """
        if self.verbose:
            print(f"\nChecking Kaggle Dataset for checkpoints...")

        # Check if dataset exists
        if not self.dataset_exists():
            if self.verbose:
                print(f"  Dataset '{self.dataset_slug}' not found")
            return None

        # Download dataset
        download_dir = self.output_dir / "temp_download"
        download_dir.mkdir(exist_ok=True)

        command = [
            'kaggle', 'datasets', 'download',
            '-d', self.dataset_slug,
            '-p', str(download_dir),
            '--unzip'
        ]

        if self.verbose:
            print(f"  Downloading from {self.dataset_slug}...")

        success = self._run_kaggle_command(command)

        if not success:
            shutil.rmtree(download_dir, ignore_errors=True)
            return None

        # Find checkpoint files
        checkpoint_files = list(download_dir.glob("checkpoint_*.pt"))

        if not checkpoint_files:
            if self.verbose:
                print(f"  No checkpoint files found in dataset")
            shutil.rmtree(download_dir, ignore_errors=True)
            return None

        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_checkpoint = checkpoint_files[0]

        # Move to output directory
        destination = self.output_dir / latest_checkpoint.name
        shutil.move(str(latest_checkpoint), str(destination))

        # Also get metadata if available
        metadata_file = download_dir / "checkpoint_metadata.json"
        if metadata_file.exists():
            shutil.copy2(metadata_file, self.metadata_file)
            self.metadata = self._load_metadata()

        # Cleanup
        shutil.rmtree(download_dir, ignore_errors=True)

        if self.verbose:
            print(f"  ✓ Downloaded: {destination.name}")
            print(f"  Size: {destination.stat().st_size / (1024**2):.1f} MB")

        return destination

    def list_available_checkpoints(self) -> List[str]:
        """
        List all checkpoints tracked in metadata.

        Returns:
            List of checkpoint filenames
        """
        return [cp['filename'] for cp in self.metadata.get('checkpoints', [])]

    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """
        Get path to the latest local checkpoint file.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        checkpoint_files = list(self.output_dir.glob("checkpoint_*.pt"))

        if not checkpoint_files:
            return None

        # Sort by modification time
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoint_files[0]

    def validate_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Validate checkpoint file integrity.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if checkpoint is valid, False otherwise
        """
        try:
            import torch

            # Try to load checkpoint
            state = torch.load(checkpoint_path, map_location='cpu')

            # Check for required keys
            required_keys = ['X_train', 'Y_train', 'iteration']
            for key in required_keys:
                if key not in state:
                    if self.verbose:
                        print(f"  ✗ Checkpoint missing required key: {key}")
                    return False

            if self.verbose:
                print(f"  ✓ Checkpoint validation passed")
                print(f"    Iteration: {state['iteration']}")
                print(f"    Evaluations: {state['X_train'].shape[0]}")

            return True

        except Exception as e:
            if self.verbose:
                print(f"  ✗ Checkpoint validation failed: {str(e)}")
            return False

    def get_session_count(self) -> int:
        """Get the number of sessions recorded."""
        return self.metadata.get('session_count', 0)

    def increment_session_count(self):
        """Increment session counter."""
        self.metadata['session_count'] = self.metadata.get('session_count', 0) + 1
        self._save_metadata()

    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """
        Remove old local checkpoint files, keeping only the most recent N.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoint_files = list(self.output_dir.glob("checkpoint_*.pt"))

        if len(checkpoint_files) <= keep_last_n:
            return

        # Sort by modification time
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[keep_last_n:]:
            if self.verbose:
                print(f"  Removing old checkpoint: {old_checkpoint.name}")
            old_checkpoint.unlink()

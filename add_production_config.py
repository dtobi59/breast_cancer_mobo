"""
Add a production configuration cell at the beginning of the notebook.
This makes it easy to switch between demo and production mode.
"""

import json
from pathlib import Path
import shutil

# Read the notebook
notebook_path = Path("breast_cancer_colab_demo.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Create backup
backup_path = notebook_path.with_suffix('.ipynb.backup11')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Create production configuration cell
# This should go after Section 2 (after paths are set up, before Section 3)
production_config_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# EXPERIMENT CONFIGURATION\n",
        "# ============================================================================\n",
        "#\n",
        "# Configure the entire notebook for DEMO or PRODUCTION mode\n",
        "# Change EXPERIMENT_MODE to switch between modes\n",
        "#\n",
        "# ============================================================================\n",
        "\n",
        "from datetime import datetime\n",
        "\n",
        "# Set experiment mode: \"DEMO\" or \"PRODUCTION\"\n",
        "EXPERIMENT_MODE = \"DEMO\"  # Change to \"PRODUCTION\" for main experiment\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(f\"EXPERIMENT MODE: {EXPERIMENT_MODE}\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "if EXPERIMENT_MODE == \"PRODUCTION\":\n",
        "    # ========================================================================\n",
        "    # PRODUCTION CONFIGURATION (20-50 hours)\n",
        "    # ========================================================================\n",
        "    \n",
        "    print(\"\\n[PRODUCTION MODE]\")\n",
        "    print(\"  - Uses FULL VinDr-Mammo dataset\")\n",
        "    print(\"  - NSGA-III: 20 population x 50 generations = 1000 evaluations\")\n",
        "    print(\"  - Training: 50 epochs per model\")\n",
        "    print(\"  - Adaptive preprocessing: ENABLED\")\n",
        "    print(\"  - Checkpointing: ENABLED\")\n",
        "    print(\"  - Expected runtime: 20-50 hours\")\n",
        "    print(\"  - GPU required: Tesla T4 or better\")\n",
        "    \n",
        "    # Global configuration\n",
        "    CONFIG = {\n",
        "        # Dataset settings\n",
        "        'use_full_dataset': True,\n",
        "        'train_batch_size': 8,\n",
        "        'val_batch_size': 8,\n",
        "        'num_workers': 2,\n",
        "        \n",
        "        # Preprocessing\n",
        "        'use_adaptive_preprocessing': True,\n",
        "        \n",
        "        # Training settings\n",
        "        'max_epochs': 50,\n",
        "        'patience': 10,\n",
        "        'learning_rate': 1e-4,\n",
        "        'weight_decay': 1e-4,\n",
        "        \n",
        "        # NSGA-III settings\n",
        "        'nsga3_population': 20,\n",
        "        'nsga3_generations': 50,\n",
        "        'enable_checkpointing': True,\n",
        "        'resume_if_available': True,\n",
        "        \n",
        "        # Run ID (auto-generated with timestamp)\n",
        "        'run_id': f\"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}\",\n",
        "        \n",
        "        # Mode\n",
        "        'mode': 'PRODUCTION'\n",
        "    }\n",
        "    \n",
        "    print(f\"\\nRun ID: {CONFIG['run_id']}\")\n",
        "    print(\"\\n[WARNING] This will take 20-50 hours!\")\n",
        "    print(\"[WARNING] Ensure GPU runtime is selected\")\n",
        "    print(\"[WARNING] Ensure Google Drive has 20+ GB free space\")\n",
        "    print(\"[WARNING] For Colab free tier: Plan for disconnections\")\n",
        "    print(\"           Checkpoints will save progress automatically.\")\n",
        "    \n",
        "elif EXPERIMENT_MODE == \"DEMO\":\n",
        "    # ========================================================================\n",
        "    # DEMO CONFIGURATION (1-2 hours)\n",
        "    # ========================================================================\n",
        "    \n",
        "    print(\"\\n[DEMO MODE]\")\n",
        "    print(\"  - Uses SUBSET of VinDr-Mammo (~100 samples)\")\n",
        "    print(\"  - NSGA-III: 5 population x 3 generations = 15 evaluations\")\n",
        "    print(\"  - Training: 5 epochs per model\")\n",
        "    print(\"  - Adaptive preprocessing: ENABLED\")\n",
        "    print(\"  - Checkpointing: ENABLED\")\n",
        "    print(\"  - Expected runtime: 1-2 hours\")\n",
        "    print(\"  - Good for testing and development\")\n",
        "    \n",
        "    # Global configuration\n",
        "    CONFIG = {\n",
        "        # Dataset settings\n",
        "        'use_full_dataset': False,\n",
        "        'train_batch_size': 4,\n",
        "        'val_batch_size': 4,\n",
        "        'num_workers': 2,\n",
        "        \n",
        "        # Preprocessing\n",
        "        'use_adaptive_preprocessing': True,\n",
        "        \n",
        "        # Training settings\n",
        "        'max_epochs': 5,\n",
        "        'patience': 3,\n",
        "        'learning_rate': 1e-4,\n",
        "        'weight_decay': 1e-4,\n",
        "        \n",
        "        # NSGA-III settings\n",
        "        'nsga3_population': 5,\n",
        "        'nsga3_generations': 3,\n",
        "        'enable_checkpointing': True,\n",
        "        'resume_if_available': True,\n",
        "        \n",
        "        # Run ID\n",
        "        'run_id': f\"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}\",\n",
        "        \n",
        "        # Mode\n",
        "        'mode': 'DEMO'\n",
        "    }\n",
        "    \n",
        "    print(f\"\\nRun ID: {CONFIG['run_id']}\")\n",
        "    print(\"\\n[INFO] This is a quick test run\")\n",
        "    print(\"[INFO] Results will not be production-quality\")\n",
        "    print(\"[INFO] Use this to verify the pipeline works end-to-end\")\n",
        "    \n",
        "else:\n",
        "    raise ValueError(f\"Invalid EXPERIMENT_MODE: {EXPERIMENT_MODE}. Must be 'DEMO' or 'PRODUCTION'\")\n",
        "\n",
        "# Display configuration\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"CONFIGURATION SUMMARY\")\n",
        "print(\"=\"*80)\n",
        "print(f\"\\nDataset:\")\n",
        "print(f\"  Full dataset: {CONFIG['use_full_dataset']}\")\n",
        "print(f\"  Batch size: {CONFIG['train_batch_size']}\")\n",
        "print(f\"\\nTraining:\")\n",
        "print(f\"  Max epochs: {CONFIG['max_epochs']}\")\n",
        "print(f\"  Patience: {CONFIG['patience']}\")\n",
        "print(f\"  Learning rate: {CONFIG['learning_rate']}\")\n",
        "print(f\"\\nOptimization:\")\n",
        "print(f\"  Population: {CONFIG['nsga3_population']}\")\n",
        "print(f\"  Generations: {CONFIG['nsga3_generations']}\")\n",
        "print(f\"  Total evals: {CONFIG['nsga3_population'] * CONFIG['nsga3_generations']}\")\n",
        "print(f\"\\nOther:\")\n",
        "print(f\"  Adaptive preprocessing: {CONFIG['use_adaptive_preprocessing']}\")\n",
        "print(f\"  Checkpointing: {CONFIG['enable_checkpointing']}\")\n",
        "print(f\"  Resume enabled: {CONFIG['resume_if_available']}\")\n",
        "print(f\"  Run ID: {CONFIG['run_id']}\")\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "\n",
        "# Set individual variables for backward compatibility\n",
        "USE_FULL_DATASET = CONFIG['use_full_dataset']\n",
        "USE_ADAPTIVE_PREPROCESSING = CONFIG['use_adaptive_preprocessing']\n",
        "RUN_ID = CONFIG['run_id']\n",
        "ENABLE_CHECKPOINTING = CONFIG['enable_checkpointing']\n",
        "RESUME_IF_AVAILABLE = CONFIG['resume_if_available']\n",
        "\n",
        "print(\"\\n[OK] Configuration loaded successfully\")\n",
        "print(\"[OK] All sections will use these settings automatically\\n\")"
    ]
}

# Find where to insert (after Section 2, before Section 3)
# Section 2 ends with the path verification cell (cell 8)
insert_idx = None
for i, cell in enumerate(cells):
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    if '## 3. Preprocessing Configuration' in source:
        insert_idx = i
        break

if insert_idx is None:
    print("[ERROR] Could not find Section 3")
    exit(1)

print(f"[OK] Will insert production config at index {insert_idx}")

# Insert the configuration cell before Section 3
cells.insert(insert_idx, production_config_cell)

# Now we need to update Section 3 and Section 8 to use CONFIG instead of their own variables
# Find and update Section 3 (Preprocessing Configuration)
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)

        # Update Section 3 preprocessing config
        if 'USE_ADAPTIVE_PREPROCESSING = True' in source and 'PREPROCESSING CONFIGURATION' in source:
            # Comment out the old variable and use CONFIG instead
            updated_source = source.replace(
                'USE_ADAPTIVE_PREPROCESSING = True',
                '# USE_ADAPTIVE_PREPROCESSING is now set by CONFIG above\n# USE_ADAPTIVE_PREPROCESSING = True'
            )
            if isinstance(cell['source'], list):
                cell['source'] = [updated_source]
            else:
                cell['source'] = updated_source
            print(f"[OK] Updated Section 3 preprocessing config at index {i}")

        # Update Section 8 dataset selection
        if 'USE_FULL_DATASET = False' in source and 'DATASET SELECTION' in source:
            updated_source = source.replace(
                'USE_FULL_DATASET = False',
                '# USE_FULL_DATASET is now set by CONFIG above\n# USE_FULL_DATASET = False'
            )
            if isinstance(cell['source'], list):
                cell['source'] = [updated_source]
            else:
                cell['source'] = updated_source
            print(f"[OK] Updated Section 8 dataset selection at index {i}")

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Added production configuration cell")
print(f"[OK] Updated existing config sections to use CONFIG")
print(f"[OK] Notebook updated with {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")
print(f"\n[USAGE] Change EXPERIMENT_MODE = 'PRODUCTION' to run main experiment")

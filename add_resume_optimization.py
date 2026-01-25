"""
Add resume optimization capability to Section 11 (NSGA-III).
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
backup_path = notebook_path.with_suffix('.ipynb.backup10')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Find Section 11 (NSGA-III Optimization)
section11_idx = None
nsga3_code_idx = None

for i, cell in enumerate(cells):
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)

    if '## 11. NSGA-III Optimization' in source:
        section11_idx = i
    elif 'from src.optimization import BreastCancerOptimizationProblem' in source:
        nsga3_code_idx = i
        break

print(f"[OK] Found Section 11 at index {section11_idx}")
print(f"[OK] Found NSGA-III code at index {nsga3_code_idx}")

# Create new NSGA-III code cell with resume capability
nsga3_with_resume = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from src.optimization import BreastCancerOptimizationProblem\n",
        "from pymoo.algorithms.moo.nsga3 import NSGA3\n",
        "from pymoo.optimize import minimize\n",
        "from pymoo.util.ref_dirs import get_reference_directions\n",
        "import pickle\n",
        "import os\n",
        "from datetime import datetime\n",
        "import time\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"NSGA-III OPTIMIZATION (DEMO VERSION)\")\n",
        "print(\"=\"*80)\n",
        "print(\"\\nWarning: Full optimization takes 20-50 hours!\")\n",
        "print(\"This demo uses: 5 population x 3 generations = 15 evaluations\")\n",
        "print(\"For production, use: 20 population x 50 generations = 1000 evaluations\\n\")\n",
        "\n",
        "# ============================================================================\n",
        "# CHECKPOINT CONFIGURATION\n",
        "# ============================================================================\n",
        "\n",
        "# Set this to True to enable checkpointing (saves after each generation)\n",
        "ENABLE_CHECKPOINTING = True\n",
        "\n",
        "# Set this to True to resume from last checkpoint if available\n",
        "RESUME_IF_AVAILABLE = True\n",
        "\n",
        "# Checkpoint directory\n",
        "checkpoint_dir = os.path.join(PROJECT_PATH, \"optimization_checkpoints\")\n",
        "os.makedirs(checkpoint_dir, exist_ok=True)\n",
        "\n",
        "# Run ID (unique identifier for this optimization run)\n",
        "# Change this for different optimization runs\n",
        "RUN_ID = \"demo_run_001\"  # or use timestamp: f\"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}\"\n",
        "\n",
        "checkpoint_path = os.path.join(checkpoint_dir, f\"nsga3_{RUN_ID}_checkpoint.pkl\")\n",
        "results_path = os.path.join(checkpoint_dir, f\"nsga3_{RUN_ID}_results.pkl\")\n",
        "\n",
        "print(f\"Checkpoint configuration:\")\n",
        "print(f\"  Checkpointing: {'ENABLED' if ENABLE_CHECKPOINTING else 'DISABLED'}\")\n",
        "print(f\"  Resume: {'ENABLED' if RESUME_IF_AVAILABLE else 'DISABLED'}\")\n",
        "print(f\"  Run ID: {RUN_ID}\")\n",
        "print(f\"  Checkpoint file: {os.path.basename(checkpoint_path)}\")\n",
        "\n",
        "# ============================================================================\n",
        "# CHECK FOR EXISTING CHECKPOINT\n",
        "# ============================================================================\n",
        "\n",
        "resume_from_checkpoint = False\n",
        "starting_generation = 0\n",
        "checkpoint_data = None\n",
        "\n",
        "if RESUME_IF_AVAILABLE and os.path.exists(checkpoint_path):\n",
        "    print(f\"\\n[FOUND] Existing checkpoint at: {checkpoint_path}\")\n",
        "    \n",
        "    try:\n",
        "        with open(checkpoint_path, 'rb') as f:\n",
        "            checkpoint_data = pickle.load(f)\n",
        "        \n",
        "        starting_generation = checkpoint_data['generation']\n",
        "        print(f\"[OK] Checkpoint loaded successfully\")\n",
        "        print(f\"     Last completed generation: {starting_generation}\")\n",
        "        print(f\"     Evaluations completed: {checkpoint_data['n_eval']}\")\n",
        "        print(f\"     Checkpoint timestamp: {checkpoint_data['timestamp']}\")\n",
        "        \n",
        "        resume_from_checkpoint = True\n",
        "        \n",
        "    except Exception as e:\n",
        "        print(f\"[ERROR] Failed to load checkpoint: {e}\")\n",
        "        print(f\"[INFO] Starting optimization from scratch\")\n",
        "        resume_from_checkpoint = False\n",
        "else:\n",
        "    if RESUME_IF_AVAILABLE:\n",
        "        print(f\"\\n[INFO] No checkpoint found. Starting new optimization run.\")\n",
        "    else:\n",
        "        print(f\"\\n[INFO] Resume disabled. Starting new optimization run.\")\n",
        "\n",
        "# ============================================================================\n",
        "# CREATE OPTIMIZATION PROBLEM\n",
        "# ============================================================================\n",
        "\n",
        "problem = BreastCancerOptimizationProblem(\n",
        "    train_dataset=small_train_dataset if not USE_FULL_DATASET else train_dataset,\n",
        "    val_dataset=small_val_dataset if not USE_FULL_DATASET else val_dataset,\n",
        "    device=device,\n",
        "    batch_size=4,\n",
        "    num_workers=2,\n",
        "    patience=3,\n",
        "    max_epochs=5 if not USE_FULL_DATASET else 50,\n",
        "    pos_weight=n_benign/n_malignant,\n",
        "    random_seed=42\n",
        ")\n",
        "\n",
        "# Generate reference directions for 4 objectives\n",
        "ref_dirs = get_reference_directions(\"das-dennis\", 4, n_partitions=3)\n",
        "\n",
        "# ============================================================================\n",
        "# CREATE OR RESTORE ALGORITHM\n",
        "# ============================================================================\n",
        "\n",
        "if resume_from_checkpoint:\n",
        "    print(\"\\n[RESUMING] Restoring algorithm state from checkpoint...\")\n",
        "    \n",
        "    # Restore algorithm state\n",
        "    algorithm = checkpoint_data['algorithm']\n",
        "    \n",
        "    # Calculate remaining generations\n",
        "    total_generations = 3  # DEMO: 3, for production use 50\n",
        "    remaining_generations = total_generations - starting_generation\n",
        "    \n",
        "    print(f\"  Total generations: {total_generations}\")\n",
        "    print(f\"  Completed: {starting_generation}\")\n",
        "    print(f\"  Remaining: {remaining_generations}\")\n",
        "    \n",
        "    if remaining_generations <= 0:\n",
        "        print(f\"\\n[COMPLETE] Optimization already finished!\")\n",
        "        print(f\"[INFO] Loading final results...\")\n",
        "        \n",
        "        if os.path.exists(results_path):\n",
        "            with open(results_path, 'rb') as f:\n",
        "                res = pickle.load(f)\n",
        "            print(f\"[OK] Results loaded from: {results_path}\")\n",
        "        else:\n",
        "            print(f\"[ERROR] Results file not found!\")\n",
        "            res = None\n",
        "        \n",
        "        # Skip to visualization\n",
        "        run_optimization = False\n",
        "    else:\n",
        "        run_optimization = True\n",
        "        n_gen = remaining_generations\n",
        "        \n",
        "else:\n",
        "    print(\"\\n[STARTING] New optimization run...\")\n",
        "    \n",
        "    # Create fresh algorithm\n",
        "    algorithm = NSGA3(\n",
        "        ref_dirs=ref_dirs,\n",
        "        pop_size=5  # DEMO: 5 instead of 20\n",
        "    )\n",
        "    \n",
        "    total_generations = 3  # DEMO: 3, for production use 50\n",
        "    n_gen = total_generations\n",
        "    run_optimization = True\n",
        "\n",
        "# ============================================================================\n",
        "# CUSTOM CALLBACK FOR CHECKPOINTING\n",
        "# ============================================================================\n",
        "\n",
        "class CheckpointCallback:\n",
        "    \"\"\"Save checkpoint after each generation.\"\"\"\n",
        "    \n",
        "    def __init__(self, checkpoint_path, results_path, run_id):\n",
        "        self.checkpoint_path = checkpoint_path\n",
        "        self.results_path = results_path\n",
        "        self.run_id = run_id\n",
        "        self.generation_times = []\n",
        "        self.last_time = time.time()\n",
        "        \n",
        "    def __call__(self, algorithm):\n",
        "        # Track time\n",
        "        current_time = time.time()\n",
        "        gen_time = current_time - self.last_time\n",
        "        self.generation_times.append(gen_time)\n",
        "        self.last_time = current_time\n",
        "        \n",
        "        # Get current generation\n",
        "        n_gen = algorithm.n_gen\n",
        "        \n",
        "        # Save checkpoint\n",
        "        checkpoint_data = {\n",
        "            'algorithm': algorithm,\n",
        "            'generation': n_gen,\n",
        "            'n_eval': algorithm.evaluator.n_eval,\n",
        "            'timestamp': datetime.now().isoformat(),\n",
        "            'run_id': self.run_id,\n",
        "            'generation_times': self.generation_times,\n",
        "        }\n",
        "        \n",
        "        with open(self.checkpoint_path, 'wb') as f:\n",
        "            pickle.dump(checkpoint_data, f)\n",
        "        \n",
        "        # Print progress\n",
        "        avg_time = sum(self.generation_times) / len(self.generation_times)\n",
        "        print(f\"  Gen {n_gen}: {algorithm.evaluator.n_eval} evals | \"\n",
        "              f\"Time: {gen_time:.1f}s | Avg: {avg_time:.1f}s/gen | \"\n",
        "              f\"Checkpoint saved\")\n",
        "\n",
        "# ============================================================================\n",
        "# RUN OPTIMIZATION\n",
        "# ============================================================================\n",
        "\n",
        "if run_optimization:\n",
        "    print(f\"\\nStarting optimization...\\n\")\n",
        "    print(f\"  Generations to run: {n_gen}\")\n",
        "    print(f\"  Population size: {algorithm.pop_size}\")\n",
        "    print(f\"  Expected evaluations: {algorithm.pop_size * n_gen}\")\n",
        "    \n",
        "    if ENABLE_CHECKPOINTING:\n",
        "        callback = CheckpointCallback(checkpoint_path, results_path, RUN_ID)\n",
        "        print(f\"  Checkpointing: ENABLED (saves after each generation)\")\n",
        "    else:\n",
        "        callback = None\n",
        "        print(f\"  Checkpointing: DISABLED\")\n",
        "    \n",
        "    print(\"\\n\" + \"-\"*80 + \"\\n\")\n",
        "    \n",
        "    # Run optimization\n",
        "    res = minimize(\n",
        "        problem,\n",
        "        algorithm,\n",
        "        ('n_gen', n_gen),\n",
        "        callback=callback,\n",
        "        verbose=True,\n",
        "        save_history=False\n",
        "    )\n",
        "    \n",
        "    # Save final results\n",
        "    print(f\"\\n[OK] Optimization complete!\")\n",
        "    print(f\"  Total evaluations: {res.algorithm.n_eval}\")\n",
        "    print(f\"  Pareto solutions: {len(res.F)}\")\n",
        "    \n",
        "    # Save final results\n",
        "    with open(results_path, 'wb') as f:\n",
        "        pickle.dump(res, f)\n",
        "    print(f\"\\n[SAVED] Final results to: {results_path}\")\n",
        "    \n",
        "    # Clean up checkpoint (optimization finished)\n",
        "    if os.path.exists(checkpoint_path):\n",
        "        os.remove(checkpoint_path)\n",
        "        print(f\"[CLEANED] Removed checkpoint (optimization complete)\")\n",
        "\n",
        "# ============================================================================\n",
        "# SUMMARY\n",
        "# ============================================================================\n",
        "\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "if run_optimization:\n",
        "    print(\"OPTIMIZATION COMPLETED SUCCESSFULLY\")\n",
        "else:\n",
        "    print(\"OPTIMIZATION ALREADY COMPLETE\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "if res is not None:\n",
        "    print(f\"\\nResults:\")\n",
        "    print(f\"  Evaluations: {res.algorithm.n_eval}\")\n",
        "    print(f\"  Pareto solutions: {len(res.F)}\")\n",
        "    print(f\"  Results saved to: {os.path.basename(results_path)}\")\n",
        "    \n",
        "    print(f\"\\nTo resume or restart:\")\n",
        "    print(f\"  - Same run:  Keep RUN_ID = '{RUN_ID}' and RESUME_IF_AVAILABLE = True\")\n",
        "    print(f\"  - New run:   Change RUN_ID to a new value\")\n",
        "    print(f\"  - From scratch: Set RESUME_IF_AVAILABLE = False or delete checkpoint\")\n",
        "    \n",
        "print(\"=\"*80 + \"\\n\")"
    ]
}

# Create checkpoint management cell
checkpoint_management = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================================\n",
        "# CHECKPOINT MANAGEMENT UTILITIES\n",
        "# ============================================================================\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"CHECKPOINT MANAGEMENT\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# List all checkpoints\n",
        "checkpoint_dir = os.path.join(PROJECT_PATH, \"optimization_checkpoints\")\n",
        "\n",
        "if os.path.exists(checkpoint_dir):\n",
        "    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]\n",
        "    \n",
        "    if checkpoint_files:\n",
        "        print(f\"\\nFound {len(checkpoint_files)} checkpoint files:\\n\")\n",
        "        \n",
        "        for f in sorted(checkpoint_files):\n",
        "            filepath = os.path.join(checkpoint_dir, f)\n",
        "            size_mb = os.path.getsize(filepath) / 1e6\n",
        "            \n",
        "            # Try to load and show info\n",
        "            try:\n",
        "                with open(filepath, 'rb') as file:\n",
        "                    data = pickle.load(file)\n",
        "                \n",
        "                if 'generation' in data:  # It's a checkpoint\n",
        "                    print(f\"  CHECKPOINT: {f}\")\n",
        "                    print(f\"    Generation: {data['generation']}\")\n",
        "                    print(f\"    Evaluations: {data['n_eval']}\")\n",
        "                    print(f\"    Timestamp: {data.get('timestamp', 'N/A')}\")\n",
        "                    print(f\"    Size: {size_mb:.1f} MB\")\n",
        "                    \n",
        "                    # Show timing info if available\n",
        "                    if 'generation_times' in data and data['generation_times']:\n",
        "                        avg_time = sum(data['generation_times']) / len(data['generation_times'])\n",
        "                        print(f\"    Avg time/gen: {avg_time:.1f}s\")\n",
        "                        \n",
        "                elif hasattr(data, 'F'):  # It's results\n",
        "                    print(f\"  RESULTS: {f}\")\n",
        "                    print(f\"    Pareto solutions: {len(data.F)}\")\n",
        "                    print(f\"    Total evaluations: {data.algorithm.n_eval}\")\n",
        "                    print(f\"    Size: {size_mb:.1f} MB\")\n",
        "                print()\n",
        "                \n",
        "            except Exception as e:\n",
        "                print(f\"  {f} (Size: {size_mb:.1f} MB) - Error loading: {e}\\n\")\n",
        "    else:\n",
        "        print(\"\\nNo checkpoint files found.\")\n",
        "else:\n",
        "    print(\"\\nCheckpoint directory does not exist.\")\n",
        "\n",
        "print(\"=\"*80)\n",
        "print(\"\\nManagement Commands:\")\n",
        "print(\"  - To delete a checkpoint: Uncomment code below and set RUN_ID_TO_DELETE\")\n",
        "print(\"  - To start fresh: Set RESUME_IF_AVAILABLE = False in optimization cell\")\n",
        "print(\"  - To continue: Keep same RUN_ID and RESUME_IF_AVAILABLE = True\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "# ============================================================================\n",
        "# DELETE CHECKPOINT (USE WITH CAUTION)\n",
        "# ============================================================================\n",
        "\n",
        "# Uncomment to delete a specific checkpoint and start fresh\n",
        "# RUN_ID_TO_DELETE = \"demo_run_001\"\n",
        "# checkpoint_to_delete = os.path.join(checkpoint_dir, f\"nsga3_{RUN_ID_TO_DELETE}_checkpoint.pkl\")\n",
        "# if os.path.exists(checkpoint_to_delete):\n",
        "#     os.remove(checkpoint_to_delete)\n",
        "#     print(f\"\\nDeleted checkpoint: {checkpoint_to_delete}\")\n",
        "# else:\n",
        "#     print(f\"\\nCheckpoint not found: {checkpoint_to_delete}\")"
    ]
}

# Replace the NSGA-III code cell
cells[nsga3_code_idx] = nsga3_with_resume

# Insert checkpoint management cell after Section 11 code
# Find where to insert (after the visualization code of Section 12, before Section 13)
insert_idx = nsga3_code_idx + 1

# Skip over Section 12 cells to insert before Section 13
while insert_idx < len(cells):
    source = cells[insert_idx]['source']
    if isinstance(source, list):
        source = ''.join(source)
    if '## 13. Test INbreast' in source:
        break
    insert_idx += 1

# Insert checkpoint management cell
cells.insert(insert_idx, checkpoint_management)

# Update notebook
nb['cells'] = cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Added resume optimization capability")
print(f"[OK] Replaced NSGA-III cell with checkpoint-enabled version")
print(f"[OK] Added checkpoint management cell")
print(f"[OK] Notebook updated with {len(cells)} cells")
print(f"[OK] Backup: {backup_path}")

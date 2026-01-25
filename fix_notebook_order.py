"""
Reorganize breast_cancer_colab_demo.ipynb to have a proper logical flow.
Removes duplicates and arranges cells in the correct order.
"""

import json
from pathlib import Path
import shutil

# Read the notebook
notebook_path = Path("breast_cancer_colab_demo.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

def get_first_line(cell):
    """Get first non-empty line from cell source."""
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    for line in source.split('\n'):
        if line.strip():
            return line.strip()
    return ""

# Create backup
backup_path = notebook_path.with_suffix('.ipynb.backup')
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Organize cells into sections
organized = {
    'header': [],
    'section_1': [],
    'section_2': [],
    'section_3': [],
    'section_4': [],
    'section_5': [],
    'section_6': [],
    'section_7': [],
    'section_8': [],
    'section_9': [],
    'section_10': [],
    'section_11': [],
    'section_12': [],
    'section_13': [],
    'section_14': [],
    'section_15': [],
    'section_16': [],
    'section_17': [],
}

# Track which cells we've already added to prevent duplicates
processed_code_hashes = set()

def cell_hash(cell):
    """Create a simple hash of cell content for duplicate detection."""
    if cell['cell_type'] == 'code':
        source = cell['source']
        if isinstance(source, list):
            source = ''.join(source)
        # Only hash first 100 chars to catch duplicates
        return source.strip()[:100]
    return None

# Categorize cells by section
i = 0
while i < len(cells):
    cell = cells[i]
    first_line = get_first_line(cell)

    # Check for duplicates in code cells
    if cell['cell_type'] == 'code':
        h = cell_hash(cell)
        if h and h in processed_code_hashes:
            print(f"[SKIP] Duplicate code cell at index {i}: {first_line[:50]}")
            i += 1
            continue
        if h:
            processed_code_hashes.add(h)

    # Main header
    if '# Breast Cancer Detection - Google Colab Demo' in first_line:
        organized['header'].append(cell)

    # Section 1: Environment Setup
    elif '## 1. Environment Setup' in first_line:
        organized['section_1'].append(cell)
        # Collect following cells until next section
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_1'].append(next_cell)
            i += 1

    # Section 2: Mount Google Drive
    elif '## 2. Mount Google Drive' in first_line:
        organized['section_2'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_2'].append(next_cell)
            i += 1

    # Section 3: Preprocessing Configuration
    elif '## 3. Preprocessing Configuration' in first_line:
        organized['section_3'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_3'].append(next_cell)
            i += 1

    # Section 4: Test Preprocessing Pipeline
    elif '## 4. Test Preprocessing' in first_line:
        # Skip if we already have section 4
        if len(organized['section_4']) > 0:
            print(f"[SKIP] Duplicate Section 4 header at index {i}")
            i += 1
            continue
        organized['section_4'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_4'].append(next_cell)
            i += 1

    # Section 5: Test Dataset Loading
    elif '## 5. Test Dataset Loading' in first_line:
        organized['section_5'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_5'].append(next_cell)
            i += 1

    # Section 6: Test Model Building
    elif '## 6. Test Model Building' in first_line:
        organized['section_6'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_6'].append(next_cell)
            i += 1

    # Section 7: Test Augmentation
    elif '## 7. Test Augmentation' in first_line or '## 7. Test Training Pipeline' in first_line:
        # Take the Augmentation one, skip Training Pipeline (that's section 8)
        if 'Augmentation' in first_line:
            if len(organized['section_7']) > 0:
                print(f"[SKIP] Duplicate Section 7 header at index {i}")
                i += 1
                continue
            organized['section_7'].append(cell)
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                next_line = get_first_line(next_cell)
                if next_line.startswith('##'):
                    i -= 1
                    break
                h = cell_hash(next_cell)
                if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                    i += 1
                    continue
                if h:
                    processed_code_hashes.add(h)
                organized['section_7'].append(next_cell)
                i += 1
        else:
            # This is mislabeled section 8, skip it
            print(f"[SKIP] Mislabeled Section 7 (Training Pipeline) at index {i}")

    # Section 8: Test Training Pipeline
    elif '## 8. Test Training Pipeline' in first_line or '## 8. Test Breast-Level Aggregation' in first_line:
        # Take the Training Pipeline one
        if 'Training Pipeline' in first_line:
            if len(organized['section_8']) > 0:
                print(f"[SKIP] Duplicate Section 8 header at index {i}")
                i += 1
                continue
            organized['section_8'].append(cell)
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                next_line = get_first_line(next_cell)
                if next_line.startswith('##'):
                    i -= 1
                    break
                h = cell_hash(next_cell)
                if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                    i += 1
                    continue
                if h:
                    processed_code_hashes.add(h)
                organized['section_8'].append(next_cell)
                i += 1
        else:
            # Mislabeled, skip
            print(f"[SKIP] Mislabeled Section 8 (Breast-Level Aggregation) at index {i}")

    # Section 9: Test Breast-Level Aggregation
    elif '## 9. Test Breast-Level Aggregation' in first_line:
        organized['section_9'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_9'].append(next_cell)
            i += 1

    # Section 10: Test Robustness Evaluation
    elif '## 10. Test Robustness' in first_line or '## 10. NSGA-III' in first_line:
        # Take Robustness, skip NSGA-III (that's section 11)
        if 'Robustness' in first_line:
            if len(organized['section_10']) > 0:
                print(f"[SKIP] Duplicate Section 10 header at index {i}")
                i += 1
                continue
            organized['section_10'].append(cell)
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                next_line = get_first_line(next_cell)
                if next_line.startswith('##'):
                    i -= 1
                    break
                h = cell_hash(next_cell)
                if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                    i += 1
                    continue
                if h:
                    processed_code_hashes.add(h)
                organized['section_10'].append(next_cell)
                i += 1
        else:
            print(f"[SKIP] Mislabeled Section 10 (NSGA-III) at index {i}")

    # Section 11: NSGA-III Optimization
    elif '## 11. NSGA-III' in first_line:
        organized['section_11'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_11'].append(next_cell)
            i += 1

    # Section 12: Visualize Pareto Front
    elif '## 12. Visualize Pareto' in first_line:
        organized['section_12'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_12'].append(next_cell)
            i += 1

    # Section 13: Test INbreast Dataset Loading
    elif '## 13. Test INbreast' in first_line or '## 13. Summary' in first_line:
        # Take INbreast, skip Summary (that's later)
        if 'INbreast' in first_line:
            if len(organized['section_13']) > 0:
                print(f"[SKIP] Duplicate Section 13 header at index {i}")
                i += 1
                continue
            organized['section_13'].append(cell)
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                next_line = get_first_line(next_cell)
                if next_line.startswith('##'):
                    i -= 1
                    break
                h = cell_hash(next_cell)
                if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                    i += 1
                    continue
                if h:
                    processed_code_hashes.add(h)
                organized['section_13'].append(next_cell)
                i += 1
        else:
            print(f"[SKIP] Mislabeled Section 13 (Summary) at index {i}")

    # Section 14: Zero-Shot Evaluation on INbreast
    elif '## 14. Zero-Shot' in first_line or '## 14. Summary' in first_line:
        # Take Zero-Shot, skip Summary
        if 'Zero-Shot' in first_line:
            if len(organized['section_14']) > 0:
                print(f"[SKIP] Duplicate Section 14 header at index {i}")
                i += 1
                continue
            # Skip duplicate section 14 summary
            organized['section_14'].append(cell)
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                next_line = get_first_line(next_cell)
                if next_line.startswith('##'):
                    i -= 1
                    break
                h = cell_hash(next_cell)
                if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                    i += 1
                    continue
                if h:
                    processed_code_hashes.add(h)
                organized['section_14'].append(next_cell)
                i += 1
        else:
            print(f"[SKIP] Mislabeled Section 14 (Summary) at index {i}")

    # Section 15: Entropy-Based Domain Adaptation
    elif '## 15. Zero-Shot' in first_line or '## 15. Entropy' in first_line:
        # Take Entropy, skip duplicate Zero-Shot
        if 'Entropy' in first_line:
            if len(organized['section_15']) > 0:
                print(f"[SKIP] Duplicate Section 15 header at index {i}")
                i += 1
                continue
            organized['section_15'].append(cell)
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                next_line = get_first_line(next_cell)
                if next_line.startswith('##'):
                    i -= 1
                    break
                h = cell_hash(next_cell)
                if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                    i += 1
                    continue
                if h:
                    processed_code_hashes.add(h)
                organized['section_15'].append(next_cell)
                i += 1
        else:
            print(f"[SKIP] Duplicate Section 15 (Zero-Shot) at index {i}")

    # Section 16: Entropy Statistics & Comparative Analysis
    elif '## 16. Entropy Statistics' in first_line or '## 16. Summary' in first_line:
        # Take Entropy Statistics, skip Summary
        if 'Entropy Statistics' in first_line:
            if len(organized['section_16']) > 0:
                print(f"[SKIP] Duplicate Section 16 header at index {i}")
                i += 1
                continue
            organized['section_16'].append(cell)
            i += 1
            while i < len(cells):
                next_cell = cells[i]
                next_line = get_first_line(next_cell)
                if next_line.startswith('##'):
                    i -= 1
                    break
                h = cell_hash(next_cell)
                if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                    i += 1
                    continue
                if h:
                    processed_code_hashes.add(h)
                organized['section_16'].append(next_cell)
                i += 1
        else:
            print(f"[SKIP] Mislabeled Section 16 (Summary) at index {i}")

    # Section 17: Final Summary
    elif '## 17. Final Summary' in first_line:
        organized['section_17'].append(cell)
        i += 1
        while i < len(cells):
            next_cell = cells[i]
            next_line = get_first_line(next_cell)
            if next_line.startswith('##'):
                i -= 1
                break
            h = cell_hash(next_cell)
            if next_cell['cell_type'] == 'code' and h and h in processed_code_hashes:
                i += 1
                continue
            if h:
                processed_code_hashes.add(h)
            organized['section_17'].append(next_cell)
            i += 1

    i += 1

# Build new cell list in correct order
new_cells = []
new_cells.extend(organized['header'])
new_cells.extend(organized['section_1'])
new_cells.extend(organized['section_2'])
new_cells.extend(organized['section_3'])
new_cells.extend(organized['section_4'])
new_cells.extend(organized['section_5'])
new_cells.extend(organized['section_6'])
new_cells.extend(organized['section_7'])
new_cells.extend(organized['section_8'])
new_cells.extend(organized['section_9'])
new_cells.extend(organized['section_10'])
new_cells.extend(organized['section_11'])
new_cells.extend(organized['section_12'])
new_cells.extend(organized['section_13'])
new_cells.extend(organized['section_14'])
new_cells.extend(organized['section_15'])
new_cells.extend(organized['section_16'])
new_cells.extend(organized['section_17'])

# Print summary
print("\n" + "="*80)
print("REORGANIZATION SUMMARY")
print("="*80)
for section_name, section_cells in organized.items():
    print(f"{section_name:15s}: {len(section_cells):2d} cells")

print(f"\nOriginal cells: {len(cells)}")
print(f"New cells:      {len(new_cells)}")
print(f"Removed:        {len(cells) - len(new_cells)} (duplicates)")

# Update notebook
nb['cells'] = new_cells

# Save
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n[OK] Notebook reorganized and saved")
print(f"[OK] Backup available at: {backup_path}")

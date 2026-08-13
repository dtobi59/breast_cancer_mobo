"""
Generate Methodology Diagram for Multi-Objective Breast Cancer Detection

Requirements:
    pip install matplotlib graphviz pillow

Usage:
    python generate_methodology_diagram.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

def create_methodology_diagram():
    """Create comprehensive methodology flowchart."""

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 28)
    ax.axis('off')

    # Color scheme
    colors = {
        'data': '#E3F2FD',
        'data_edge': '#1976D2',
        'process': '#FFF3E0',
        'process_edge': '#F57C00',
        'optimization': '#F3E5F5',
        'optimization_edge': '#7B1FA2',
        'evaluation': '#E8F5E9',
        'evaluation_edge': '#388E3C',
        'output': '#FCE4EC',
        'output_edge': '#C2185B'
    }

    # Helper function to draw box
    def draw_box(x, y, w, h, text, box_color, edge_color, fontsize=10):
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=box_color,
                            edgecolor=edge_color,
                            linewidth=2.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text,
               ha='center', va='center',
               fontsize=fontsize, fontweight='bold',
               multialignment='center')

    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2, color='black', style='->', lw=2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                               arrowstyle=style,
                               color=color,
                               linewidth=lw,
                               mutation_scale=20)
        ax.add_patch(arrow)

    # Title
    ax.text(5, 27.5, 'Methodology: Multi-Objective Optimization for\nBreast Cancer Detection under Dataset Shift',
           ha='center', va='top', fontsize=16, fontweight='bold')

    # ============================================================
    # PHASE 1: DATA PREPARATION
    # ============================================================
    y_start = 25

    # Phase label
    ax.text(5, y_start, 'PHASE 1: DATA PREPARATION',
           ha='center', va='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2))

    # Input datasets
    draw_box(0.5, y_start-2, 1.8, 1.5, 'VinDr-Mammo\nVietnamese\n~15,000 images',
            colors['data'], colors['data_edge'], 9)
    draw_box(7.7, y_start-2, 1.8, 1.5, 'INbreast\nPortuguese\n~410 images',
            colors['data'], colors['data_edge'], 9)

    # Preprocessing
    draw_box(3.5, y_start-2, 3, 1.5, 'DICOM Preprocessing\n720×480 Magma RGB\nBreast extraction',
            colors['process'], colors['process_edge'], 9)

    # Arrows to preprocessing
    draw_arrow(2.3, y_start-1.25, 3.5, y_start-1.25, colors['data_edge'])
    draw_arrow(7.7, y_start-1.25, 6.5, y_start-1.25, colors['data_edge'])

    # BI-RADS filtering
    draw_box(3.5, y_start-4, 3, 1.2, 'BI-RADS Filtering\nBenign: 1-3 | Malignant: 5-6',
            colors['process'], colors['process_edge'], 9)
    draw_arrow(5, y_start-2, 5, y_start-4, colors['process_edge'])

    # Splits
    draw_box(0.5, y_start-6, 2, 1.5, 'VinDr Split\nTrain 80%\nVal 20%',
            colors['data'], colors['data_edge'], 9)
    draw_box(7.5, y_start-6, 2, 1.5, 'INbreast Split\nCalib 20%\nTest 80%',
            colors['data'], colors['data_edge'], 9)

    draw_arrow(4.5, y_start-4.8, 2.5, y_start-5.25, colors['data_edge'])
    draw_arrow(5.5, y_start-4.8, 7.5, y_start-5.25, colors['data_edge'])

    # ============================================================
    # PHASE 2: MULTI-OBJECTIVE OPTIMIZATION
    # ============================================================
    y_opt = y_start - 8

    ax.text(5, y_opt, 'PHASE 2: MULTI-OBJECTIVE OPTIMIZATION',
           ha='center', va='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=2))

    # Hyperparameter space
    draw_box(0.5, y_opt-2.2, 2.5, 1.8, '5D Search Space\nlr, wd, dropout\naug_str, unfreeze',
            colors['optimization'], colors['optimization_edge'], 8.5)

    # Model training
    draw_box(3.5, y_opt-2.2, 3, 1.8, 'ResNet152 Training\nImageNet pretrained\nAdamW optimizer\nEarly stopping',
            colors['process'], colors['process_edge'], 8.5)

    # 4 Objectives
    draw_box(7, y_opt-2.2, 2.5, 1.8, '4 Objectives\n-PR-AUC\n-AUROC\nBrier\nDegradation',
            colors['optimization'], colors['optimization_edge'], 8.5)

    # Arrows
    draw_arrow(1.5, y_start-6, 1.5, y_opt-2.2, colors['data_edge'])
    draw_arrow(3, y_opt-1.3, 3.5, y_opt-1.3, colors['optimization_edge'])
    draw_arrow(6.5, y_opt-1.3, 7, y_opt-1.3, colors['optimization_edge'])

    # BoTorch qNEHVI
    draw_box(2, y_opt-4.5, 6, 1.8, 'BoTorch qNEHVI\n40 evaluations: 16 initial + 6 iterations × 4 batch\nGaussian Processes + Acquisition Function',
            colors['optimization'], colors['optimization_edge'], 9)

    draw_arrow(5, y_opt-2.2, 5, y_opt-4.5, colors['optimization_edge'])
    draw_arrow(8.5, y_start-6, 8.5, y_opt-1.3, colors['data_edge'], style='->', lw=1.5)  # INbreast calib for Obj 4

    # Pareto Front
    draw_box(2, y_opt-6.8, 6, 1.5, 'Pareto Front\nNon-dominated solutions\n~20-30 solutions',
            colors['optimization'], colors['optimization_edge'], 9)

    draw_arrow(5, y_opt-4.5, 5, y_opt-6.8, colors['optimization_edge'])

    # ============================================================
    # PHASE 3: SOLUTION SELECTION
    # ============================================================
    y_sel = y_opt - 9

    ax.text(5, y_sel, 'PHASE 3: SOLUTION SELECTION',
           ha='center', va='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2))

    draw_box(1.5, y_sel-2, 7, 1.5, 'Select 4 Key Solutions from Pareto Front\nMax AUROC | Min Brier | Balanced Knee | Max PR-AUC',
            colors['optimization'], colors['optimization_edge'], 9)

    draw_arrow(5, y_opt-6.8, 5, y_sel-2, colors['optimization_edge'])

    # ============================================================
    # PHASE 4: ZERO-SHOT EVALUATION
    # ============================================================
    y_eval = y_sel - 4

    ax.text(5, y_eval, 'PHASE 4: ZERO-SHOT EVALUATION',
           ha='center', va='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))

    # Train 4 models
    draw_box(3.5, y_eval-2, 3, 1.3, 'Train 4 Models\nwith selected hyperparameters',
            colors['process'], colors['process_edge'], 9)

    draw_arrow(5, y_sel-2, 5, y_eval-2, colors['optimization_edge'])

    # Evaluation boxes
    draw_box(0.5, y_eval-4, 4, 1.5, 'Evaluate on VinDr Val\nBreast-level Noisy-OR\nPR-AUC, AUROC, Brier\nSelect threshold',
            colors['evaluation'], colors['evaluation_edge'], 9)

    draw_box(5.5, y_eval-4, 4, 1.5, 'Zero-Shot on INbreast Test\nNo retraining\nNoisy-OR aggregation\nTransfer threshold',
            colors['evaluation'], colors['evaluation_edge'], 9)

    draw_arrow(5, y_eval-2, 2.5, y_eval-4, colors['evaluation_edge'])
    draw_arrow(5, y_eval-2, 7.5, y_eval-4, colors['evaluation_edge'])

    # ============================================================
    # PHASE 5: DOMAIN SHIFT ANALYSIS
    # ============================================================
    y_analysis = y_eval - 6

    ax.text(5, y_analysis, 'PHASE 5: DOMAIN SHIFT ANALYSIS',
           ha='center', va='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9C4', edgecolor='#F9A825', linewidth=2))

    draw_box(2, y_analysis-2, 6, 1.5, 'Compare 4 Solutions\nPR-AUC drop | AUROC drop | Brier increase\nGeneralization vs Source Performance',
            colors['evaluation'], colors['evaluation_edge'], 9)

    draw_arrow(2.5, y_eval-4, 5, y_analysis-2, colors['evaluation_edge'])
    draw_arrow(7.5, y_eval-4, 5, y_analysis-2, colors['evaluation_edge'])

    # ============================================================
    # PHASE 6: OUTPUTS
    # ============================================================
    y_out = y_analysis - 4

    ax.text(5, y_out, 'PHASE 6: REPORTING',
           ha='center', va='center', fontsize=13, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FCE4EC', edgecolor='#C2185B', linewidth=2))

    draw_box(1, y_out-2, 3.5, 1.5, 'Visualizations\nDistribution charts\nMulti-solution comparison\nDomain shift plots',
            colors['output'], colors['output_edge'], 8.5)

    draw_box(5.5, y_out-2, 3.5, 1.5, 'Deliverables\nCSV tables\nModel checkpoints\nPNG figures (300 DPI)',
            colors['output'], colors['output_edge'], 8.5)

    draw_arrow(5, y_analysis-2, 2.75, y_out-2, colors['output_edge'])
    draw_arrow(5, y_analysis-2, 7.25, y_out-2, colors['output_edge'])

    # Legend
    legend_y = 0.5
    ax.text(5, legend_y+0.5, 'Legend',
           ha='center', va='center', fontsize=11, fontweight='bold')

    legend_items = [
        mpatches.Patch(facecolor=colors['data'], edgecolor=colors['data_edge'], label='Data'),
        mpatches.Patch(facecolor=colors['process'], edgecolor=colors['process_edge'], label='Process'),
        mpatches.Patch(facecolor=colors['optimization'], edgecolor=colors['optimization_edge'], label='Optimization'),
        mpatches.Patch(facecolor=colors['evaluation'], edgecolor=colors['evaluation_edge'], label='Evaluation'),
        mpatches.Patch(facecolor=colors['output'], edgecolor=colors['output_edge'], label='Output')
    ]

    ax.legend(handles=legend_items, loc='lower center', ncol=5, fontsize=9,
             bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()

    # Save
    plt.savefig('methodology_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("✓ Saved: methodology_diagram.png (300 DPI)")

    plt.savefig('methodology_diagram.pdf', bbox_inches='tight', facecolor='white')
    print("✓ Saved: methodology_diagram.pdf (vector)")

    plt.show()

if __name__ == '__main__':
    create_methodology_diagram()

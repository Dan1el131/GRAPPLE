"""
Generate KDD-quality visualization for embedding distortion comparison
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Data from Table 4
datasets = ['Cora', 'CiteSeer']
euclidean_ad = [1.0129, 1.0157]
euclidean_map = [0.7026, 0.6363]
hyperbolic_ad = [0.3284, 0.1335]
hyperbolic_map = [0.0095, 0.0077]

# Calculate improvements
ad_improvement = [(e - h) / e * 100 for e, h in zip(euclidean_ad, hyperbolic_ad)]
map_improvement = [(e - h) / e * 100 for e, h in zip(euclidean_map, hyperbolic_map)]

# Color scheme - professional and print-friendly
color_euclidean = '#3498db'  # Blue
color_hyperbolic = '#e74c3c'  # Red
color_improvement = '#2ecc71'  # Green

# Create figure with multiple subplots
fig = plt.figure(figsize=(16, 5))

# ========== Subplot 1: AD Comparison (Bar Chart) ==========
ax1 = plt.subplot(1, 3, 1)

x = np.arange(len(datasets))
width = 0.35

bars1 = ax1.bar(x - width/2, euclidean_ad, width, label='Euclidean', 
                color=color_euclidean, edgecolor='black', linewidth=1.2, alpha=0.85)
bars2 = ax1.bar(x + width/2, hyperbolic_ad, width, label='Hyperbolic', 
                color=color_hyperbolic, edgecolor='black', linewidth=1.2, alpha=0.85)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Average Distortion (AD)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Datasets', fontsize=12, fontweight='bold')
ax1.set_title('(a) Average Distortion Comparison', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=11)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim(0, max(euclidean_ad) * 1.15)

# ========== Subplot 2: MAP Comparison (Bar Chart) ==========
ax2 = plt.subplot(1, 3, 2)

bars3 = ax2.bar(x - width/2, euclidean_map, width, label='Euclidean', 
                color=color_euclidean, edgecolor='black', linewidth=1.2, alpha=0.85)
bars4 = ax2.bar(x + width/2, hyperbolic_map, width, label='Hyperbolic', 
                color=color_hyperbolic, edgecolor='black', linewidth=1.2, alpha=0.85)

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Mean Average Precision (MAP)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Datasets', fontsize=12, fontweight='bold')
ax2.set_title('(b) MAP@10 Comparison', fontsize=13, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=11)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(euclidean_map) * 1.15)

# ========== Subplot 3: Improvement Percentage (Grouped Bar) ==========
ax3 = plt.subplot(1, 3, 3)

x3 = np.arange(len(datasets))
width3 = 0.35

bars5 = ax3.bar(x3 - width3/2, ad_improvement, width3, label='AD Reduction (%)', 
                color='#9b59b6', edgecolor='black', linewidth=1.2, alpha=0.85)
bars6 = ax3.bar(x3 + width3/2, map_improvement, width3, label='MAP Reduction (%)', 
                color='#f39c12', edgecolor='black', linewidth=1.2, alpha=0.85)

# Add value labels
for bars in [bars5, bars6]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Datasets', fontsize=12, fontweight='bold')
ax3.set_title('(c) Hyperbolic vs Euclidean Improvement', fontsize=13, fontweight='bold', pad=15)
ax3.set_xticks(x3)
ax3.set_xticklabels(datasets, fontsize=11)
ax3.legend(loc='upper right', fontsize=10, framealpha=0.95, edgecolor='black')
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim(0, max(max(ad_improvement), max(map_improvement)) * 1.15)

plt.tight_layout()
plt.savefig('distortion_comparison_kdd.pdf', dpi=300, bbox_inches='tight')
plt.savefig('distortion_comparison_kdd.png', dpi=300, bbox_inches='tight')
print("Saved: distortion_comparison_kdd.pdf")
print("Saved: distortion_comparison_kdd.png")
plt.show()


# ========== Alternative: Combined visualization with lines ==========
fig2, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: AD Comparison with lines
ax_left = axes[0]
x_pos = np.arange(len(datasets))

# Plot lines
ax_left.plot(x_pos, euclidean_ad, marker='o', markersize=10, linewidth=2.5, 
            label='Euclidean', color=color_euclidean, markeredgecolor='black', 
            markeredgewidth=1.5)
ax_left.plot(x_pos, hyperbolic_ad, marker='s', markersize=10, linewidth=2.5, 
            label='Hyperbolic', color=color_hyperbolic, markeredgecolor='black', 
            markeredgewidth=1.5)

# Add value annotations
for i, (e, h) in enumerate(zip(euclidean_ad, hyperbolic_ad)):
    ax_left.annotate(f'{e:.4f}', (i, e), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    ax_left.annotate(f'{h:.4f}', (i, h), textcoords="offset points", 
                    xytext=(0, -15), ha='center', fontsize=10, fontweight='bold')

ax_left.set_ylabel('Average Distortion (AD)', fontsize=12, fontweight='bold')
ax_left.set_xlabel('Datasets', fontsize=12, fontweight='bold')
ax_left.set_title('(a) Average Distortion', fontsize=13, fontweight='bold', pad=15)
ax_left.set_xticks(x_pos)
ax_left.set_xticklabels(datasets, fontsize=11)
ax_left.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')
ax_left.grid(True, alpha=0.3, linestyle='--')
ax_left.set_ylim(0, max(euclidean_ad) * 1.2)

# Right: MAP Comparison with lines
ax_right = axes[1]

ax_right.plot(x_pos, euclidean_map, marker='o', markersize=10, linewidth=2.5, 
             label='Euclidean', color=color_euclidean, markeredgecolor='black', 
             markeredgewidth=1.5)
ax_right.plot(x_pos, hyperbolic_map, marker='s', markersize=10, linewidth=2.5, 
             label='Hyperbolic', color=color_hyperbolic, markeredgecolor='black', 
             markeredgewidth=1.5)

# Add value annotations
for i, (e, h) in enumerate(zip(euclidean_map, hyperbolic_map)):
    ax_right.annotate(f'{e:.4f}', (i, e), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    ax_right.annotate(f'{h:.4f}', (i, h), textcoords="offset points", 
                     xytext=(0, -15), ha='center', fontsize=10, fontweight='bold')

ax_right.set_ylabel('Mean Average Precision (MAP)', fontsize=12, fontweight='bold')
ax_right.set_xlabel('Datasets', fontsize=12, fontweight='bold')
ax_right.set_title('(b) Mean Average Precision', fontsize=13, fontweight='bold', pad=15)
ax_right.set_xticks(x_pos)
ax_right.set_xticklabels(datasets, fontsize=11)
ax_right.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')
ax_right.grid(True, alpha=0.3, linestyle='--')
ax_right.set_ylim(0, max(euclidean_map) * 1.2)

plt.tight_layout()
plt.savefig('distortion_comparison_lines_kdd.pdf', dpi=300, bbox_inches='tight')
plt.savefig('distortion_comparison_lines_kdd.png', dpi=300, bbox_inches='tight')
print("\nSaved: distortion_comparison_lines_kdd.pdf")
print("Saved: distortion_comparison_lines_kdd.png")
plt.show()

print("\n" + "="*60)
print("Statistics Summary:")
print("="*60)
for i, dataset in enumerate(datasets):
    print(f"\n{dataset}:")
    print(f"  AD:  Euclidean={euclidean_ad[i]:.4f}, Hyperbolic={hyperbolic_ad[i]:.4f}")
    print(f"       → Improvement: {ad_improvement[i]:.1f}%")
    print(f"  MAP: Euclidean={euclidean_map[i]:.4f}, Hyperbolic={hyperbolic_map[i]:.4f}")
    print(f"       → Change: {map_improvement[i]:.1f}%")
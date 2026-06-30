"""
Rank Collapse 可视化增强工具
生成论文级别的多面板对比图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def plot_comprehensive_analysis(results_dict, metrics_dict, dataset_name, save_path='rank_analysis.pdf'):
    """
    生成综合分析图（4个子图）
    
    1. 奇异值频谱图
    2. 累积能量图
    3. 秩指标对比柱状图
    4. 条件数对比
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 配色方案
    colors = {
        'GRAPPLE': '#2E86AB',
        'GRAPPLE w/o Scat': '#A23B72',
        'Collapse Baseline': '#F18F01'
    }
    
    # ============ 子图1: 奇异值频谱 ============
    ax1 = fig.add_subplot(gs[0, 0])
    
    for variant_name, singular_vals in results_dict.items():
        log_vals = np.log10(singular_vals + 1e-10)
        n_plot = min(50, len(log_vals))
        indices = np.arange(1, n_plot + 1)
        
        ax1.plot(indices, log_vals[:n_plot], 
                 label=variant_name, color=colors[variant_name],
                 linewidth=2.5, marker='o', markersize=3, markevery=5, alpha=0.8)
    
    ax1.set_xlabel('Singular Value Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('log₁₀(Singular Value)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Singular Value Spectrum', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(fontsize=10, frameon=True, shadow=True, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # ============ 子图2: 累积能量 ============
    ax2 = fig.add_subplot(gs[0, 1])
    
    for variant_name, singular_vals in results_dict.items():
        # 计算累积能量
        energy = singular_vals ** 2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        
        n_plot = min(50, len(cumulative_energy))
        indices = np.arange(1, n_plot + 1)
        
        ax2.plot(indices, cumulative_energy[:n_plot], 
                 label=variant_name, color=colors[variant_name],
                 linewidth=2.5, marker='s', markersize=3, markevery=5, alpha=0.8)
    
    ax2.axhline(y=0.9, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='90% threshold')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Energy Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Cumulative Energy', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(fontsize=10, frameon=True, shadow=True, loc='lower right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim([0, 1.05])
    
    # ============ 子图3: 秩指标对比 ============
    ax3 = fig.add_subplot(gs[1, 0])
    
    metrics_to_plot = ['effective_rank', 'stable_rank', 'numerical_rank']
    metric_labels = ['Effective\nRank', 'Stable\nRank', 'Numerical\nRank']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    variant_list = ['GRAPPLE', 'GRAPPLE w/o Scat', 'Collapse Baseline']
    
    for i, variant in enumerate(variant_list):
        values = [metrics_dict[variant][m] for m in metrics_to_plot]
        ax3.bar(x + i * width, values, width, 
                label=variant, color=colors[variant], alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax3.set_xlabel('Rank Metrics', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Rank Metrics Comparison', fontsize=13, fontweight='bold', loc='left')
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(metric_labels, fontsize=10)
    ax3.legend(fontsize=9, frameon=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # ============ 子图4: 条件数 & 能量集中度 ============
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 双Y轴图
    ax4_twin = ax4.twinx()
    
    # 条件数（左Y轴）
    condition_numbers = [metrics_dict[v]['condition_number'] for v in variant_list]
    x_pos = np.arange(len(variant_list))
    
    bars1 = ax4.bar(x_pos - 0.2, condition_numbers, 0.4, 
                    label='Condition Number', color='#E63946', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # 能量集中度（右Y轴）
    energy_ratios = [metrics_dict[v]['energy_ratio'] for v in variant_list]
    bars2 = ax4_twin.bar(x_pos + 0.2, energy_ratios, 0.4, 
                         label='Energy Ratio (Top 10%)', color='#06A77D', alpha=0.7, edgecolor='black', linewidth=1.2)
    
    ax4.set_xlabel('Model Variant', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Condition Number (log scale)', fontsize=11, fontweight='bold', color='#E63946')
    ax4_twin.set_ylabel('Energy Ratio', fontsize=11, fontweight='bold', color='#06A77D')
    
    ax4.set_title('(d) Condition Number & Energy Concentration', fontsize=13, fontweight='bold', loc='left')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['GRAPPLE', 'w/o Scat', 'Collapse'], fontsize=10, rotation=15, ha='right')
    
    ax4.set_yscale('log')
    ax4.tick_params(axis='y', labelcolor='#E63946')
    ax4_twin.tick_params(axis='y', labelcolor='#06A77D')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 图例
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left', frameon=True, shadow=True)
    
    # 总标题
    fig.suptitle(f'Rank Collapse Analysis - {dataset_name}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comprehensive analysis saved to: {save_path}")
    plt.close()


def plot_simple_spectrum(results_dict, dataset_name, save_path='spectrum_simple.pdf'):
    """
    生成简洁版频谱图（单图，用于论文主图）
    """
    plt.figure(figsize=(8, 5))
    
    colors = {
        'GRAPPLE': '#2E86AB',
        'GRAPPLE w/o Scat': '#A23B72',
        'Collapse Baseline': '#F18F01'
    }
    
    markers = {
        'GRAPPLE': 'o',
        'GRAPPLE w/o Scat': 's',
        'Collapse Baseline': '^'
    }
    
    for variant_name, singular_vals in results_dict.items():
        log_vals = np.log10(singular_vals + 1e-10)
        n_plot = min(50, len(log_vals))
        indices = np.arange(1, n_plot + 1)
        
        plt.plot(indices, log_vals[:n_plot], 
                 label=variant_name, 
                 color=colors[variant_name],
                 marker=markers[variant_name],
                 markersize=5,
                 markevery=5,
                 linewidth=3,
                 alpha=0.85)
    
    plt.xlabel('Singular Value Index', fontsize=14, fontweight='bold')
    plt.ylabel('log₁₀(Singular Value)', fontsize=14, fontweight='bold')
    plt.title(f'{dataset_name}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, shadow=True, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Simple spectrum saved to: {save_path}")
    plt.close()


def create_latex_table(metrics_dict, dataset_name, save_path='rank_table.tex'):
    """生成 LaTeX 表格代码"""
    
    latex_code = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Rank Collapse Metrics on {dataset_name}}}
\\label{{tab:rank_collapse_{dataset_name.lower()}}}
\\begin{{tabular}}{{lccc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{GRAPPLE}} & \\textbf{{w/o Scat}} & \\textbf{{Collapse}} \\\\
\\midrule
"""
    
    metrics_order = [
        ('effective_rank', 'Effective Rank', '{:.2f}'),
        ('stable_rank', 'Stable Rank', '{:.2f}'),
        ('numerical_rank', 'Numerical Rank', '{:.0f}'),
        ('energy_ratio', 'Energy Ratio (Top-10\\%)', '{:.4f}'),
        ('condition_number', 'Condition Number', '{:.2e}'),
    ]
    
    for key, name, fmt in metrics_order:
        val1 = metrics_dict['GRAPPLE'][key]
        val2 = metrics_dict['GRAPPLE w/o Scat'][key]
        val3 = metrics_dict['Collapse Baseline'][key]
        
        # 找出最好的值（根据指标类型）
        if key == 'energy_ratio' or key == 'condition_number':
            best_idx = np.argmin([val1, val2, val3])  # 越小越好
        else:
            best_idx = np.argmax([val1, val2, val3])  # 越大越好
        
        values = [val1, val2, val3]
        formatted = []
        for i, v in enumerate(values):
            s = fmt.format(v)
            if i == best_idx:
                s = f"\\textbf{{{s}}}"  # 加粗最好的值
            formatted.append(s)
        
        latex_code += f"{name} & {formatted[0]} & {formatted[1]} & {formatted[2]} \\\\\n"
    
    latex_code += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex_code)
    
    print(f"✓ LaTeX table saved to: {save_path}")
    return latex_code


# 示例使用（添加到主实验脚本的末尾）
if __name__ == '__main__':
    print("This is a visualization utility module.")
    print("Import and use in the main experiment script.")
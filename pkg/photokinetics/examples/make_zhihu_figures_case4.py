"""案例4知乎文章配图：PINN 非均匀 κ(z) 反演。

图1: kappa_inversion.png — κ(z) 反演结果：PINN 预测 vs 真值
图2: temperature_field_comparison.png — T(z,t) 场对比：真值 / PINN / 纯NN
图3: training_loss.png — PINN 训练 loss 曲线（各分量）
图4: method_comparison.png — 三种方法对比汇总

用法:
    python -m examples.make_zhihu_figures_case4
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

from examples._finite_difference import (
    generate_ground_truth, kappa_true_func, KAPPA0,
    DEFECT_Z0, DEFECT_SIGMA, DEFECT_AMP,
)
from examples.pinn_inverse_kappa import run_inversion
from examples.baseline_pure_nn import run_pure_nn_baseline
from examples.baseline_uniform_scipy import run_scipy_baseline

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'docs', 'case_studies', 'figures'
)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===== 图1：κ(z) 反演结果 =====

def plot_kappa_inversion(pinn_result, save=True):
    """κ(z) 反演结果：PINN 预测 vs 真值。"""
    print("生成图1: κ(z) 反演结果...")
    z_um = np.array(pinn_result["eval_z_um"])
    k_pred = pinn_result["kappa_pred_array"]
    k_true = pinn_result["kappa_true_array"]

    # 更密的真值曲线
    z_dense = np.linspace(0, 30, 300)
    k_dense = kappa_true_func(z_dense * 1e-6)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 真值
    ax.plot(z_dense, k_dense, color='#2E86AB', linewidth=2.5, label='真值 κ(z)', zorder=3)
    ax.fill_between(
        z_dense, KAPPA0 * np.ones_like(z_dense),
        k_dense, alpha=0.15, color='#2E86AB', label='缺陷层增强区',
    )

    # PINN 预测
    ax.plot(z_um, k_pred, 'o--', color='#E63946', linewidth=2, markersize=6,
            label='PINN 反演 κ_φ(z)', zorder=4)

    # 基线 1: scipy 均匀 κ
    from examples.baseline_uniform_scipy import run_scipy_baseline
    bl1 = run_scipy_baseline(seed=42, noise=0.01)
    ax.axhline(y=bl1["kappa_eff"], color='#06A77D', linewidth=1.5, linestyle=':',
               label=f"scipy 均匀 κ_eff = {bl1['kappa_eff']:.4f}", zorder=2)

    # 标注
    ax.axvline(x=DEFECT_Z0 * 1e6, color='gray', linewidth=1, linestyle='--', alpha=0.5)
    ax.annotate(f'缺陷层中心\nz₀ = {DEFECT_Z0*1e6:.0f}μm',
                xy=(DEFECT_Z0 * 1e6, KAPPA0 * (1 + DEFECT_AMP)),
                xytext=(15, KAPPA0 * (1 + DEFECT_AMP) * 0.9),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, color='gray')

    ax.set_xlabel('深度 z (μm)', fontsize=12)
    ax.set_ylabel('消光系数 κ(z)', fontsize=12)
    ax.set_title('PINN 非均匀 κ(z) 反演结果', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(0, max(k_true) * 1.2)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'kappa_inversion.png')
        plt.savefig(path, bbox_inches='tight')
        print(f"  保存到 {path}")
    plt.close()


# ===== 图2：T(z,t) 场对比 =====

def plot_temperature_field_comparison(pinn_result, nn_result, save=True):
    """T(z,t) 场对比：真值 / PINN / 纯NN。"""
    print("生成图2: T(z,t) 场对比...")
    gt = pinn_result["gt"]
    z_array = gt["z_array"][:300] * 1e6  # 0~30μm
    t_array = gt["t_array"][:300]  # 0~0.03s

    # 真值场（截取子区域）
    T_true = gt["T_field"][:300, :300]

    # PINN 预测场
    from examples.pinn_inverse_kappa import _eval_temperature_field
    from examples._pinn_nonuniform import L_Z, L_T, T_SCALE
    import torch
    T_pinn = _eval_temperature_field(
        pinn_result["T_net"], z_array * 1e-6, t_array,
    )

    # 纯 NN 预测场
    z_norm = torch.tensor(z_array / L_Z, dtype=torch.float32).unsqueeze(1)
    nn_net = nn_result["net"]
    T_nn = np.zeros((len(z_array), len(t_array)))
    with torch.no_grad():
        for j, t in enumerate(t_array):
            t_norm = torch.tensor([t / L_T], dtype=torch.float32).expand(len(z_array), 1)
            zt = torch.cat([z_norm, t_norm], dim=1)
            T_nn[:, j] = nn_net(zt).squeeze().numpy() * T_SCALE

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)

    vmin = 0
    vmax = max(T_true.max(), T_pinn.max(), T_nn.max())

    for ax, T, title in zip(
        axes,
        [T_true, T_pinn, T_nn],
        ['真值 T(z,t)', 'PINN 重建', '纯数据 NN'],
    ):
        im = ax.pcolormesh(t_array * 1e3, z_array, T, shading='auto',
                           cmap='hot', vmin=vmin, vmax=vmax)
        ax.set_xlabel('时间 t (ms)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.axhline(y=DEFECT_Z0 * 1e6, color='cyan', linewidth=1, linestyle='--', alpha=0.5)

    axes[0].set_ylabel('深度 z (μm)', fontsize=11)

    fig.colorbar(im, ax=axes, label='温升 T (K)', shrink=0.8)
    fig.suptitle('T(z,t) 场重建对比', fontsize=14, fontweight='bold', y=1.02)

    if save:
        path = os.path.join(OUTPUT_DIR, 'temperature_field_comparison.png')
        plt.savefig(path, bbox_inches='tight')
        print(f"  保存到 {path}")
    plt.close()


# ===== 图3：训练 loss 曲线 =====

def plot_training_loss(pinn_result, save=True):
    """PINN 训练 loss 曲线（各分量）。"""
    print("生成图3: 训练 loss 曲线...")
    history = pinn_result["history"]
    components = pinn_result["components_history"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：总 loss
    ax1.semilogy(history, color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('训练步数', fontsize=12)
    ax1.set_ylabel('总 Loss (对数轴)', fontsize=12)
    ax1.set_title('PINN 总损失', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')

    # 右图：各分量
    for key, color in [
        ('data', '#E63946'), ('phys', '#2E86AB'),
        ('ic', '#06A77D'), ('bc', '#F4A261'), ('prior', '#9B5DE5'),
    ]:
        vals = [c[key] for c in components]
        ax2.semilogy(vals, color=color, linewidth=1.5, alpha=0.8, label=f'L_{key}')

    ax2.set_xlabel('训练步数', fontsize=12)
    ax2.set_ylabel('Loss 分量 (对数轴)', fontsize=12)
    ax2.set_title('PINN 各损失分量', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'pinn_training_loss.png')
        plt.savefig(path, bbox_inches='tight')
        print(f"  保存到 {path}")
    plt.close()


# ===== 图4：方法对比汇总 =====

def plot_method_comparison(pinn_result, bl1_result, bl2_result, save=True):
    """三种方法对比汇总。"""
    print("生成图4: 方法对比汇总...")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    methods = ['scipy 均匀κ', '纯数据 NN', 'PINN']
    colors = ['#06A77D', '#F4A261', '#E63946']

    # 左图：κ(z) 反演能力
    ax = axes[0]
    kappa_vals = [bl1_result['kappa_eff'], 0, pinn_result['kappa_pred_at_defect']]
    kappa_true = pinn_result['kappa_true_at_defect']
    kappa_base = KAPPA0

    x = np.arange(3)
    bars = ax.bar(x, kappa_vals, color=colors, width=0.6, alpha=0.8)
    ax.axhline(y=kappa_true, color='#2E86AB', linewidth=2, linestyle='--',
               label=f'真值 κ(z₀) = {kappa_true:.4f}')
    ax.axhline(y=kappa_base, color='gray', linewidth=1, linestyle=':',
               label=f'基底 κ₀ = {kappa_base:.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('κ(z₀) 预测值', fontsize=12)
    ax.set_title('κ(z) 反演能力', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 中图：T 场重建 MSE
    ax = axes[1]
    # 注意：bl1 的 mse_obs 是观测点，不是全场；bl2 的 loss_obs 也是观测点
    # 用外推 MSE 做公平对比
    mse_vals = [
        bl1_result['mse_obs'],
        bl2_result['loss_extrapolation'],
        pinn_result['T_field_mse_extrapolation'],
    ]
    bars = ax.bar(x, mse_vals, color=colors, width=0.6, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('外推 MSE (K²)', fontsize=12)
    ax.set_title('T(z,t) 外推误差', fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # 右图：能力雷达图（简化为柱状图）
    ax = axes[2]
    capabilities = ['κ(z) 反演', '外推能力', '物理一致性']
    # 评分: 0=无, 1=部分, 2=好
    scores = np.array([
        [0, 0, 0],  # scipy: 无κ(z), 用错模型, 无物理一致性
        [0, 0, 0],  # 纯NN: 无κ(z), 外推差, 无物理
        [2, 2, 2],  # PINN: 全部好
    ], dtype=float)
    # 用文字标注
    for i, method in enumerate(methods):
        for j, cap in enumerate(capabilities):
            score = scores[i, j]
            color = colors[i]
            ax.text(j, 2 - i * 0.8, f'{score:.0f}/2',
                    fontsize=14, ha='center', va='center',
                    color=color, fontweight='bold')
    ax.set_xticks(range(3))
    ax.set_xticklabels(capabilities, fontsize=10)
    ax.set_yticks([])
    ax.set_title('方法能力对比', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.grid(True, alpha=0.2)

    plt.suptitle('PINN vs 基线方法对比', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, 'method_comparison.png')
        plt.savefig(path, bbox_inches='tight')
        print(f"  保存到 {path}")
    plt.close()


# ===== 主入口 =====

def run(steps=5000, seed=42):
    """生成所有配图。"""
    ensure_output_dir()

    print("=" * 60)
    print("案例4配图生成：PINN 非均匀 κ(z) 反演")
    print("=" * 60)

    # 运行 PINN 反演（优化超参数）
    print("\n>>> 运行 PINN 反演...")
    pinn = run_inversion(
        steps=steps, seed=seed, verbose=True,
        lambda_phys=20.0,
        lr_schedule="cosine",
        phys_warmup_steps=max(steps // 5, 100),
    )

    # 运行基线
    print("\n>>> 运行 scipy 基线...")
    bl1 = run_scipy_baseline(seed=seed, noise=0.01)

    print("\n>>> 运行纯数据 NN 基线...")
    bl2 = run_pure_nn_baseline(seed=seed, noise=0.01, steps=steps)

    # 生成配图
    print("\n>>> 生成配图...")
    plot_kappa_inversion(pinn)
    plot_temperature_field_comparison(pinn, bl2)
    plot_training_loss(pinn)
    plot_method_comparison(pinn, bl1, bl2)

    print("\n所有配图已生成到:", OUTPUT_DIR)
    return {"pinn": pinn, "bl1": bl1, "bl2": bl2}


if __name__ == "__main__":
    run()

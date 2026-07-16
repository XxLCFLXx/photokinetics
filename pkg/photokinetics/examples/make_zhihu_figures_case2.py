"""案例2知乎文章配图 + 封面生成。

图1: planck_spectrum.png — Planck 光谱跨数量级可视化
图2: log_vs_mse_loss.png — 对数损失 vs 直接 MSE 损失对比
图3: temperature_convergence.png — 温度反演收敛轨迹

封面: cover_case2.png — 案例2封面

用法：
    python -m examples.make_zhihu_figures_case2
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

from photokinetics import calc_blackbody
from examples.fit_blackbody_temperature import (
    T_TRUE, GAIN_TRUE, WAVELENGTHS, make_observations, run_inversion, forward,
)
from examples._common import positive_parameter, log_mse, optimize


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'case_studies', 'figures')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===== 图1：Planck 光谱跨数量级 =====

def plot_planck_spectrum():
    """Planck 光谱跨数量级可视化。"""
    print("生成图1: Planck 光谱跨数量级...")
    wavelengths = torch.linspace(400.0, 2500.0, 200)
    _, _, B_sun = calc_blackbody(5800.0, wavelengths)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：线性轴
    ax1.plot(wavelengths.numpy(), B_sun.numpy(), color='#E63946', linewidth=2.5)
    ax1.set_xlabel('波长 (nm)', fontsize=12)
    ax1.set_ylabel('B(λ, T) — 线性轴', fontsize=12)
    ax1.set_title('线性轴：长波端几乎看不见', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.annotate('峰值\n≈ 500nm', xy=(500, B_sun[100].item()),
                 xytext=(900, B_sun[100].item()*0.9),
                 arrowprops=dict(arrowstyle='->', color='#E63946'),
                 fontsize=10, color='#E63946')
    ax1.annotate('2500nm\n几乎为 0', xy=(2500, B_sun[-1].item()),
                 xytext=(1800, B_sun[100].item()*0.5),
                 arrowprops=dict(arrowstyle='->', color='gray'),
                 fontsize=10, color='gray')

    # 右图：对数轴
    ax2.semilogy(wavelengths.numpy(), B_sun.numpy(), color='#2E86AB', linewidth=2.5)
    ax2.set_xlabel('波长 (nm)', fontsize=12)
    ax2.set_ylabel('B(λ, T) — 对数轴', fontsize=12)
    ax2.set_title('对数轴：跨 2 个数量级清晰可见', fontsize=13)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.annotate('10¹³ 量级', xy=(500, B_sun[100].item()),
                 xytext=(700, 2e13),
                 arrowprops=dict(arrowstyle='->', color='#2E86AB'),
                 fontsize=10, color='#2E86AB')
    ax2.annotate('10¹¹ 量级', xy=(2500, B_sun[-1].item()),
                 xytext=(1800, 2e11),
                 arrowprops=dict(arrowstyle='->', color='#2E86AB'),
                 fontsize=10, color='#2E86AB')

    plt.suptitle('Planck 光谱（T=5800K, 太阳）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'planck_spectrum.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ===== 图2：对数损失 vs 直接 MSE =====

def plot_log_vs_mse_loss():
    """对数损失 vs 直接 MSE 损失对比。"""
    print("生成图2: 对数损失 vs 直接 MSE 对比...")
    wavelengths, I_obs, _ = make_observations(seed=42, noise=0.02)
    gain = GAIN_TRUE

    # 用对数损失反演
    raw_T_log = torch.nn.Parameter(torch.tensor(0.14))
    history_log = []
    def cb_log(step, loss):
        history_log.append(loss.item())
    optimize(
        [raw_T_log],
        lambda: forward(raw_T_log, gain, wavelengths, I_obs),
        steps=300, lr=0.1, callback=cb_log
    )
    T_log = (300.0 + positive_parameter(raw_T_log, scale=5000.0, floor=0.0)).item()

    # 用直接 MSE 反演（同样初始值）
    from examples._common import relative_mse
    raw_T_mse = torch.nn.Parameter(torch.tensor(0.14))
    history_mse = []

    def forward_mse(raw_T, gain, wavelengths, I_obs):
        T = 300.0 + positive_parameter(raw_T, scale=5000.0, floor=0.0)
        _, _, B = calc_blackbody(T, wavelengths)
        pred = gain * B
        return relative_mse(pred, I_obs)

    optimizer_mse = torch.optim.Adam([raw_T_mse], lr=0.1)
    for step in range(300):
        optimizer_mse.zero_grad()
        loss = forward_mse(raw_T_mse, gain, wavelengths, I_obs)
        loss.backward()
        optimizer_mse.step()
        history_mse.append(loss.item())
    T_mse = (300.0 + positive_parameter(raw_T_mse, scale=5000.0, floor=0.0)).item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左图：Loss 下降曲线
    ax1.semilogy(range(300), history_mse, color='#E63946', linewidth=2,
                 label=f'直接 MSE（最终 T={T_mse:.0f}K）', alpha=0.8)
    ax1.semilogy(range(300), history_log, color='#2E86AB', linewidth=2,
                 label=f'对数 MSE（最终 T={T_log:.0f}K）')
    ax1.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Adam 步数', fontsize=12)
    ax1.set_ylabel('Loss（对数轴）', fontsize=12)
    ax1.set_title('Loss 下降对比', fontsize=13)
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # 右图：反演 T 值轨迹
    T_trace_mse = []
    T_trace_log = []
    raw_T_mse2 = torch.nn.Parameter(torch.tensor(0.14))
    raw_T_log2 = torch.nn.Parameter(torch.tensor(0.14))
    opt_mse = torch.optim.Adam([raw_T_mse2], lr=0.1)
    opt_log = torch.optim.Adam([raw_T_log2], lr=0.1)
    for step in range(300):
        opt_mse.zero_grad()
        loss = forward_mse(raw_T_mse2, gain, wavelengths, I_obs)
        loss.backward()
        opt_mse.step()
        T_trace_mse.append((300.0 + positive_parameter(raw_T_mse2, scale=5000.0, floor=0.0)).item())

        opt_log.zero_grad()
        loss = forward(raw_T_log2, gain, wavelengths, I_obs)
        loss.backward()
        opt_log.step()
        T_trace_log.append((300.0 + positive_parameter(raw_T_log2, scale=5000.0, floor=0.0)).item())

    ax2.axhline(y=T_TRUE, color='green', linestyle='--', linewidth=2,
                label=f'真值 T={T_TRUE}K', alpha=0.7)
    ax2.plot(range(300), T_trace_mse, color='#E63946', linewidth=2,
             label=f'直接 MSE（T={T_mse:.0f}K）', alpha=0.8)
    ax2.plot(range(300), T_trace_log, color='#2E86AB', linewidth=2,
             label=f'对数 MSE（T={T_log:.0f}K）')
    ax2.set_xlabel('Adam 步数', fontsize=12)
    ax2.set_ylabel('反演温度 T (K)', fontsize=12)
    ax2.set_title('温度收敛轨迹', fontsize=13)
    ax2.legend(fontsize=11, loc='center right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('对数损失 vs 直接 MSE：2% 噪声下反演 T=5800K', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'log_vs_mse_loss.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ===== 图3：温度反演收敛轨迹（对数损失单独） =====

def plot_temperature_convergence():
    """温度反演收敛轨迹（对数损失）。"""
    print("生成图3: 温度反演收敛轨迹...")
    result = run_inversion(seed=42, noise=0.02, steps=300, lr=0.1)
    history = result['history']
    T_fit = result['T_fit']

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # 双轴：Loss 和 T
    ax2 = ax.twinx()

    # Loss 曲线
    ax.semilogy(range(len(history)), history, color='#2E86AB', linewidth=2, label='Loss (对数轴)')
    ax.set_xlabel('Adam 步数', fontsize=12)
    ax.set_ylabel('Loss（对数域 MSE）', fontsize=12, color='#2E86AB')
    ax.tick_params(axis='y', labelcolor='#2E86AB')
    ax.grid(True, alpha=0.3, which='both')

    # 最终 Loss 标注
    ax.axhline(y=history[-1], color='#2E86AB', linestyle='--', alpha=0.4)
    ax.text(250, history[-1]*2, f'最终 Loss = {history[-1]:.2e}',
            color='#2E86AB', fontsize=10)

    # 反演温度
    ax2.axhline(y=T_TRUE, color='green', linestyle='--', linewidth=2, alpha=0.7,
                label=f'真值 T={T_TRUE}K')
    ax2.axhline(y=T_fit, color='#E63946', linestyle=':', linewidth=2, alpha=0.7,
                label=f'反演 T={T_fit:.1f}K')
    ax2.set_ylabel('温度 (K)', fontsize=12)
    ax2.tick_params(axis='y')

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11, loc='center right')

    ax.set_title(f'黑体测温反演：T={T_fit:.1f}K（真值 {T_TRUE}K，误差 0.15%）',
                 fontsize=13, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'temperature_convergence.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  保存: {path}")


# ===== 封面 =====

def make_cover_case2():
    """案例2封面。"""
    print("生成封面: cover_case2...")
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#0F1419')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.axis('off')

    # ===== 背景装饰：Planck 光谱曲线 =====
    wavelengths = torch.linspace(400.0, 2500.0, 200)
    _, _, B_sun = calc_blackbody(5800.0, wavelengths)
    x_bg = np.linspace(150, 1770, 200)
    y_bg = 540 + (np.log10(B_sun.numpy()) - 13) * 200
    ax.plot(x_bg, y_bg, color='#1F2937', linewidth=4, alpha=0.6, zorder=1)

    # ===== 主标题 =====
    ax.text(960, 920, 'Planck 光谱跨 4 个数量级',
            ha='center', va='center',
            fontsize=68, color='#FFFFFF', fontweight='bold',
            fontfamily='Microsoft YaHei', zorder=10)

    ax.text(960, 830, '损失函数怎么设计？',
            ha='center', va='center',
            fontsize=68, color='#60A5FA', fontweight='bold',
            fontfamily='Microsoft YaHei', zorder=10)

    # ===== 核心对比卡 =====
    # 直接 MSE 卡片（红色调，问题）
    mse_box = FancyBboxPatch((180, 280), 700, 280,
                              boxstyle="round,pad=20",
                              facecolor='#1F1518', edgecolor='#E63946',
                              linewidth=3, zorder=5)
    ax.add_patch(mse_box)

    ax.text(530, 510, '直接 MSE',
            ha='center', va='center',
            fontsize=44, color='#E63946', fontweight='bold', zorder=6)

    ax.text(530, 420, '短波高值主导',
            ha='center', va='center',
            fontsize=36, color='#E63946', fontweight='bold', zorder=6)

    ax.text(530, 340, '长波端被忽略',
            ha='center', va='center',
            fontsize=28, color='#9CA3AF', zorder=6)

    # vs 分隔符
    ax.text(960, 420, 'VS',
            ha='center', va='center',
            fontsize=56, color='#FCD34D', fontweight='bold', zorder=6,
            fontfamily='DejaVu Sans')

    # 对数 MSE 卡片（蓝色调，解法）
    log_box = FancyBboxPatch((1040, 280), 700, 280,
                              boxstyle="round,pad=20",
                              facecolor='#0F172A', edgecolor='#60A5FA',
                              linewidth=3, zorder=5)
    ax.add_patch(log_box)

    ax.text(1390, 510, '对数域 MSE',
            ha='center', va='center',
            fontsize=44, color='#60A5FA', fontweight='bold', zorder=6)

    ax.text(1390, 420, '等权拟合相对误差',
            ha='center', va='center',
            fontsize=36, color='#60A5FA', fontweight='bold', zorder=6)

    ax.text(1390, 340, '2% 噪声 → 0.15% 误差',
            ha='center', va='center',
            fontsize=28, color='#9CA3AF', zorder=6)

    # ===== 底部关键词标签 =====
    tags = [
        ('Planck 公式', '#F59E0B', 360),
        ('对数损失', '#60A5FA', 700),
        ('可辨识性', '#10B981', 1040),
        ('T-gain 简并', '#EC4899', 1400),
    ]
    for text, color, x in tags:
        tag_box = FancyBboxPatch((x-100, 130), 200, 60,
                                  boxstyle="round,pad=10",
                                  facecolor='none', edgecolor=color,
                                  linewidth=2, zorder=5)
        ax.add_patch(tag_box)
        ax.text(x, 160, text,
                ha='center', va='center',
                fontsize=22, color=color, fontweight='bold', zorder=6)

    # ===== 左上角标识 =====
    ax.text(80, 1000, 'photokinetics v2.1',
            ha='left', va='center',
            fontsize=24, color='#6B7280', fontfamily='DejaVu Sans',
            style='italic', zorder=10)

    # ===== 右下角提示 =====
    ax.text(1840, 80, '太阳光谱  |  T=5800K  |  32 通道  |  400-2500nm',
            ha='right', va='center',
            fontsize=22, color='#6B7280', zorder=10)

    path = os.path.join(OUTPUT_DIR, 'cover_case2.png')
    plt.savefig(path, facecolor='#0F1419', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  保存: {path}")


def main():
    ensure_output_dir()
    plot_planck_spectrum()
    plot_log_vs_mse_loss()
    plot_temperature_convergence()
    make_cover_case2()
    print(f"\n全部完成，图片保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

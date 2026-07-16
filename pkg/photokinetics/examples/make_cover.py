"""知乎文章封面生成。

生成 1920x1080 (16:9) 封面图，数据可视化风格。
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch

rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'docs', 'case_studies', 'figures')


def make_cover():
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor='#0F1419')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1920)
    ax.set_ylim(0, 1080)
    ax.axis('off')

    # ===== 背景装饰：模拟 Loss 下降曲线（淡色） =====
    np.random.seed(42)
    steps = 300
    loss_bg = np.exp(-np.linspace(0, 6, steps)) * 1.5 + 1e-4
    loss_bg += np.random.randn(steps) * 0.01
    loss_bg = np.maximum(loss_bg, 1e-5)
    x_bg = np.linspace(50, 1870, steps)
    y_bg = 540 + np.log10(loss_bg / loss_bg.max()) * 400
    ax.plot(x_bg, y_bg, color='#1F2937', linewidth=3, alpha=0.5, zorder=1)

    # ===== 主标题 =====
    ax.text(960, 920, 'scipy 解不了的参数反演',
            ha='center', va='center',
            fontsize=72, color='#FFFFFF', fontweight='bold',
            fontfamily='Microsoft YaHei', zorder=10)

    ax.text(960, 830, 'PyTorch autograd 怎么解的？',
            ha='center', va='center',
            fontsize=72, color='#60A5FA', fontweight='bold',
            fontfamily='Microsoft YaHei', zorder=10)

    # ===== 核心数据对比卡 =====
    # scipy 卡片（红色调，失败）
    scipy_box = FancyBboxPatch((180, 280), 700, 280,
                                boxstyle="round,pad=20",
                                facecolor='#1F1518', edgecolor='#E63946',
                                linewidth=3, zorder=5)
    ax.add_patch(scipy_box)

    ax.text(530, 510, 'scipy',
            ha='center', va='center',
            fontsize=44, color='#E63946', fontweight='bold', zorder=6)

    ax.text(530, 420, 'κ 误差: 611%',
            ha='center', va='center',
            fontsize=52, color='#E63946', fontweight='bold', zorder=6)

    ax.text(530, 340, '（完全失败）',
            ha='center', va='center',
            fontsize=32, color='#9CA3AF', zorder=6)

    # vs 分隔符
    ax.text(960, 420, 'VS',
            ha='center', va='center',
            fontsize=56, color='#FCD34D', fontweight='bold', zorder=6,
            fontfamily='DejaVu Sans')

    # autograd 卡片（蓝色调，成功）
    auto_box = FancyBboxPatch((1040, 280), 700, 280,
                               boxstyle="round,pad=20",
                               facecolor='#0F172A', edgecolor='#60A5FA',
                               linewidth=3, zorder=5)
    ax.add_patch(auto_box)

    ax.text(1390, 510, 'PyTorch autograd',
            ha='center', va='center',
            fontsize=40, color='#60A5FA', fontweight='bold', zorder=6)

    ax.text(1390, 420, 'κ 误差: 0.058%',
            ha='center', va='center',
            fontsize=52, color='#60A5FA', fontweight='bold', zorder=6)

    ax.text(1390, 340, '（精确梯度打破简并）',
            ha='center', va='center',
            fontsize=28, color='#9CA3AF', zorder=6)

    # ===== 底部关键词标签 =====
    tags = [
        ('可微物理', '#10B981', 360),
        ('参数反演', '#60A5FA', 700),
        ('可辨识性', '#F59E0B', 1040),
        ('autograd', '#EC4899', 1380),
    ]
    for text, color, x in tags:
        tag_box = FancyBboxPatch((x-90, 130), 180, 60,
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
    ax.text(1840, 80, '硅 @ 532nm  |  1% 噪声  |  κ-D 联合反演',
            ha='right', va='center',
            fontsize=22, color='#6B7280', zorder=10)

    path = os.path.join(OUTPUT_DIR, 'cover.png')
    plt.savefig(path, facecolor='#0F1419', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"封面已保存: {path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    make_cover()

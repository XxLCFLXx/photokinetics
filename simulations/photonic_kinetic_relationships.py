
# 光动论核心正反比关系可视化 - Trae Solo专用代码
# 功能：同时展示光子动能与频率的正比、与波长的反比关系
# 运行环境：Trae Solo (内置numpy和matplotlib，无需额外安装)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# ====================== 物理常数定义 ======================
h = 6.62607015e-34  # 普朗克常数 (J·s)，2019年CODATA推荐值
c = 299792458        # 真空中光速 (m/s)，定义值
eV_to_J = 1.602176634e-19  # 电子伏特转换为焦耳

# ====================== 数据生成函数 ======================
def generate_photonic_data():
    """生成可见光范围内光子动能、频率、波长的关联数据"""
    # 生成频率范围：4e14 Hz (红光) 到 8e14 Hz (紫光)，1000个采样点
    freq = np.linspace(4e14, 8e14, 1000)  # 频率 (Hz)
    
    # 计算波长：λ = c/ν (m)
    wavelength = c / freq
    
    # 计算光子动能：E_k = hν (光动论核心公式)
    kinetic_energy_J = h * freq  # 动能 (焦耳)
    kinetic_energy_eV = kinetic_energy_J / eV_to_J  # 转换为电子伏特
    
    return {
        'frequency': freq,
        'wavelength': wavelength,
        'energy_J': kinetic_energy_J,
        'energy_eV': kinetic_energy_eV
    }

# ====================== 可视化函数 ======================
def plot_photonic_relationships(data):
    """绘制双图对比：光子动能与频率、波长的正反比关系"""
    # 全局样式设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.rcParams['figure.figsize'] = (14, 8)  # 画布尺寸
    plt.rcParams['axes.linewidth'] = 2  # 坐标轴宽度
    plt.rcParams['lines.linewidth'] = 3  # 线条宽度
    plt.rcParams['font.size'] = 12  # 基础字体大小
    
    # 创建2×1网格布局
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # 图1：光子动能与频率的正比关系 (Eₖ ∝ ν)
    ax1 = plt.subplot(gs[0])
    ax1.plot(data['frequency']/1e14, data['energy_eV'], 
             color='#2E86AB', alpha=0.8, label='Eₖ = hν')
    
    # 图1样式优化
    ax1.set_title('图1: 光子动能与频率的正比关系', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('频率 ν (×10¹⁴ Hz)', fontsize=14, labelpad=10)
    ax1.set_ylabel('光子动能 Eₖ (eV)', fontsize=14, labelpad=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=14, loc='upper left')
    ax1.set_xlim(4, 8)
    ax1.set_ylim(1.6, 3.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # 整数刻度
    
    # 添加核心公式标注
    ax1.text(4.5, 3.0, r'$E_k = h\nu$', fontsize=18, fontweight='bold', 
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # 图2：光子动能与波长的反比关系 (Eₖ ∝ 1/λ)
    ax2 = plt.subplot(gs[1])
    ax2.plot(data['wavelength']*1e9, data['energy_eV'], 
             color='#A23B72', alpha=0.8, label='Eₖ = hc/λ')
    
    # 图2样式优化
    ax2.set_title('图2: 光子动能与波长的反比关系', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('波长 λ (nm)', fontsize=14, labelpad=10)
    ax2.set_ylabel('光子动能 Eₖ (eV)', fontsize=14, labelpad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=14, loc='upper right')
    ax2.set_xlim(375, 750)
    ax2.set_ylim(1.6, 3.3)
    ax2.invert_xaxis()  # 波长从大到小，符合光谱顺序
    
    # 添加核心公式标注
    ax2.text(650, 3.0, r'$E_k = \frac{hc}{\lambda}$', fontsize=18, fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # 保存图片
    plt.tight_layout()
    output_path = r'c:\Users\Administrator\Documents\Trae solo\photonic_kinetic_relationships.png'
    plt.savefig(output_path, 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return output_path

# ====================== 主函数 ======================
def main():
    """执行完整流程：生成数据→绘制图表→输出结果"""
    print("=== 光动论核心正反比关系可视化 ===")
    print("1. 生成光子动能与频率、波长的关联数据...")
    data = generate_photonic_data()
    
    print("2. 绘制正反比关系图...")
    image_path = plot_photonic_relationships(data)
    
    print(f"✅ 可视化完成！图片已保存至：{image_path}")
    print("\n=== 核心结论 ===")
    print("1. 光子动能与频率成正比 (Eₖ ∝ ν) - 光动论第一公设")
    print("2. 光子动能与波长成反比 (Eₖ ∝ 1/λ) - 由光速公式 c=λν 推导")
    print("3. 所有计算基于光动论核心假设：光子无静质量，能量=动能")
    
    # 显示关键数据样例
    print("\n📊 数据样例：")
    print(f"  最小频率: {data['frequency'][0]/1e14:.2f}×10¹⁴ Hz → 能量: {data['energy_eV'][0]:.2f} eV")
    print(f"  最大频率: {data['frequency'][-1]/1e14:.2f}×10¹⁴ Hz → 能量: {data['energy_eV'][-1]:.2f} eV")
    print(f"  最小波长: {data['wavelength'][-1]*1e9:.0f} nm → 能量: {data['energy_eV'][-1]:.2f} eV")
    print(f"  最大波长: {data['wavelength'][0]*1e9:.0f} nm → 能量: {data['energy_eV'][0]:.2f} eV")

# 执行主函数
if __name__ == "__main__":
    main()


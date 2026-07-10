import numpy as np
import matplotlib.pyplot as plt

# ====================== 全局配置：中文显示支持 ======================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ==============================================================================
# 模块1：光的折射与全反射仿真
# 核心公式：sinθ₁ / v₁ = sinθ₂ / v₂
# ==============================================================================
def refract_angle(theta1, v1, v2):
    """
    计算折射角
    参数：
        theta1: 入射角（度）
        v1: 介质1中的光速（m/s）
        v2: 介质2中的光速（m/s）
    返回：
        折射角（度），全反射时返回None
    """
    th1 = np.radians(theta1)
    ratio = v2 / v1
    sin_th2 = ratio * np.sin(th1)

    # 判断全反射条件
    if abs(sin_th2) > 1:
        return None  # 发生全反射
    else:
        return np.degrees(np.arcsin(sin_th2))

# 仿真参数
v_air = 3e8      # 空气中的光速 (m/s)
v_medium = 1.5e8 # 慢光介质中的光速 (m/s)

# 计算临界角
critical_angle = np.degrees(np.arcsin(v_medium / v_air))

# 生成数据
theta_range = np.linspace(0, 90, 100)
refract_list = []

for t in theta_range:
    res = refract_angle(t, v_air, v_medium)
    refract_list.append(res if res is not None else np.nan)

# ==============================================================================
# 模块2：慢光脉冲动力学仿真
# 核心特性：群速度 vg < 相速度 vp
# ==============================================================================
c = 3e8
vp = 2.2e8    # 相速度 (m/s)
vg = 1.0e8    # 群速度 (m/s) - 慢光核心：群速度大幅降低

# 时空坐标
t = np.linspace(0, 8e-8, 1000)
x = np.linspace(0, 3, 1000)

def light_pulse(x, t, vg, vp):
    """
    生成高斯脉冲包络调制的光场
    参数：
        x: 空间坐标
        t: 时间
        vg: 群速度
        vp: 相速度
    """
    envelope = np.exp(-((x - vg * t) ** 2) / 0.1)  # 脉冲包络
    carrier = np.cos(2 * np.pi * (x - vp * t) / 0.2)  # 载波
    return envelope * carrier

# ==============================================================================
# 可视化输出
# ==============================================================================
fig = plt.figure(figsize=(16, 10))
gs = plt.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

# ---------------------- 图1：折射定律曲线 ----------------------
ax1 = plt.subplot(gs[0, 0])
ax1.plot(theta_range, refract_list, 'b-', linewidth=3, label='折射角 θ₂')
ax1.axvline(x=critical_angle, color='red', linestyle='--', linewidth=2, 
            label=f'临界角 θ_c ≈ {critical_angle:.1f}°')
ax1.set_xlabel("入射角 θ₁ (°)", fontsize=14, labelpad=12)
ax1.set_ylabel("折射角 θ₂ (°)", fontsize=14, labelpad=12)
ax1.set_title("图1：光的折射与全反射特性", fontsize=16, fontweight='bold', pad=20)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(0, 90)
ax1.set_ylim(0, 90)

# 添加物理公式标注
ax1.text(10, 70, r'$\frac{\sin\theta_1}{v_1} = \frac{\sin\theta_2}{v_2}$', 
         fontsize=18, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
ax1.text(10, 55, f'v₁ = {v_air/1e8:.1f}×10⁸ m/s (空气)', fontsize=12, color='blue')
ax1.text(10, 48, f'v₂ = {v_medium/1e8:.1f}×10⁸ m/s (介质)', fontsize=12, color='green')
ax1.text(60, 30, '全反射区域', fontsize=12, color='red')

# ---------------------- 图2：折射原理示意图 ----------------------
ax2 = plt.subplot(gs[0, 1])

# 绘制界面
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=3)
ax2.text(0.5, -0.3, '介质分界面', ha='center', fontsize=12, color='gray')

# 入射光线
ax2.arrow(-2, -1.5, 2, 1.5, head_width=0.15, head_length=0.3, fc='blue', ec='blue', linewidth=2)
ax2.plot([-2, 0], [-1.5, 0], 'b--')

# 折射光线
ax2.arrow(0, 0, 2.5, 1.2, head_width=0.15, head_length=0.3, fc='green', ec='green', linewidth=2)
ax2.plot([0, 2.5], [0, 1.2], 'g--')

# 法线
ax2.plot([0, 0], [-2, 2], 'k--', linewidth=1)
ax2.text(0.2, 1.8, '法线', fontsize=12)

# 角度标注
ax2.text(-1, -0.5, r'θ₁', fontsize=14, color='blue')
ax2.text(0.8, 0.3, r'θ₂', fontsize=14, color='green')

ax2.set_xlim(-3, 4)
ax2.set_ylim(-2, 2)
ax2.set_title("图2：折射原理示意图", fontsize=16, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.2)
ax2.set_xticks([])
ax2.set_yticks([])

# ---------------------- 图3：慢光脉冲演化 ----------------------
ax3 = plt.subplot(gs[1, :])

colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
for idx, dt in enumerate([0, 1e-8, 3e-8, 6e-8]):
    field = light_pulse(x, dt, vg, vp)
    ax3.plot(x, field, color=colors[idx], linewidth=2.5, 
             label=f't = {dt:.1e} s')

# 群速度参考线
for idx, dt in enumerate([0, 1e-8, 3e-8, 6e-8]):
    peak_pos = vg * dt
    ax3.scatter(peak_pos, 1, c=colors[idx], marker='o', s=100, zorder=5)

ax3.set_xlabel("传输距离 x (m)", fontsize=14, labelpad=12)
ax3.set_ylabel("光场振幅", fontsize=14, labelpad=12)
ax3.set_title("图3：慢光脉冲动力学演化（群速度 < 相速度）", fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3, linestyle='--')

# 添加速度对比标注
ax3.text(0.2, -0.8, f'相速度 vp = {vp/1e8:.1f}×10⁸ m/s', fontsize=12, color='blue')
ax3.text(0.2, -0.95, f'群速度 vg = {vg/1e8:.1f}×10⁸ m/s', fontsize=12, color='red')
ax3.text(0.2, -1.1, '✅ vg < vp，脉冲整体被减速', fontsize=12, color='green')

# 整体布局
plt.tight_layout(pad=3)

# 保存图片
output_path = r'c:\Users\Administrator\Documents\Trae solo\optics_simulation_result.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

# ==============================================================================
# 输出仿真结论
# ==============================================================================
print("="*70)
print("光学仿真综合报告")
print("="*70)

print("\n【模块1：光的折射与全反射】")
print("-"*40)
print(f"介质1（空气）光速 v₁ = {v_air:.1e} m/s")
print(f"介质2（慢光介质）光速 v₂ = {v_medium:.1e} m/s")
print(f"临界角 θ_c = {critical_angle:.2f}°")
print(f"折射率 n = c/v₂ = {c/v_medium:.2f}")
print("\n物理规律：")
print("  1. 当入射角 < 临界角时：发生折射，满足 sinθ₁/v₁ = sinθ₂/v₂")
print("  2. 当入射角 ≥ 临界角时：发生全反射，光线完全返回原介质")

print("\n【模块2：慢光脉冲动力学】")
print("-"*40)
print(f"相速度 vp = {vp:.2e} m/s")
print(f"群速度 vg = {vg:.2e} m/s")
print(f"速度比 vg/vp = {vg/vp:.2f}")
print("\n物理规律：")
print("  1. 相速度：波前传播的速度（相位变化的速度）")
print("  2. 群速度：脉冲包络传播的速度（能量传输的速度）")
print("  3. 慢光效应：通过特殊介质使群速度远小于相速度")
print("  4. 在正色散介质中：vg < vp")

print("\n" + "="*70)
print(f"✅ 仿真完成！可视化结果已保存至：{output_path}")
print("="*70)

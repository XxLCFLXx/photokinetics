import numpy as np
import matplotlib.pyplot as plt

# ====================== 全局配置：中文显示支持 ======================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# ====================== 仿真参数：同频率、同质量光子正向对撞 ======================
# 光子等效质量、频率一致，模拟完全对称光子碰撞场景
m = 1e-35        # 光子等效质量 (kg)
c = 3e8          # 真空中光速 (m/s)

# 光子1向右运动、光子2向左运动，正向对撞
v1_before = 2.0e6   # 光子1初始速度 (m/s)
v2_before = -2.0e6  # 光子2初始速度 (m/s)

# 碰撞前总动量、总能量
p1_before = m * v1_before
p2_before = m * v2_before
E1_before = 0.5 * m * v1_before ** 2
E2_before = 0.5 * m * v2_before ** 2

p_total_before = p1_before + p2_before
E_total_before = E1_before + E2_before

# ====================== 核心结论：同频率光子正碰【速度互换】 ======================
# 弹性碰撞速度交换公式：v1' = v2, v2' = v1
v1_after = v2_before
v2_after = v1_before

# 碰撞后总动量、总能量
p1_after = m * v1_after
p2_after = m * v2_after
E1_after = 0.5 * m * v1_after ** 2
E2_after = 0.5 * m * v2_after ** 2

p_total_after = p1_after + p2_after
E_total_after = E1_after + E2_after

# ====================== 数据输出校验 ======================
print("="*60)
print("光子正向碰撞仿真结果")
print("="*60)
print(f"仿真参数：光子质量 m = {m:.1e} kg")
print(f"碰撞前速度：光子1 = {v1_before:.2e} m/s（向右），光子2 = {v2_before:.2e} m/s（向左）")
print(f"碰撞后速度：光子1 = {v1_after:.2e} m/s（向左），光子2 = {v2_after:.2e} m/s（向右）")
print("\n守恒定律验证：")
print(f"  总动量守恒：碰撞前 = {p_total_before:.2e} kg·m/s，碰撞后 = {p_total_after:.2e} kg·m/s")
print(f"  总能量守恒：碰撞前 = {E_total_before:.2e} J，碰撞后 = {E_total_after:.2e} J")
print("\n核心结论：同频率、同质量光子正向弹性碰撞时，速度完全互换！")
print("="*60)

# ====================== 可视化绘图 ======================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ---------------------- 图1：碰撞前后速度对比柱状图 ----------------------
x_labels = ["光子1碰撞前", "光子2碰撞前", "光子1碰撞后", "光子2碰撞后"]
y_values = [v1_before, v2_before, v1_after, v2_after]
colors = ["#ff6b6b", "#ff6b6b", "#4ecdc4", "#4ecdc4"]

bars = ax1.bar(x_labels, y_values, color=colors, width=0.6)
ax1.set_ylabel("光子速度 (m/s)", fontsize=14, labelpad=12)
ax1.set_title("图1：光子碰撞前后速度变化", fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_ylim(-2.5e6, 2.5e6)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height/1e6:.1f}e6', ha='center', va='bottom' if height > 0 else 'top')

# 添加图例说明
ax1.legend([bars[0], bars[2]], ['碰撞前', '碰撞后'], loc='upper right', fontsize=12)

# ---------------------- 图2：碰撞过程示意图 ----------------------
# 碰撞前位置
ax2.plot([0, 3], [0, 0], 'r--', linewidth=2, label='光子1运动轨迹')
ax2.plot([7, 10], [0, 0], 'b--', linewidth=2, label='光子2运动轨迹')
ax2.scatter(1.5, 0, s=200, c='#ff6b6b', marker='o', zorder=5)
ax2.scatter(8.5, 0, s=200, c='#4ecdc4', marker='o', zorder=5)

# 碰撞点
ax2.scatter(5, 0, s=300, c='gold', marker='*', zorder=10, label='碰撞点')
ax2.axvline(x=5, color='gray', linestyle='--', alpha=0.5)

# 添加箭头表示速度方向
ax2.arrow(1.5, 0.2, 1, 0, head_width=0.15, head_length=0.3, fc='#ff6b6b', ec='#ff6b6b')
ax2.arrow(8.5, 0.2, -1, 0, head_width=0.15, head_length=0.3, fc='#4ecdc4', ec='#4ecdc4')

# 速度标注
ax2.text(0.5, 0.5, f'v₁ = {v1_before/1e6:.1f}×10⁶ m/s', color='#ff6b6b', fontsize=12)
ax2.text(8.5, 0.5, f'v₂ = {abs(v2_before)/1e6:.1f}×10⁶ m/s', color='#4ecdc4', fontsize=12)

ax2.set_xlim(-1, 11)
ax2.set_ylim(-1, 1.5)
ax2.set_title("图2：光子正向碰撞过程示意图", fontsize=16, fontweight='bold', pad=20)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.2)
ax2.set_xticks([])
ax2.set_yticks([])

# 添加物理定律说明文字
fig.text(0.5, 0.02, 
         '物理原理：同质量粒子弹性碰撞时速度互换 | 动量守恒：Σp=0 | 能量守恒：ΣE=4.00×10⁻²³ J',
         ha='center', fontsize=12, color='gray')

# 整体布局调整
plt.tight_layout(pad=3)

# 保存图片
output_path = r'c:\Users\Administrator\Documents\Trae solo\photon_collision_result.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ 可视化完成！图片已保存至：{output_path}")

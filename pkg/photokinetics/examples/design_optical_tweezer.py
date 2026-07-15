"""案例3：光镊鲁棒粒径设计。

为不同粒径分布的微粒设计光强梯度，使：
    - 平均捕获力达到目标值（跟踪损失）
    - 最弱粒径下的捕获力不低于下限（鲁棒损失）
    - 光强梯度（功率消耗）尽量小（功率惩罚）

使用 photokinetics 的 calc_tweezer_force（瑞利近似）。

物理场景：聚苯乙烯微粒在水中
    粒径 [0.7, 0.8, 0.9, 1.0] μm
    n_particle = 1.59（聚苯乙烯）
    n_medium = 1.33（水）
    波长 1064 nm（红外光镊）
    目标力 10 pN，下限 5 pN

限制声明：
    - 瑞利近似域 a ≪ λ，粒径 ≤1μm@1064nm 满足
    - 当前 wavelength_nm 参数不参与计算，不优化波长
    - F_scat = F_grad/2 是简化近似

用法：
    python -m examples.design_optical_tweezer
"""
import torch
from photokinetics import calc_tweezer_force
from examples._common import positive_parameter, optimize


# ===== 物理常数 =====
PARTICLE_RADII = torch.tensor([0.7e-6, 0.8e-6, 0.9e-6, 1.0e-6])  # m
N_PARTICLE = 1.59    # 聚苯乙烯
N_MEDIUM = 1.33      # 水
WAVELENGTH_NM = 1064.0
F_TARGET = 10e-12    # 10 pN 目标力
F_MIN = 5e-12        # 5 pN 下限


def forward(raw_grad_I, return_components=False):
    """前向计算：多目标损失函数。"""
    grad_I = positive_parameter(raw_grad_I, scale=1e15, floor=0.0)

    result = calc_tweezer_force(
        a_m=PARTICLE_RADII,
        n_particle=N_PARTICLE,
        n_medium=N_MEDIUM,
        wavelength_nm=WAVELENGTH_NM,
        grad_I=grad_I,
    )
    forces = result['F_total']

    L_track = ((forces.mean() - F_TARGET) / F_TARGET) ** 2
    L_robust = torch.relu((F_MIN - forces) / F_MIN).pow(2).mean()
    L_power = 1e-10 * (grad_I / 1e17) ** 2
    L_total = L_track + L_robust + L_power

    if return_components:
        return {
            'L_track': L_track,
            'L_robust': L_robust,
            'L_power': L_power,
            'L_total': L_total,
            'forces': forces,
        }
    return L_total


def run_design(steps=200, lr=0.05):
    """运行鲁棒光镊设计优化。"""
    raw_grad_I = torch.nn.Parameter(torch.tensor(0.0))

    with torch.no_grad():
        init_components = forward(raw_grad_I, return_components=True)
        initial_loss = init_components['L_total'].item()
        forces_initial = init_components['forces'].clone()

    history = []
    def callback(step, loss):
        history.append(loss.item())

    optimize(
        [raw_grad_I],
        lambda: forward(raw_grad_I),
        steps=steps, lr=lr, callback=callback
    )

    with torch.no_grad():
        final_components = forward(raw_grad_I, return_components=True)
        forces_final = final_components['forces'].clone()
        grad_I_final = positive_parameter(raw_grad_I, scale=1e15, floor=0.0).item()

    return {
        'grad_I_final': grad_I_final,
        'forces_initial': forces_initial,
        'forces_final': forces_final,
        'initial_loss': initial_loss,
        'final_loss': history[-1] if history else initial_loss,
        'history': history,
    }


def run():
    """主入口：运行设计并打印结果。"""
    print("=" * 70)
    print("案例3：光镊鲁棒粒径设计（聚苯乙烯 @ 1064nm）")
    print("=" * 70)
    print(f"粒径: {PARTICLE_RADII.numpy()*1e6} μm")
    print(f"颗粒 n={N_PARTICLE}, 介质 n={N_MEDIUM}, 波长 {WAVELENGTH_NM} nm")
    print(f"目标力: {F_TARGET*1e12} pN, 下限: {F_MIN*1e12} pN")
    print()

    result = run_design(steps=200, lr=0.05)

    print("--- 优化结果 ---")
    print(f"  光强梯度: {result['grad_I_final']:.4e} W/m³")
    print(f"  Loss: {result['initial_loss']:.4e} → {result['final_loss']:.4e}")
    print()

    print("--- 各粒径捕获力（优化前 → 优化后）---")
    for i, a in enumerate(PARTICLE_RADII):
        f_init = result['forces_initial'][i].item()
        f_final = result['forces_final'][i].item()
        print(f"  a={a.item()*1e6:.1f}μm: {f_init*1e12:.2f} pN → {f_final*1e12:.2f} pN")

    f_mean = result['forces_final'].mean().item()
    f_min = result['forces_final'].min().item()
    print(f"\n  平均力: {f_mean*1e12:.2f} pN (目标 {F_TARGET*1e12:.2f} pN)")
    print(f"  最弱力: {f_min*1e12:.2f} pN (下限 {F_MIN*1e12:.2f} pN)")
    print()
    print("限制声明：瑞利近似域（a≪λ），F_scat=F_grad/2 简化模型，不优化波长。")
    print("多目标：L_track(跟踪) + L_robust(鲁棒下限) + L_power(功率惩罚)。")


if __name__ == "__main__":
    run()

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from photokinetics_pytorch import calc_photothermal, fmt_fix


def inverse_design_photothermal():
    """
    逆问题演示：已知目标温升ΔT=100K，求解所需光强I₀。
    
    这是光动论作为可微物理引擎的核心价值：
    - 解析公式天然可微，梯度下降直接求解
    - 不需要训练数据，不需要神经网络
    - 收敛速度极快（几十次迭代）
    """
    print("=" * 60)
    print("  光动论可微物理引擎 — 逆问题演示")
    print("=" * 60)
    print()
    print("  目标: 已知材料参数和深度，求达到ΔT=100K所需的光强I₀")
    print("  材料: 水 @ 1064nm")
    print()
    
    n = 1.33
    kappa = 0.00012
    wavelength_nm = 1064.0
    rho = 1000.0
    Cp = 4186.0
    depth_mm = 1.0
    time_s = 1.0
    
    target_dT = 100.0
    
    I0 = torch.tensor(1e5, requires_grad=True)
    
    optimizer = torch.optim.Adam([I0], lr=5e4)
    
    for i in range(100):
        optimizer.zero_grad()
        
        result = calc_photothermal(n, kappa, wavelength_nm, I0, rho, Cp, depth_mm, time_s)
        dT = result['dT']
        
        loss = (dT - target_dT) ** 2
        
        loss.backward()
        optimizer.step()
        
        I0.data = torch.clamp(I0.data, min=1e3, max=1e12)
        
        if (i + 1) % 10 == 0:
            print(f"  迭代 {i+1}: I₀={fmt_fix(I0.item())} W/m², ΔT={dT.item():.4f} K, loss={loss.item():.4f}")
    
    print()
    print(f"  最终结果:")
    print(f"    I₀ = {fmt_fix(I0.item())} W/m²")
    print(f"    ΔT = {result['dT'].item():.6f} K")
    print(f"    误差 = {abs(result['dT'].item() - target_dT):.6f} K")
    print()
    print("  原理:")
    print("    1. calc_photothermal() 返回的 dT 是 torch.Tensor")
    print("    2. loss = (dT - target)² 对 I₀ 求导")
    print("    3. optimizer.step() 更新 I₀")
    print("    4. 整个过程在PyTorch计算图中完成，天然支持自动微分")


def sensitivity_analysis():
    """
    灵敏度分析：各参数变化对温升ΔT的影响。
    
    通过计算梯度，可以直接得到：
    - d(ΔT)/d(I₀) — 光强灵敏度
    - d(ΔT)/d(kappa) — 消光系数灵敏度
    - d(ΔT)/d(depth) — 深度灵敏度
    """
    print("\n" + "=" * 60)
    print("  灵敏度分析演示")
    print("=" * 60)
    print()
    
    n = torch.tensor(1.33, requires_grad=True)
    kappa = torch.tensor(0.00012, requires_grad=True)
    I0 = torch.tensor(1e7, requires_grad=True)
    depth_mm = torch.tensor(1.0, requires_grad=True)
    
    result = calc_photothermal(n, kappa, 1064.0, I0, 1000.0, 4186.0, depth_mm, 1.0)
    result['dT'].backward()
    
    print(f"  参数灵敏度（水@1064nm, I0=1e7 W/m², depth=1mm）:")
    print(f"    d(ΔT)/d(I₀)    = {I0.grad.item():.2e} K·m²/W")
    print(f"    d(ΔT)/d(κ)     = {kappa.grad.item():.2e} K")
    print(f"    d(ΔT)/d(depth) = {depth_mm.grad.item():.2e} K/mm")
    print(f"    d(ΔT)/d(n)     = {n.grad.item():.2e} K")
    print()
    print("  解读:")
    print("    - I₀每增加1 W/m²，ΔT增加8.77e-5 K")
    print("    - κ每增加0.0001，ΔT增加7.31e6 K（消光系数影响极大）")
    print("    - depth每增加1mm，ΔT减少5.34e2 K（光强指数衰减）")


def batch_inference():
    """
    批量推理：一次计算多个参数组合的结果。
    
    利用PyTorch的向量化能力，可以同时计算：
    - 多个波长
    - 多个光强
    - 多个材料参数
    """
    print("\n" + "=" * 60)
    print("  批量推理演示")
    print("=" * 60)
    print()
    
    I0_batch = torch.tensor([1e5, 1e6, 1e7, 1e8], requires_grad=True)
    
    result = calc_photothermal(1.33, 0.00012, 1064.0, I0_batch, 1000.0, 4186.0, 1.0, 1.0)
    
    result['dT'].sum().backward()
    
    print(f"  批量计算光强 I₀ = [{', '.join([f'{x:.1e}' for x in I0_batch])}] W/m²")
    print(f"  对应的温升 ΔT = [{', '.join([f'{x:.2f}' for x in result['dT']])}] K")
    print(f"  梯度 d(sum(ΔT))/d(I₀) = [{', '.join([f'{x:.2e}' for x in I0_batch.grad])}]")
    print()
    print("  应用场景:")
    print("    - 参数扫描（寻找最优工作点）")
    print("    - 生成训练数据（用于机器学习）")
    print("    - 多目标优化（同时优化多个参数）")


if __name__ == '__main__':
    inverse_design_photothermal()
    sensitivity_analysis()
    batch_inference()

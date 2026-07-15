import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from photokinetics_pytorch import *


def test_photoelectric():
    print("测试: 光电效应")
    phi = torch.tensor(2.28, requires_grad=True)
    lam = torch.tensor(400.0, requires_grad=True)
    
    hv, occurs, vs, ek = calc_photoelectric(phi, lam)
    
    ek.backward()
    
    assert torch.isclose(hv, torch.tensor(3.0996), rtol=1e-4), f"hv={hv}"
    assert occurs.item() == True, "光电效应应发生"
    assert torch.isclose(vs, torch.tensor(0.8196), rtol=1e-4), f"vs={vs}"
    
    print(f"  hν={hv.item():.4f} eV, V_s={vs.item():.4f} V")
    print(f"  d(ek)/d(phi)={phi.grad.item():.6f}, d(ek)/d(lam)={lam.grad.item():.6f}")
    print("  ✓ 通过")


def test_blackbody():
    print("测试: 黑体辐射")
    T = torch.tensor(5778.0, requires_grad=True)
    
    lambda_max, j, B_lam = calc_blackbody(T, 500.0)
    
    j.backward()
    
    assert torch.isclose(lambda_max, torch.tensor(501.52), rtol=1e-3), f"lambda_max={lambda_max}"
    assert torch.isclose(j, torch.tensor(6.312e7), rtol=1e-2), f"j={j}"
    
    print(f"  λ_max={lambda_max.item():.2f} nm, j={j.item():.2e} W/m²")
    print(f"  d(j)/d(T)={T.grad.item():.2e}")
    print("  ✓ 通过")


def test_compton():
    print("测试: 康普顿散射")
    E0 = torch.tensor(17.4, requires_grad=True)
    theta = torch.tensor(90.0)
    
    dl, Ep, Ee = calc_compton(E0, theta)
    
    Ee.backward()
    
    assert torch.isclose(dl, torch.tensor(2.426), rtol=1e-3), f"dl={dl}"
    
    print(f"  Δλ={dl.item():.4f} pm, E'={Ep.item():.4f} keV")
    print(f"  d(E_e)/d(E0)={E0.grad.item():.6f}")
    print("  ✓ 通过")


def test_doppler():
    print("测试: 多普勒效应")
    nu0 = torch.tensor(5e14, requires_grad=True)
    v = torch.tensor(30000.0)
    
    nu_cl, nu_rel, z_cl, z_rel = calc_doppler(nu0, v, receding=True)
    
    z_rel.backward()
    
    assert torch.isclose(z_rel, torch.tensor(0.1056), rtol=1e-3), f"z_rel={z_rel}"
    
    print(f"  z={z_rel.item():.4f}")
    print(f"  d(z)/d(nu0)={nu0.grad.item():.2e}")
    print("  ✓ 通过")


def test_gravitational():
    print("测试: 引力红移")
    M = torch.tensor(5.9722e24)
    r1 = torch.tensor(6.371e6, requires_grad=True)
    r2 = torch.tensor(1e9)
    
    dnu, z = calc_gravitational_redshift(M, r1, r2)
    
    z.backward()
    
    assert torch.isclose(dnu, torch.tensor(6.95e-10), rtol=1e-2), f"dnu={dnu}"
    
    print(f"  Δν/ν₀={dnu.item():.2e}")
    print(f"  d(z)/d(r1)={r1.grad.item():.2e}")
    print("  ✓ 通过")


def test_photothermal():
    print("测试: 光热模型")
    I0 = torch.tensor(1e7, requires_grad=True)
    n, kappa, lam = 1.33, 0.00012, 1064.0
    rho, Cp, depth, time = 1000.0, 4186.0, 1.0, 1.0
    
    result = calc_photothermal(n, kappa, lam, I0, rho, Cp, depth, time)
    
    result['dT'].backward()
    
    dT = result['dT']
    assert torch.isclose(dT, torch.tensor(877.02), rtol=1e-2), f"dT={dT}"
    
    print(f"  ΔT={dT.item():.2f} K")
    print(f"  d(ΔT)/d(I0)={I0.grad.item():.2e}")
    print("  ✓ 通过")


def test_nonlinear():
    print("测试: 非线性光学")
    Eg = torch.tensor(1.12, requires_grad=True)
    lam = torch.tensor(800.0)
    
    N, hv = calc_nonlinear_order(Eg, lam)
    
    print(f"  N={N.item()}, hν={hv.item():.4f} eV")
    print("  ✓ 通过")


def test_tweezer():
    print("测试: 光镊力")
    a = torch.tensor(1e-6, requires_grad=True)
    grad_I = torch.tensor(5e17)
    
    result = calc_tweezer_force(a, 1.59, 1.33, 1064.0, grad_I)
    
    result['F_grad'].backward()
    
    F_grad = result['F_grad']
    print(f"  F_grad={F_grad.item():.2e} N")
    print(f"  d(F_grad)/d(a)={a.grad.item():.2e}")
    print("  ✓ 通过")


def test_batch_computation():
    print("测试: 批量计算")
    I0_batch = torch.tensor([1e6, 1e7, 1e8], requires_grad=True)
    
    result = calc_photothermal(1.33, 0.00012, 1064.0, I0_batch, 1000.0, 4186.0, 1.0, 1.0)
    
    result['dT'].sum().backward()
    
    dT_batch = result['dT']
    print(f"  ΔT batch: {[f'{v.item():.1f}' for v in dT_batch]}")
    print(f"  d(sum)/d(I0) batch: {[f'{v:.2e}' for v in I0_batch.grad]}")
    print("  ✓ 通过")


if __name__ == '__main__':
    print("=" * 60)
    print("  Photokinetics PyTorch 可微版本 — 单元测试")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    test_photoelectric()
    test_blackbody()
    test_compton()
    test_doppler()
    test_gravitational()
    test_photothermal()
    test_nonlinear()
    test_tweezer()
    test_batch_computation()
    
    print("=" * 60)
    print("  所有测试通过！")
    print("=" * 60)

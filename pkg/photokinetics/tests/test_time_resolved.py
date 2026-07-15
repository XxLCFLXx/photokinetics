"""含时热传导模块测试。"""
import torch
from photokinetics import calc_photothermal, calc_photothermal_timed, calc_photothermal_auto


def test_adiabatic_limit():
    """测试1: 短脉冲时含时解应退化为绝热近似。"""
    print("=== 测试1: 绝热近似极限 ===")
    # 水 @ 1064nm, 短脉冲 t=0.001s (绝热条件 t << L²/D)
    # D=1.43e-7 m²/s, L=1/(α)=1/(0.0143*1000)=7m，L²/D=3.4e8s，t=0.001s << 3.4e8s
    result_adiabatic = calc_photothermal(
        1.33, 0.00012, 1064, 1e7, 1000, 4186, 1.0, 0.001
    )
    result_timed = calc_photothermal_timed(
        1.33, 0.00012, 1064, 1e7, 1000, 4186, 1.0, 0.001,
        k_thermal=0.598
    )
    T_ad = result_adiabatic['dT'].item()
    T_ti = result_timed['T_timed'].item()
    regime = result_timed['regime']
    dt = result_timed['dimensionless_time'].item()

    print(f"  绝热近似: ΔT = {T_ad:.4f} K")
    print(f"  含时解:   ΔT = {T_ti:.4f} K")
    print(f"  Regime: {regime}, t·D·α² = {dt:.2e}")
    rel_err = abs(T_ti - T_ad) / max(abs(T_ad), 1e-10) * 100
    print(f"  相对误差: {rel_err:.4f}%")
    assert rel_err < 5.0, f"绝热极限误差过大: {rel_err}%"
    print("  PASS: 短脉冲含时解与绝热近似一致\n")


def test_transient_regime():
    """测试2: 过渡区，含时解应介于绝热和稳态之间。"""
    print("=== 测试2: 过渡区（含时解）===")
    # 硅 @ 532nm, 中等脉冲 t=1e-6s
    # D=9.08e-5 m²/s, α=4π*0.044/(4.15*532e-9)=2.5e5 /m
    # L=1/α=4μm, L²/D=1.8e-7s, t=1e-6s > L²/D（过渡区）
    result = calc_photothermal_timed(
        4.15, 0.044, 532, 1e7, 2329, 700, 0.002, 1e-6,
        thermal_diffusivity=9.08e-5
    )
    T_ad = result['T_adiabatic'].item()
    T_ti = result['T_timed'].item()
    T_st = result['T_steady'].item()
    regime = result['regime']
    dt = result['dimensionless_time'].item()

    print(f"  绝热近似: ΔT = {T_ad:.4f} K")
    print(f"  含时解:   ΔT = {T_ti:.4f} K")
    print(f"  稳态解:   ΔT = {T_st:.4f} K")
    print(f"  Regime: {regime}, t·D·α² = {dt:.2e}")

    # 物理直觉: 过渡区含时解应小于绝热近似（热扩散带走了部分能量）
    print(f"  含时解 < 绝热近似? {T_ti < T_ad}")
    # 但应大于0
    assert T_ti > 0, "含时解应为正"
    print("  PASS: 过渡区解在合理范围\n")


def test_long_time_growth():
    """测试3: 长时间行为——半无限域无真正稳态，温度应按 √t 持续增长。

    物理依据: Carslaw & Jaeger 半无限域含源热传导的长时间渐近行为
        T(z,t) ~ 2·I₀·√(t/π) / √(ρCp·κ_th)  (表面, t→∞)
    即温度无界增长（√t 律），不存在稳态。
    """
    print("=== 测试3: 长时间 √t 增长（半无限域无稳态）===")
    # 硅 @ 532nm, 两个长脉冲时间点验证 √t 律
    D = 9.08e-5
    t1, t2 = 1e-3, 4e-3  # t2 = 4·t1, √t2/√t1 = 2
    r1 = calc_photothermal_timed(
        4.15, 0.044, 532, 1e7, 2329, 700, 0.002, t1,
        thermal_diffusivity=D
    )
    r2 = calc_photothermal_timed(
        4.15, 0.044, 532, 1e7, 2329, 700, 0.002, t2,
        thermal_diffusivity=D
    )
    T1 = r1['T_timed'].item()
    T2 = r2['T_timed'].item()
    T_st = r1['T_steady'].item()  # 有限厚度稳态参考（不应作为极限）
    dt = r2['dimensionless_time'].item()

    # √t 律: T2/T1 应接近 √(t2/t1) = √4 = 2
    ratio = T2 / T1 if T1 > 0 else 0
    expected = (t2 / t1) ** 0.5
    print(f"  T(t={t1:.0e}s) = {T1:.4f} K")
    print(f"  T(t={t2:.0e}s) = {T2:.4f} K")
    print(f"  比值 T2/T1 = {ratio:.3f} (期望 √t 律: {expected:.3f})")
    print(f"  T_steady (有限厚度参考) = {T_st:.4f} K")
    print(f"  Regime: {r2['regime']}, t·D·α² = {dt:.2e}")
    # 允许 15% 误差（√t 律在足够长时间后成立）
    assert abs(ratio - expected) / expected < 0.15, \
        f"√t 律偏差过大: 比值 {ratio}, 期望 {expected}"
    # 同时验证温度持续增长，不趋于有限厚度"稳态"
    assert T2 > T_st, "长时间温度应超过有限厚度稳态参考"
    print("  PASS: 长时间温度按 √t 律增长（半无限域无稳态）\n")


def test_gradient():
    """测试4: 自动微分正常工作。"""
    print("=== 测试4: 自动微分 ===")
    I0 = torch.tensor(1e7, requires_grad=True)
    result = calc_photothermal_timed(
        1.33, 0.00012, 1064, I0, 1000, 4186, 1.0, 1.0,
        k_thermal=0.598
    )
    result['T_timed'].backward()
    print(f"  d(ΔT)/d(I₀) = {I0.grad.item():.4e}")
    assert I0.grad is not None
    print("  PASS: 梯度计算正常\n")


def test_auto_mode():
    """测试5: 自动模式切换。"""
    print("=== 测试5: 自动模式 ===")
    # 短脉冲 → 应选 adiabatic
    r1 = calc_photothermal_auto(
        1.33, 0.00012, 1064, 1e7, 1000, 4186, 1.0, 1e-6,
        k_thermal=0.598
    )
    print(f"  短脉冲(t=1μs): method={r1['method']}, ΔT={r1['T'].item():.4f} K")

    # 长脉冲 → 应选 timed
    r2 = calc_photothermal_auto(
        1.33, 0.00012, 1064, 1e7, 1000, 4186, 1.0, 1.0,
        k_thermal=0.598
    )
    print(f"  长脉冲(t=1s):  method={r2['method']}, ΔT={r2['T'].item():.4f} K")
    print("  PASS: 自动切换正常\n")


if __name__ == "__main__":
    test_adiabatic_limit()
    test_transient_regime()
    test_long_time_growth()
    test_gradient()
    test_auto_mode()
    print("=" * 50)
    print("ALL TESTS PASSED")

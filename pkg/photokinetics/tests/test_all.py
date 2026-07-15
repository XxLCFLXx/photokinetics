"""Photokinetics 单元测试。"""
import torch
from photokinetics import (
    calc_photoelectric,
    calc_blackbody,
    calc_compton,
    calc_doppler,
    calc_gravitational_redshift,
    calc_photothermal,
    calc_nonlinear_order,
    calc_tweezer_force,
)


def test_photoelectric():
    phi = torch.tensor(2.28, requires_grad=True)
    lam = torch.tensor(400.0, requires_grad=True)
    hv, occurs, vs, ek = calc_photoelectric(phi, lam)
    ek.backward()
    assert torch.isclose(hv, torch.tensor(3.0996), rtol=1e-4)
    assert occurs.item() == True
    assert torch.isclose(vs, torch.tensor(0.8196), rtol=1e-4)


def test_blackbody():
    T = torch.tensor(5778.0, requires_grad=True)
    lambda_max, j, _ = calc_blackbody(T, 500.0)
    j.backward()
    assert torch.isclose(lambda_max, torch.tensor(501.52), rtol=1e-3)
    assert torch.isclose(j, torch.tensor(6.312e7), rtol=1e-2)


def test_compton():
    E0 = torch.tensor(17.4, requires_grad=True)
    dl, Ep, Ee = calc_compton(E0, 90.0)
    Ee.backward()
    assert torch.isclose(dl, torch.tensor(2.426), rtol=1e-3)


def test_doppler():
    nu0 = torch.tensor(5e14, requires_grad=True)
    _, _, _, z_rel = calc_doppler(nu0, 30000.0, receding=True)
    z_rel.backward()
    assert torch.isclose(z_rel, torch.tensor(0.1056), rtol=1e-3)


def test_gravitational():
    r1 = torch.tensor(6.371e6, requires_grad=True)
    dnu, z = calc_gravitational_redshift(5.9722e24, r1, 1e9)
    z.backward()
    assert torch.isclose(dnu, torch.tensor(6.95e-10), rtol=1e-2)


def test_photothermal():
    I0 = torch.tensor(1e7, requires_grad=True)
    result = calc_photothermal(1.33, 0.00012, 1064, I0, 1000, 4186, 1.0, 1.0)
    result['dT'].backward()
    assert torch.isclose(result['dT'], torch.tensor(877.02), rtol=1e-2)


def test_nonlinear():
    N, hv = calc_nonlinear_order(1.12, 800.0)
    assert N.item() == 1


def test_tweezer():
    a = torch.tensor(1e-6, requires_grad=True)
    result = calc_tweezer_force(a, 1.59, 1.33, 1064.0, 5e17)
    result['F_grad'].backward()
    assert a.grad is not None


def test_batch():
    I0_batch = torch.tensor([1e6, 1e7, 1e8], requires_grad=True)
    result = calc_photothermal(1.33, 0.00012, 1064, I0_batch, 1000, 4186, 1.0, 1.0)
    result['dT'].sum().backward()
    assert I0_batch.grad is not None

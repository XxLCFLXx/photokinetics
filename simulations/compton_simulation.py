import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.stats import gaussian_kde

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 孤子剖面 (求解 ϕ⁴ 模型的径向方程) ==========
def soliton_profile(v=1.0, lam=1.0, r_max=10, n_points=1000):
    """
    求解球对称静态孤子方程:
    d²ρ/dr² + (2/r) dρ/dr = λ ρ (ρ² - v²)
    边界条件: dρ/dr(0)=0, ρ(∞) = -v
    """
    def ode(r, y):
        rho, drho = y
        # 使用np.where处理边界条件，避免数组判断问题
        d2rho = np.where(r > 1e-8, 
                        lam * rho * (rho**2 - v**2) - (2/r) * drho,
                        lam * rho * (rho**2 - v**2))
        return np.vstack((drho, d2rho))

    def bc(ya, yb):
        # ya[1] = dρ/dr at r=0, yb[0] = ρ(∞) + v should be 0
        return np.array([ya[1], yb[0] + v])

    r = np.linspace(1e-8, r_max, n_points)
    # 初始猜测: 从 +v 到 -v 的 tanh 形状
    rho_guess = v * np.tanh(-r)   # 近似解
    drho_guess = -v / np.cosh(r)**2
    sol = solve_bvp(ode, bc, r, np.vstack((rho_guess, drho_guess)), tol=1e-8)
    return sol.x, sol.y[0]

# 生成孤子剖面图
r, rho = soliton_profile()
plt.figure(figsize=(4, 3))
plt.plot(r, rho, 'b-', lw=2, label=r'$\rho(r)$')
plt.axhline(-1, ls='--', lw=1, color='gray', alpha=0.7, label=r'vacuum $-v$')
plt.xlabel(r'Radial distance $r$', fontsize=12)
plt.ylabel(r'$\rho(r)$', fontsize=12)
plt.title('Soliton profile', fontsize=12)
plt.xlim(0, 8)
plt.ylim(-1.2, 1.2)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('soliton_profile.png', dpi=300)
plt.close()

# ========== 2. 费米子基态概率密度 (解析近似: 高斯型) ==========
def fermion_density(r, sigma=0.5, norm=1.0):
    """归一化的高斯概率密度"""
    return norm / (sigma * np.sqrt(2*np.pi)) * np.exp(-r**2 / (2*sigma**2))

r_psi = np.linspace(0, 3, 500)
psi2 = fermion_density(r_psi, sigma=0.6, norm=1.0)  # sigma 可调，模拟束缚态宽度

plt.figure(figsize=(4, 3))
plt.plot(r_psi, psi2, 'r-', lw=2, label=r'$|\psi(r)|^2$')
plt.fill_between(r_psi, psi2, alpha=0.3, color='red')
plt.xlabel(r'Radial distance $r$', fontsize=12)
plt.ylabel(r'$|\psi(r)|^2$', fontsize=12)
plt.title('Fermion bound state probability density', fontsize=12)
plt.xlim(0, 2.5)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('fermion_density.png', dpi=300)
plt.close()

# ========== 3. 康普顿散射电子反冲动能分布 (蒙特卡洛) ==========
# 物理常数
h = 6.626e-34    # J·s
c = 3e8          # m/s
m_e = 9.109e-31  # kg
eV = 1.602e-19   # J/eV
m_e_eV = m_e * c**2 / eV  # 511 keV

def compton_recoil_energy(E0_eV, theta):
    """
    给定入射光子能量 E0 (eV) 和散射角 theta (rad)，
    返回电子反冲动能 (eV)
    """
    E0 = E0_eV
    m = m_e_eV
    E_gamma = E0 / (1 + (E0/m) * (1 - np.cos(theta)))
    return E0 - E_gamma

# 模拟参数
E_photon = 100e3      # 100 keV 入射光子
N_samples = 50000
# 各向同性散射: cos(theta) 在 [-1,1] 均匀分布
theta = np.arccos(np.random.uniform(-1, 1, N_samples))
E_recoil = compton_recoil_energy(E_photon, theta)

plt.figure(figsize=(5, 3))
plt.hist(E_recoil, bins=80, density=True, color='green', alpha=0.7, edgecolor='black', lw=0.5)
plt.xlabel('Electron recoil kinetic energy (eV)', fontsize=12)
plt.ylabel('Probability density', fontsize=12)
plt.title(f'Compton scattering (E$_{{γ}}$ = {E_photon/1000:.0f} keV)', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('compton_recoil.png', dpi=300)
plt.close()

# ========== 可选: 合并三张图为一张 (用于论文主图) ==========
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

# 孤子剖面
axes[0].plot(r, rho, 'b-', lw=2)
axes[0].axhline(-1, ls='--', lw=1, color='gray')
axes[0].set_xlabel(r'$r$')
axes[0].set_ylabel(r'$\rho(r)$')
axes[0].set_title('(a) Soliton profile')
axes[0].set_xlim(0, 8)
axes[0].grid(alpha=0.3)

# 费米子密度
axes[1].plot(r_psi, psi2, 'r-', lw=2)
axes[1].fill_between(r_psi, psi2, alpha=0.3, color='red')
axes[1].set_xlabel(r'$r$')
axes[1].set_ylabel(r'$|\psi(r)|^2$')
axes[1].set_title('(b) Fermion bound state')
axes[1].set_xlim(0, 2.5)
axes[1].grid(alpha=0.3)

# 康普顿分布
axes[2].hist(E_recoil, bins=80, density=True, color='green', alpha=0.7, edgecolor='black', lw=0.5)
axes[2].set_xlabel('Electron recoil energy (eV)')
axes[2].set_ylabel('Probability density')
axes[2].set_title('(c) Compton recoil spectrum')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_combined.png', dpi=300)
plt.close()
print("所有图片已生成：soliton_profile.png, fermion_density.png, compton_recoil.png, fig_combined.png")

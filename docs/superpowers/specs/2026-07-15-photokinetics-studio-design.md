# 设计规格：Photokinetics Studio 实战案例产品原型

**日期**：2026-07-15
**状态**：已与用户确认（通过可视化伴侣逐步确认三层架构 + 三个案例流程）
**目标用户**：科研人员 + 工业研发
**交付层级**：完整产品原型（代码 + 测试 + 文章 + 交互式 Web 入口）

---

## 1. 目标与范围

### 1.1 目标

构建一个完整的可微物理实战产品原型，通过三个真实场景展示 `photokinetics v2.1` 包的独特价值：**可微分物理引擎能够被 AI 优化算法直接调用解决逆问题**。

产品同时服务于：
- **科研人员**：参数反演、实验拟合、可复现研究
- **工业研发**：激光加工参数设计、仪器标定、批量评估

### 1.2 范围（In Scope）

| 项 | 内容 |
|---|---|
| 三个 Python 案例脚本 | 每个含合成数据生成、优化循环、结果验收 |
| 三个回归测试 | 每个案例的梯度可用性、loss 下降、参数恢复 |
| 共享工具层 | `examples/_common.py`：正值参数、相对损失、对数损失、优化循环 |
| 交互式 Web 入口 | 单页应用，三案例切换，参数面板 + 结果可视化 |
| 文章 | 三篇知乎/README 可发布的讲解文章 |
| 数据导出 | CSV/JSON 结果 + matplotlib 图表 |

### 1.3 首版明确限制（Out of Scope）

- **CPU/float32 运行**：不声明支持 GPU 和 float64 高精度
- **合成数据**：不依赖外部实验数据文件，用固定种子合成
- **瑞利近似域**：光镊案例粒径 ≤ 1μm@1064nm，不优化波长
- **单机部署**：Web 入口为本地静态站点，不做云端后端

---

## 2. 整体架构

### 2.1 三层架构

```
LAYER 1 · 统一入口
    Photokinetics Studio（单页 Web App）
    - 三案例切换
    - 共享参数面板与结果导出
    - 本地静态站点（HTML/JS，可选 Flask 单文件后端）

LAYER 2 · 三个实战案例
    案例1: 含时光热参数反演（科研核心，承接 v2.1）
    案例2: 黑体光谱测温与仪器标定（仪器标定）
    案例3: 光镊鲁棒粒径设计（工业设计，多目标优化）

LAYER 3 · 共享基础设施（复用 v2.1）
    photokinetics 包 API:
        - calc_photothermal_timed
        - calc_blackbody
        - calc_tweezer_force
    examples/_common.py:
        - positive_parameter
        - relative_mse / log_mse
        - optimize 通用循环
    可视化与导出:
        - matplotlib 图表
        - CSV/JSON 数据导出
        - 优化轨迹记录
```

### 2.2 文件落位

```
photokinetics/pkg/photokinetics/
├── examples/
│   ├── _common.py                       # 共享工具层
│   ├── fit_transient_photothermal.py    # 案例1
│   ├── fit_blackbody_temperature.py     # 案例2
│   └── design_optical_tweezer.py        # 案例3
├── tests/
│   ├── test_inverse_photothermal.py     # 案例1测试
│   ├── test_blackbody_inversion.py      # 案例2测试
│   └── test_tweezer_design.py           # 案例3测试
└── docs/
    └── case_studies/                    # 三篇文章
        ├── 01_photothermal_inversion.md
        ├── 02_blackbody_thermometry.md
        └── 03_tweezer_robust_design.md

photokinetics/studio/                    # 交互式 Web 入口
├── index.html
├── app.js
├── style.css
└── server.py                            # 可选 Flask 单文件后端
```

---

## 3. 共享工具层设计（examples/_common.py）

### 3.1 正值参数化

```python
def positive_parameter(raw, scale=1.0, floor=0.0):
    """保证参数始终为正，通过 softplus 变换。"""
    return floor + scale * torch.nn.functional.softplus(raw)
```

**理由**：直接操作 `.data` 会绕过 autograd，应通过参数化保证物理约束。

### 3.2 损失函数

```python
def relative_mse(prediction, target, eps=1e-12):
    """相对误差 MSE，避免高值主导。"""
    scale = torch.clamp(torch.abs(target), min=eps)
    return torch.mean(((prediction - target) / scale) ** 2)

def log_mse(prediction, target, eps=1e-30):
    """对数域 MSE，适用于跨数量级数据（如 Planck 光谱）。"""
    return torch.mean((torch.log(prediction + eps) - torch.log(target + eps)) ** 2)
```

### 3.3 通用优化循环

```python
def optimize(parameters, closure, steps, lr, callback=None):
    """Adam 优化循环，记录每步 loss。"""
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = closure()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
        if callback is not None:
            callback(step, loss)
    return history
```

### 3.4 数据导出

```python
def export_results(filepath, **kwargs):
    """将结果导出为 JSON（标量）+ CSV（数组）。"""
    ...
```

---

## 4. 案例 1：含时光热参数反演

### 4.1 实战问题

给定多个深度、多个时刻测得的温升曲线 T(z,t)，联合反演：
- 消光系数 `kappa`
- 热扩散率 `D`

### 4.2 物理场景

| 参数 | 值 | 说明 |
|---|---|---|
| 材料 | 硅 @ 532nm | α≈2.5×10⁵/m，强吸收 |
| κ_true | 0.044 | 真实消光系数 |
| D_true | 9.08e-5 m²/s | 真实热扩散率 |
| 深度网格 | z ∈ [0, 10] μm | 10 点对数分布 |
| 时间网格 | t ∈ [1ns, 1ms] | 10 点对数分布，跨 6 个数量级 |
| 噪声 | 1% 高斯 | seed=42 |

**物理意义**：时间跨度覆盖绝热→过渡→长时区域，充分展示 v2.1 含时解价值。

### 4.3 工作流

1. **合成观测数据**：用 κ_true, D_true 调用 `calc_photothermal_timed` 生成 T(z,t)，加 1% 噪声
2. **参数初始化**：`softplus(raw)` 保证正值，从错误初值开始（κ_init = 10×κ_true）
3. **Adam 优化**：
   - 前向：`T_pred = calc_photothermal_timed(kappa, D, ...)["T_timed"]`
   - 损失：`L = relative_mse(T_pred, T_obs)`
   - 500 步，lr=0.05，记录每步 loss/κ/D 轨迹
4. **结果验收**：参数恢复误差、拟合曲线对比、优化轨迹图、CSV/JSON 导出

### 4.4 关键约束

- **必须用 `calc_photothermal_timed`**，不用 `calc_photothermal_auto`（后者批量布尔问题 + 离散切换不利于优化）
- **可辨识性教学点**：κ 和 D 在短时区近似简并（绝热区 T∝α·t，与 D 无关），需用过渡区数据才能联合反演

### 4.5 验收标准

| 条件 | 阈值 |
|---|---|
| 梯度可用 | `parameter.grad is not None` 且 finite |
| Loss 下降 | 500 步后 final_loss < initial_loss |
| 无噪声参数恢复 | κ、D 相对误差 < 5% |
| 1% 噪声参数恢复 | κ、D 相对误差 < 10% |

---

## 5. 案例 2：黑体光谱测温与仪器标定

### 5.1 实战问题

根据多波长通道测得的辐射强度，反演：
- 黑体温度 `T`
- 仪器增益 `gain`

### 5.2 物理场景

| 参数 | 值 | 说明 |
|---|---|---|
| 场景 | 太阳光谱 | 备选：工业钢水 300-2000℃ |
| T_true | 5800 K | 真实温度 |
| gain_true | 1.0 | 真实增益 |
| 波长范围 | 400-2500 nm | 可见光到近红外 |
| 通道数 | 32 | 均匀采样 |
| 噪声 | 2% 高斯 | seed=42 |

### 5.3 工作流

1. **合成观测光谱**：`I_obs = gain_true * calc_blackbody(T_true, wavelengths)[2]` + 2% 噪声
2. **参数初始化**：
   - `T = 300 + softplus(raw_T) * 5000`（范围 300-5300K）
   - `gain = softplus(raw_gain)`
3. **对数域 Adam 优化**：
   - 前向：`_, _, B = calc_blackbody(T, wavelengths)` → `pred = gain * B`
   - 损失：`L = log_mse(pred, I_obs)`（Planck 谱跨 3-4 数量级）
   - 300 步，lr=0.1
4. **标定结果**：温度反演误差、光谱拟合曲线、维恩位移验证、通道敏感性分析

### 5.4 关键约束

- **对数损失必要**：Planck 光谱跨数量级，直接 MSE 让高值主导
- **可辨识性教学点**：T 和 gain 同时自由时简并，首版固定 gain 或加温度锚点
- **黑体高指数区截断**：`blackbody.py` 把指数截到 500，避免极端短波/低温区

### 5.5 验收标准

| 条件 | 阈值 |
|---|---|
| 梯度可用 | `parameter.grad is not None` 且 finite |
| Loss 下降 | 300 步后 final_loss < initial_loss |
| 无噪声温度恢复 | 误差 < 1% |
| 2% 噪声温度恢复 | 误差 < 3% |

---

## 6. 案例 3：光镊鲁棒粒径设计

### 6.1 实战问题

为不同粒径分布的微粒设计光强梯度，使：
- 平均捕获力达到目标值
- 最弱粒径下的捕获力不低于下限
- 光强梯度（功率消耗）尽量小

### 6.2 物理场景

| 参数 | 值 | 说明 |
|---|---|---|
| 粒径分布 | [0.7, 0.8, 0.9, 1.0] μm | 4 种微粒 |
| 颗粒折射率 | 1.59 | 聚苯乙烯 |
| 介质折射率 | 1.33 | 水 |
| 波长 | 1064 nm | 红外光镊 |
| 目标力 | 10 pN | 平均捕获力 |
| 下限力 | 5 pN | 最弱粒径下限 |

### 6.3 工作流

1. **定义粒径分布与目标**：粒径批次张量化
2. **多目标损失函数**：
   ```python
   forces = calc_tweezer_force(a_m, n_p, n_m, λ, grad_I)["F_total"]
   L_track = ((forces.mean() - F_target)/F_target) ** 2      # 跟踪损失
   L_robust = torch.relu((F_min - forces)/F_min).pow(2).mean()  # 鲁棒下限
   L_power = 1e-4 * (grad_I/1e17) ** 2                       # 功率惩罚
   L = L_track + L_robust + L_power
   ```
3. **Adam 优化**：200 步，lr=0.05，张量广播一次计算所有粒径力
4. **设计对比**：单粒径 vs 鲁棒设计、各粒径力柱状图、灵敏度分析

### 6.4 关键约束

- **瑞利近似边界**：a ≪ λ，粒径 ≤1μm@1064nm 满足；**不应宣称优化波长**（当前 wavelength_nm 不参与计算）
- **简化声明**：`F_scat = F_grad / 2` 是简化近似，非完整光镊散射模型
- **grad_I 正值约束**：用 `softplus`

### 6.5 验收标准

| 条件 | 阈值 |
|---|---|
| 梯度可用 | `parameter.grad is not None` 且 finite |
| Loss 下降 | 200 步后 final_loss < initial_loss |
| 目标力误差 | < 5% |
| 最弱粒径力 | ≥ 5 pN 下限 |
| 单粒径 vs 鲁棒差异 | 可视化对比 |

---

## 7. 交互式 Web 入口（Photokinetics Studio）

### 7.1 技术方案

- **前端**：单页 HTML + 原生 JS（不引入框架，保持轻量）
- **后端**：可选 Flask 单文件（`server.py`），提供 Python 执行能力
- **图表**：Chart.js 或 Plotly.js（CDN 引入）
- **部署**：本地静态站点，可选部署到 GitHub Pages

### 7.2 页面结构

```
顶栏：Photokinetics Studio · 案例切换标签（1/2/3）
左栏：参数面板（动态根据案例渲染）
右栏：结果区
    - 上：优化轨迹（loss/参数收敛曲线）
    - 下：拟合结果（观测vs预测/柱状图/参数表）
底栏：导出 CSV / 复制结果 / 运行优化按钮
```

### 7.3 交互流程

1. 用户选择案例标签
2. 左栏渲染该案例的参数输入（材料、初值、噪声、步数、lr）
3. 点击"运行优化"→ 调用后端执行 Python 脚本（或前端预设结果展示）
4. 右栏实时显示优化轨迹和结果
5. 用户可导出 CSV/JSON 或复制结果

### 7.4 首版简化

- 首版可先做**前端预设结果展示**（预先跑好的结果数据嵌入 JS），不依赖后端
- 验证用户对 UI 的接受度后，再补 Flask 后端做真实执行

---

## 8. 文章交付（docs/case_studies/）

每篇文章结构：

1. **问题背景**：为什么这个逆问题重要
2. **传统方法痛点**：为什么传统仿真做不到
3. **可微物理方案**：photokinetics 如何解决
4. **代码 walkthrough**：关键代码段讲解
5. **结果展示**：图表 + 数值
6. **可辨识性/物理洞察**：教学点
7. **如何复现**：`pip install photokinetics && python examples/xxx.py`

三篇文章同时服务于：
- 知乎引流（内容路径）
- GitHub README（开源路径）
- 咨询获客（咨询路径）

---

## 9. 接口风险与规避

| 风险 | 规避措施 |
|---|---|
| device/dtype 不透传（强制 float32） | 首版明确声明 CPU/float32 |
| `calc_photothermal_auto` 批量布尔问题 | 案例1固定用 `calc_photothermal_timed` |
| 黑体高指数区截断 | 选合理波长范围，避免初始化落入饱和区 |
| 光镊 wavelength_nm 不参与计算 | 不宣称优化波长，明确标注 |
| regime 批量输出语义不统一 | 把 regime 当诊断信息，不参与损失 |

---

## 10. 实施顺序

| 优先级 | 任务 | 依赖 |
|---|---|---|
| P0 | `examples/_common.py` 共享工具层 | 无 |
| P1 | 案例1脚本 + 测试 | _common.py |
| P2 | 案例2脚本 + 测试 | _common.py |
| P3 | 案例3脚本 + 测试 | _common.py |
| P4 | Web 入口（前端预设结果版） | 三案例完成 |
| P5 | 三篇文章 | 三案例完成 |
| P6 | 提交 GitHub + 更新 PyPI | 全部完成 |

---

## 11. 成功标准

- [ ] 三个案例脚本可独立运行（`python examples/xxx.py`）
- [ ] 三个测试全部通过（梯度 + loss 下降 + 参数恢复）
- [ ] Web 入口可展示三案例的优化轨迹和结果
- [ ] 三篇文章可发布到知乎
- [ ] 所有代码提交到 GitHub，README 更新案例链接

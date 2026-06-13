# MiniLab-Hand

**MiniLab-Hand** 是一个专注于灵巧手（Sharpa 仿人五指灵巧手）在手操纵（In-Hand Manipulation）任务的高性能强化学习仿真与训练框架。该项目基于物理引擎 MuJoCo，从大型 RL infrastructure 中进行了解耦、重构和性能优化，具备完整的算法训练、跨平台渲染和部署链条。

此项目非常适合展示多进程并行编程、强化学习物理仿真适配、跨平台性能调优以及 Sim-to-Real 部署流程的工程实践能力。

---

## 🌟 核心技术亮点 (Engineering & Algorithmic Highlights)

### 1. 高性能多进程并行架构与高效 IPC 通信
* **APPO (Asynchronous PPO)**：实现了异步近端策略优化算法。将数据收集子进程（Collector）与策略学习主进程（Learner）进行高度解耦与异步并行，避免了同步 PPO 在等待环境 Step 时的硬件空闲。
* **共享内存 IPC**：基于 C++ 扩展和 System-V/POSIX 共享内存机制，设计了高速数据通道 `RolloutRingBuffer` 与权重同步组件 `SharedWeightSync`。在进程间传递海量仿真数据时实现了零拷贝（Zero-Copy）的高速数据共享，极大提升了吞吐效率。

### 2. 跨平台多线程死锁调优与系统适配
* **MPS 硬件加速死锁攻克**：针对 PyTorch 默认的多进程 `spawn` 上下文在 macOS (Metal Performance Shaders, MPS) 下并发访问 GPU 造成的 Metal API 底层死锁问题，进行了精细化硬件调度：将 Collector 数据收集端限制在 `CPU` 上运行，而将 Learner 参数更新端保持在 `MPS` 显卡上，保障了跨平台硬件训练的稳定性与高吞吐。
* **GUI 线程死锁绕过**：针对 macOS Cocoa 框架只允许在主线程进行 UI 渲染的机制限制，引入并集成了 `mjpython` 包装工具，完美解决了仿真窗口多线程启动时卡死/崩溃的问题。

### 3. 非对称 Actor-Critic (Asymmetric AC) 算法实践
* **特权观测信息 (Privileged Info)**：支持 `separated` 观测模式，实现真正的非对称 Actor-Critic。
  * **Actor（策略网络）**：仅接收受限的本体感知状态（Dof Joint States）与当前控制目标。
  * **Critic（价值网络）**：接收额外的**特权信息**，如物体精确的质心位置（COM Offset）、摩擦系数（Friction Scale）、物体真实质量（Mass）等。
  * 这种设计显著降低了灵巧手物体操纵等高度非线性、高自由度控制任务的探索难度，实现快速收敛。

### 4. Sim-to-Real 部署就绪设计
* **开箱即用 ONNX 导出**：项目打通了 `eval` 与 `export` 链条。在策略回放期间自动将 PyTorch 网络模型导出为 `policy.onnx` 计算图，并使用 ONNX Runtime 自动进行等价性数值校验。
* **部署规范兼容**：导出的 ONNX 模型能够无缝集成到真实机器人控制器（如 C++ 部署端）或 Windows/Linux 端的 Isaac Sim 仿真软件中，完美兼容工业界 Sim-to-Real 的部署管线。

---

## 🛠️ 快速开始 (Getting Started)

### 1. 环境准备
本项目推荐使用 `uv` 进行现代化的 Python 依赖管理，一键同步虚拟环境：
```bash
# 激活并同步依赖
uv sync
```

### 2. 算法训练
我们提供了标准同步 PPO 与高效异步 APPO 两种算法的执行入口：

* **运行 PPO 训练 (RSL-RL)**：
  ```bash
  ./.venv/bin/mjpython scripts/train_rsl_rl.py task=sharpa_inhand/mujoco algo.max_iterations=300
  ```
* **运行 APPO 训练 (多进程 CPU Collector + MPS Learner)**：
  ```bash
  ./.venv/bin/mjpython scripts/train_appo.py task=sharpa_inhand/mujoco algo.max_iterations=100 algo.num_envs=2048 training.collector_device=cpu training.no_play=true
  ```

### 3. 策略交互式可视化演示 (Play)
使用 `mjpython` 驱动三维交互式 MuJoCo 渲染器，可以实时查看模型操纵效果：
```bash
# 回放并演示最新的 APPO 训练策略
./.venv/bin/mjpython scripts/play_interactive.py --algo appo --task sharpa_inhand --sim mujoco interactive.action_mode=policy algo.load_run=-1
```

* **界面快捷操作**：
  * `Space`（空格键）：暂停/继续仿真。
  * `鼠标左键拖拽`：旋转视角；`鼠标双击`：聚焦/锁定目标物体。

### 4. 导出部署模型 (ONNX Export)
一键加载训练好的模型检查点并导出 ONNX：
```bash
./.venv/bin/mjpython scripts/train_appo.py task=sharpa_inhand/mujoco training.play_only=true algo.load_run=-1
```
导出的模型将存储在对应日志时间戳文件夹下的 `policy.onnx` 中。

---

## 📂 项目结构概览

* `conf/` - 算法、任务和环境配置系统（支持 Hydra 和 Registry 的灵活插拔式配置）。
* `scripts/` - 组装了 PPO、APPO 训练流和交互式可视化的轻量级脚本。
* `src/unilab/` - 框架核心模块。
  * `algos/` - APPO 异步多进程、RSL-RL PPO 强化学习核心算法实现。
  * `base/` - 基础环境定义与 `SimBackend` 后端统一抽象接口。
  * `envs/` - 灵巧手在手旋转仿真环境的特化逻辑。
  * `ipc/` - 跨进程双通道共享内存缓冲区与数据对齐模块。

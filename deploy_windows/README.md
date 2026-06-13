# 🚀 MiniLab-Hand Windows Isaac Sim 部署与运行极简指南

为了让您能够非常顺畅地在 Windows 上使用 Isaac Sim / Isaac Lab 演示策略，请按照本指南的详细步骤进行操作。

---

## 🔍 第一部分：如何在 Windows 上定位可执行环境

在 Isaac Sim 中运行 Python 脚本，**不能**使用系统默认的普通 Python（如 python.exe），必须使用载入了 Omniverse 环境特有依赖和环境变量的专门解释器。

您可以在您的 Windows 电脑上通过以下两种方式定位到当前可执行的环境：

### 方式 A：通过 Isaac Lab 运行（推荐）
如果您在 Windows 上安装并配置了 Isaac Lab：
1. **寻找 `isaaclab.bat`**：
   * 打开 CMD 或 PowerShell，运行命令：
     ```powershell
     where isaaclab
     ```
     如果输出了一条路径（例如 `C:\Omniverse\IsaacLab\isaaclab.bat`），说明已配置全局环境变量，您可以直接使用 `isaaclab` 命令。
   * 如果 `where` 命令找不到，请在您的电脑中找到您的 Isaac Lab 安装主目录，其根目录下必定有一个名为 `isaaclab.bat` 的批处理脚本。
2. **运行命令**：
   ```powershell
   # 在 deploy_windows 路径下执行：
   path\to\isaaclab.bat -p play_isaac.py
   ```

### 方式 B：通过 Isaac Sim 内置的 `python.bat` 运行
如果您仅安装了 Isaac Sim（未安装/配置 Isaac Lab），或者上述方法不可用：
1. **寻找 `python.bat` 目录**：
   * Isaac Sim 默认由 Omniverse Launcher 写入以下路径：
     `C:\Users\<您的Windows用户名>\AppData\Local\ov\pkg\`
   * 进入该文件夹，您会看到名为 `isaac-sim-4.0.0` 或类似版本的文件夹（例如 `isaac-sim-2023.1.1`）。
   * 该文件夹根目录下包含一个名为 **`python.bat`** 的可执行批处理脚本。
2. **运行命令**：
   ```powershell
   # 在 deploy_windows 路径下执行：
   C:\Users\<您的Windows用户名>\AppData\Local\ov\pkg\isaac-sim-XXXX\python.bat play_isaac.py
   ```

---

## 📝 第二部分：详细迁移与演示步骤

请按照以下步骤逐步完成策略的迁移与演示：

### 1. 导出 ONNX 策略文件（Mac 端操作）
* **PPO 算法**：
  您之前已经完成了 PPO 策略训练并播放过，可以直接在以下本地路径找到已经导出的 `policy.onnx`：
  * **本地文件路径**：[logs/rsl_rl_ppo/SharpaInhandRotation/2026-06-13_15-40-51_mujoco/policy.onnx](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab-Hand/logs/rsl_rl_ppo/SharpaInhandRotation/2026-06-13_15-40-51_mujoco/policy.onnx)
  * **GitHub 地址**：`https://github.com/fireowl-sw/MiniLab-Hand/blob/main/logs/rsl_rl_ppo/SharpaInhandRotation/2026-06-13_15-40-51_mujoco/policy.onnx`
* **APPO 算法**：
  如果您希望演示刚刚跑完的 APPO 策略，需要在 Mac 端生成 ONNX：
  ```bash
  ./.venv/bin/mjpython scripts/train_appo.py task=sharpa_inhand/mujoco training.play_only=true algo.load_run=-1
  ```
  生成后的 `policy.onnx` 会保存在类似如下的文件夹中：
  `logs/appo/SharpaInhandRotation/2026-06-13_16-33-17_mujoco/policy.onnx`

### 2. 准备 Windows 部署目录（Windows 端操作）
1. 在 Windows 电脑上克隆或拉取本仓库的代码。
2. 将刚才生成的 `policy.onnx` 复制，放置到 Windows 本地的 `deploy_windows` 文件夹根目录下。
   * **GitHub 对应脚本地址**：[play_isaac.py](https://github.com/fireowl-sw/MiniLab-Hand/blob/main/deploy_windows/play_isaac.py)
   * **GitHub 对应配置地址**：[deploy_config.yaml](https://github.com/fireowl-sw/MiniLab-Hand/blob/main/deploy_windows/deploy_config.yaml)

### 3. 在 Isaac Sim 中生成机械手 USD 资源文件
1. 打开 Windows 端的 **Isaac Sim** GUI 界面。
2. 菜单栏找到并打开：`Isaac Utils` -> `MJCF Importer`。
3. 选择并加载本文件夹下的手部描述 XML：
   * **本地描述路径**：[deploy_windows/assets/robots/sharpa_wave/right_sharpa_wave.xml](file:///Users/fireowl/Documents/auto_ws/robot_ws/MiniLab-Hand/deploy_windows/assets/robots/sharpa_wave/right_sharpa_wave.xml)
   * **GitHub 对应地址**：`https://github.com/fireowl-sw/MiniLab-Hand/blob/main/deploy_windows/assets/robots/sharpa_wave/right_sharpa_wave.xml`
4. 勾选 **`Create Articulation`**，然后点击 **`Import`**。
5. 导入完成后，在 Stage 树状图中找到关节，确保它们的 Drive 控制模式已设置为 **`Position`**（即位置控制/PD控制器）。
6. 点击菜单栏 `File` -> `Save As`，将该文件在本地保存为：
   `deploy_windows/assets/robots/sharpa_wave/right_sharpa_wave.usd` （放在与 XML 同一目录下）

### 4. 运行演示
打开 Windows 终端（CMD 或 PowerShell），进入到 `deploy_windows` 目录，使用在 **第一部分** 中找到的执行器运行：

* **使用 Isaac Lab 启动**：
  ```powershell
  # 如果 isaaclab.bat 在环境变量中：
  isaaclab -p play_isaac.py
  # 或者输入完整路径：
  C:\path\to\isaaclab.bat -p play_isaac.py
  ```
* **使用 Isaac Sim 启动**：
  ```powershell
  C:\Users\<您的Windows用户名>\AppData\Local\ov\pkg\isaac-sim-XXXX\python.bat play_isaac.py
  ```
这将启动带有机械手和红色方块的仿真渲染窗口，您的策略将自动控制手指协调旋转方块。

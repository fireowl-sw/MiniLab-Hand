# Windows Isaac Sim 部署与演示说明

本文件夹包含专为 Windows 端的 **Isaac Sim** / **Isaac Lab** 编写的独立演示脚本和配套资产。

您可以直接在 Windows 电脑上使用 `git pull` 拉取此文件夹。

## 📂 文件结构

* `play_isaac.py`：Standalone Python 控制和推理脚本，负责驱动机械手。
* `deploy_config.yaml`：保存了 22 维关节的名称映射、默认关节角度等关键超参数。
* `assets/robots/sharpa_wave/`：灵巧手的 MuJoCo XML 模型和三维 STL 网格文件。

---

## 🚀 Windows 运行步骤

### 第一步：拷贝 ONNX 策略模型
在 macOS 端完成训练并导出 ONNX 策略后，将 `policy.onnx` 文件复制并放置在本文件夹根目录下：
```
deploy_windows/
├── assets/
├── deploy_config.yaml
├── play_isaac.py
└── policy.onnx  <-- 放置在此处
```

### 第二步：在 Isaac Sim 中生成机械手 USD
1. 在 Windows 上打开 **Isaac Sim**。
2. 菜单栏点击：`Isaac Utils` -> `MJCF Importer`。
3. 导入本文件夹中的 `assets/robots/sharpa_wave/right_sharpa_wave.xml`（确保勾选 `Create Articulation`）。
4. 在 Stage 树状图中检查各关节，确保其控制模式为 **Position**（位置驱动）。
5. 点击 `File` -> `Save As`，将导入的模型保存为：
   `assets/robots/sharpa_wave/right_sharpa_wave.usd`

### 第三步：启动演示
打开 Windows 的命令行终端（CMD 或 PowerShell），进入本 `deploy_windows` 目录，执行：

```bash
# 使用 Isaac Lab 环境运行
isaaclab.bat -p play_isaac.py
```

或使用 Isaac Sim 自带的 Python 解释器直接运行：
```bash
C:\Users\<您的用户名>\AppData\Local\ov\pkg\isaac-sim-4.X.X\python.bat play_isaac.py
```
这会开启一个独立的渲染窗口，加载您训练好的策略来操控灵巧手旋转红色方块。

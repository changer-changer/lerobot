# Tac3D SDK 使用指南

本指南旨在向您介绍如何使用 Tac3D SDK (Python) 获取触觉点云信息。

## 1. 系统架构

Tac3D 系统由两部分组成：

1.  **Tac3D Core (后端)**：一个 C++ 可执行程序 (`Tac3D`)，负责连接硬件传感器、处理原始图像并计算 3D 变形。它作为 UDP 服务端运行，广播处理后的数据帧。
2.  **PyTac3D (前端 SDK)**：一个 Python 模块 (`PyTac3D.py`)，作为 UDP 客户端运行，接收并解码来自 Core 的 BSON 格式数据。

## 2. 环境准备

### 硬件连接
确保 Tac3D 硬件已连接并被系统识别。

### 软件依赖
- **Python 3.10+** (推荐)
- **PyMongo** (提供 `bson` 模块)
- **NumPy**
- **ruamel.yaml**
- **OpenCV** (用于图像解码)

## 3. 运行 Tac3D Core

在运行任何 Python 程序之前，必须先启动 Tac3D Core：
```bash
cd src/tac3d_ros2/Tac3D-SDK-v3.3.0/Tac3D-Core/linux-x86_64
./Tac3D -c config/DL1-GWM0053 -i 127.0.0.1 -p 9988
```
- `-c`: 传感器配置文件路径。
- `-i`: 绑定的 IP 地址（本地运行使用 127.0.0.1）。
- `-p`: SDK 连接的端口（默认 9988）。

## 4. 编写你的 Python 程序

### 最小示例代码
```python
import time
from PyTac3D import Sensor
import numpy as np

def main():
    # 1. 初始化传感器。端口需与 Tac3D Core 设置的端口一致。
    sensor = Sensor(port=9988, maxQSize=10)
    
    # 2. 阻塞等待，直到传感器连接并开始传输数据。
    sensor.waitForFrame()
    print("成功连接到 Tac3D!")

    try:
        while True:
            # 3. 获取最新的触觉数据帧。
            frame = sensor.getFrame()
            
            if frame is not None:
                # 4. 访问数据。
                # SN: 传感器序列号
                sn = frame['SN']
                # 3D_Positions: (400, 3) 原始点云位置
                pos = frame['3D_Positions']
                # 3D_Displacements: (400, 3) 每个点的 XYZ 变形量
                displacement = frame['3D_Displacements']
                # 3D_Forces: (400, 3) 每个点受到的 XYZ 方向力
                forces = frame['3D_Forces']
                
                # 将原始位置与变形量相加，即可得到实时的 3D 形状。
                current_shape = pos + displacement
                
                # 打印一些统计数据。
                max_force = np.linalg.norm(forces, axis=1).max()
                print(f"帧序号 {frame['index']} | 最大受力: {max_force:.2f}N")
            
            # 控制读取频率（例如 60Hz）。
            time.sleep(1/60.0)
            
    except KeyboardInterrupt:
        print("停止运行...")

if __name__ == "__main__":
    main()
```

### 核心数据结构：`frame` 字典说明
| 键名 | 类型 | 描述 |
| :--- | :--- | :--- |
| `SN` | `str` | 物理传感器的序列号。 |
| `index` | `int` | 递增的帧序号。 |
| `sendTimestamp` | `float` | 后端 Core 发送数据的时间戳（秒）。 |
| `recvTimestamp` | `float` | 前端 SDK 接收数据的时间戳（秒）。 |
| `3D_Positions` | `numpy.ndarray` | (400, 3) 原始网格坐标。 |
| `3D_Displacements`| `numpy.ndarray` | (400, 3) 相对于基准位置的 XYZ 位移。 |
| `3D_Forces` | `numpy.ndarray` | (400, 3) 每个点受到的 XYZ 矢量力。 |

## 5. 进阶功能

### 在线校准
为了消除由于温度或外部光线变化产生的“零位漂移”，可以调用校准函数：
```python
sensor.calibrate(sn)
```

### 回调函数模式
除了使用 `getFrame()` 主动轮询，还可以在初始化时设置回调函数：

```python
def my_callback(frame, param):
    print(f"收到数据帧 {frame['index']}")

sensor = Sensor(recvCallback=my_callback)
```

## 6. Tac3D Core 二进制文件管理建议

Tac3D Core 文件夹较大（包含模型和库文件），且多个项目通常只需要共用一个 Core。以下是推荐的高效管理方案：

### 方案 A：集中管理 + 环境变量 (推荐)

将 Core 存放在一个固定的位置（例如 `~/software/tac3d`），避免在每个项目中重复拷贝。

1. **统一存放**：

   ```bash
   mkdir -p ~/software
   mv src/tac3d_ros2/Tac3D-SDK-v3.3.0/Tac3D-Core ~/software/tac3d
   ```

2. **设置环境变量**：

   在 `~/.bashrc` 中添加：

   ```bash
   export TAC3D_HOME=~/software/tac3d
   ```

3. **创建快捷指令 (Alias)**：

   在 `~/.bashrc` 中继续添加：

   ```bash
   alias run-tac3d='cd $TAC3D_HOME/linux-x86_64 && ./Tac3D -c config/DL1-GWM0053 -i 127.0.0.1 -p 9988'
   ```

   之后在任何目录下只需输入 `run-tac3d` 即可启动，程序无需随着项目移动。

### 方案 B：软链接 (Symbolic Link)

如果您的 Python 项目必须引用特定的 SDK 文件或配置文件，可以使用软链接：

```bash
ln -s ~/software/tac3d/PyTac3D.py ./PyTac3D.py
```

### 方案 C：Systemd 后台服务化

对于长期运行的机器人工作站，可以将 Core 注册为系统服务，开机自启且在后台运行。

您可以创建一个 `tac3d.service` 文件：

```ini
[Unit]
Description=Tac3D Core Background Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/cuizhixing/software/tac3d/linux-x86_64
ExecStart=/home/cuizhixing/software/tac3d/linux-x86_64/Tac3D -c config/DL1-GWM0053 -i 127.0.0.1 -p 9988
Restart=always

[Install]
WantedBy=multi-user.target
```

这样您只需要关注 Python 侧的开发，后端会自动常驻后台。

## 7. 常见疑问解答 (FAQ)

### Q: 我发现可以直接 `pip install pytac3d`，这样可以吗？

**A**: **可以，而且这是最推荐的“现代化”安装方式。**

如果您执行 `pip install pytac3d`，系统会自动安装 `PyTac3D` 库及其所有依赖项（如 `numpy`, `ruamel.yaml` 等）。

- **优点**：无需手动拷贝 `PyTac3D.py` 文件，您可以像使用普通 Python 库一样在任何地方 `import PyTac3D`。
- **注意事项**：
  - **版本匹配**：请确保 `pip` 安装的版本与您正在使用的 `Tac3D Core` 后端版本兼容。如果 Core 版本比较旧，可能需要使用 SDK 文件夹里自带的驱动文件以保证协议一致。
  - **安装包全名**：在 `pip` 上该包的全名通常是 `pytac3d`。

**总结更新**：
1. **懒人模式**：`pip install pytac3d`，然后直接写代码。
2. **保守模式**：如果遇到连接问题，将 SDK 里的 `PyTac3D.py` 考入项目，这能保证驱动与 Core 绝对匹配。

### Q: SDK 里的 `Tac3D-API` 文件夹是干什么的？

**A**: `Tac3D-API` 是驱动程序的**原始代码库**。
- 它包含了 Python 和 C++ 两个版本的 API 源码。
- 当您要开发新项目时，应该从 `Tac3D-API/python/PyTac3D/` 中把 `PyTac3D.py` 拷贝到您的项目目录下。
- 其中 `Tac3DClient` 文件夹下还有更底层的客户端实现和示例代码。

**总结**：`pip` 装的是“工具箱”，`Tac3D-API` 提供的是“钥匙”。你需要把“钥匙”拿出来放到你的项目里，才能配合“工具箱”打开传感器的数据大门。

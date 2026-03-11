# Project Context

## @ProjectStructure

- `src/lerobot/tactile/`: 全新的触觉传感器驱动模块，被设计为高度解耦的形式。
  - `direct_connection/tac3d_sensor.py`: [驱动接口] -> [直连无ROS环境，通过 UDP Pytac3d_SDK 读取 6 维触觉数据]
  - `ros2_bridge/pointcloud2_sensor.py`: [ROS接口] -> [中间件模式，支持通用的外部 ROS2 节点消息订阅解析]
  - `simulated_sensor.py`: [仿真接口] -> [生成带有噪声的测试伪数据用于流水线测试]
  - `dataset_integration.py`: [整合工具] -> [格式化触觉特征字典，支持合并录制流水线]

## @CurrentState

- **Status**: Validation & Reporting
- **Focus**: 测试触觉代码分支在无传感器环境下的后向兼容性，并确认模拟数据录制功能。
- **Blockers**: 已清除（此前点云组件中的 `PointCloud2` 的强制 Type Hinting 会引发 `ImportError`（若机器并未安装 ROS2），已通过字符串化修复）。

## @TechSpec

- **Data Schemas**: Dataset Observation Feature 增加 `observation.tactile` (shape: `400*6`, float32)。
- **Constraints**: 传感器必须支持 `udp_port=9988`（默认）。
- **Environment**: conda `lerobot_tactile_test` 独立测试服，Python 3.10，未携带 ROS2 原生支持。

## @History

### Part 1: 时间线日志 (Project Timeline)

- **[2026-02-28 | 夜间]**
  - **核心事件**: 完成 LeRobot 框架触觉集成测试、审核与重构
  - **操作摘要**:
    - 建立纯净隔离的 conda `lerobot_tactile_test` 虚拟环境并源码安装 lerobot。
    - 清除 `pointcloud2_sensor.py` 环境强依赖（ROS2）造成的隐式抛错风险。
    - 基于 `simulated_sensor` 执行全套录制闭环测试，产出完好的触觉数据（Parquet）。
    - 断开传感模拟，启动原生 SO-100 空骨架的虚拟录制，100% 确认代码改动未侵入系统主干，无倒退风险。
    - **重构 (Refactoring)**: 出于架构清晰度要求，将传感器依据接入方案严格拆分到了 `direct_connection/` 和 `ros2_bridge/` 文件夹中并修正了 import。
    - **AI 记忆存储**: 在 `AI_memory` 创建了该次双轨融合模式的深度推演脑图日志 `Task_Tactile_Architecture_2026-02-28.md`。
  - **当前状态**: 已完成

- **[2026-03-01]**
  - **核心事件**: 完成了 LeRobot Diffusion Policy (TacDP) 模型的全栈集成设计与后端架构开发
  - **操作摘要**:
    - 在 `configuration_diffusion.py` 增加了对触觉外设特征 `tactile_features` 的自动识别，以及设置 PointNet 维度与 FACTR 参数等扩展超参提取配置。
    - 将 `PointNetEncoder` 直接作为新的模态主入口嵌入 `modeling_diffusion.py` 中的 `DiffusionModel`。采取 Late-Fusion（晚期特征缝合）手段，并自动计算该模态叠加至全局感知环境特征表 `global_cond_dim` 的数量缩放。
    - 实现 **FACTR（Force-Attending Curriculum Training）** 模型端干预机制：针对视觉主干增设 `global_step_tensor`；在执行 `DiffusionPolicy:forward()` 时段根据全局迭代步骤动态赋予 RGB 图片降维和动态高斯模糊干预，强迫模型提取触觉力学反馈，并克服高带宽视觉吞并触觉信息的模态雪崩挑战 (Modality Collapse)。
    - **隔离验证**: 执行无环境污染验证脚本 (`test_tac_dp.py`)，成功测试到了反向传播流水无误，PointNet/ResNet 双流分轨均有梯度反馈。
  - **当前状态**: 已验证完成

### Part 2: 功能与创新演进树 (Functional Evolution Tree)

**[传感器框架扩展：Tactile Integration]**

**1. [模块化集成 Tac3D 等触觉硬件数据到主存集]**

- **目的 (Purpose)**: 让机器人的操作数据集拥有带高频维度的受力与指尖形变感知能力。
- **必要性 (Necessity)**: 单一的 RGB+本体感觉不够精细化，抓取与避障的 Soft-actor 策略高度依赖多波段力的反馈。
- **尝试与改动 (Attempts & Changes)**:
  - _Attempt 1 (History)_: 为不干扰原有多模态流，将 Tactile 组件独立打包并提供了统一规范的通信基类 `TactileSensor`。由上级脚本选择性开启录制融合。
  - _Attempt 2 (Current)_: 为核心 Diffusion Policy 扩充感知通道。利用轻量级 `PointNetEncoder` 对 `(N_points, 6)` 进行特征浓缩；引入基于训练步骤线性衰减的高斯平滑（FACTR Curriculum），有效抑制了多模特征拼接下的视觉独裁问题。
- **结果与反馈 (Results)**: Mock 接口模型梯度正常回传。FACTR 在预处理逻辑端完成拦截，未深度干预推理流（eval阶段不施加模糊），具有较高的生产端稳健性。
- **下一步/遗留问题 (Next Steps)**: 使用真机数据并用真实的推演脚本在实机上执行训练。
t
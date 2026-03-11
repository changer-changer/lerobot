# Task_PointCloud_DP_20260301

## A. 认知锚点

- 状态: Execution Phase (正在实现 TacDP)
- 上下文: 用户已审核通过 `implementation_plan.md`，目前我们准备对 `configuration_diffusion.py` 和 `modeling_diffusion.py` 进行修改，以接入在 `tactile_sensor.py` 中被定义好的触觉数据（尺寸为 `(num_points, 6)`）。为了避免模态坍塌（Modality Collapse），将引入 FACTR Curriculum (对早期训练时的视觉特征施加高斯模糊，迫使模型利用触觉特征)。

## B. 深度认识

- **Data Shapes**: 触觉输入的维度是 point clouds `(B, T, num_points, 6)`。这不同于普通的 1D state 或者 2D images。我们需要实现一个 1D CNN + Pooling 形式的 `PointNetEncoder`。
- **FACTR Mechanism**: FACTR（Force-Attending Curriculum Training）要求在视觉流（Images）进入 CNN/ResNet 之前，施加一个随 step 下降的 Gaussian Blur。

## C. 动态计划

- [x] 读取 `PROJECT_CONTEXT.md` 熟悉开发环境与准则
- [ ] 修改 `configuration_diffusion.py` 加入触觉参数和 FACTR 参数
- [ ] 读取 `modeling_diffusion.py` 寻找插入 PointNetEncoder 与 FACTR augment 的合适位置
- [ ] 实现代码并创建 mock 脚本 `test_tac_dp.py` 验证 pipeline

## D. 学习与发现

（待更新）

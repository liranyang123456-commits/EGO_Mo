# 投稿前实验证据补齐流程

研究协议 ID：`85393aba60ecdc40`

## 1. 四段前瞻性测试

模型已于 2026-09-26 冻结（`datasets/final_model_manifest.json`，清单哈希
`e50df3c9…`）：16 个 `physnet_cvg_*` 门控 PhysNet、16 个 `imunet_cvx_*`
IMUNet，以及生成预测、标签和评测的 12 个代码/配置文件。此后不得修改这些文件，
否则 `run` 会拒绝执行。若在看到任何 T 指标之前还想加 K–O 训练数据重训，只能删除清单后重新冻结。

### 1a. 采集排期

| 段 | 日期 | 操作者/握法 |
|---|---|---|
| T1 | 第 1 天 | 操作者 A，惯用握法 |
| T2 | 第 1 天 | 操作者 A 换握法，或操作者 B |
| T3 | 第 2 天（重新开机、重新装夹后） | 操作者 A |
| T4 | 第 2 天 | 另一操作者或另一握法 |

每段开录前后在 `datasets/prospective_test_checklist.json` 对应槽位填写 `session`、`day`、`operator`、`grip`。

### 1b. 每天开工（桌面快捷方式「EGO_Mo 采集」）

1. 双击「EGO_Mo 采集」。左侧点 **0 扫描设备**，在缩略图里点选左目；确认 USB IMU（COM3）和蓝牙 IMU 都已连上。
2. 点 **1 帧率门禁** →「开始」，跑 8 秒。两路 IMU 都应接近 200 Hz，否则重插 USB、重连蓝牙后再试。
3. 点 **9 IMU 刚性检查** →「开始」，按提示录 60–90 s：先静止 5 s，点头 ±20° 三次、摇头 ±20° 三次、绕光轴转 ±25° 三次，每次之间停 2 s，最后静止 5 s。停止后界面必须显示「合格」。不合格说明 IMU 松动，先固定好再重录，**不合格当天不能采 T**。

### 1c. 录一段 T（每段约 4 分钟）

1. 棋盘平放固定在桌上不动，照明稳定，蓝牙 IMU 粘在棋盘上不要碰。
2. 左侧点 **10 轨迹**。在上方「主题」下拉框**手动选 `T 独立测试`**（程序只在训练主题录满后才会自动推荐 T），右侧信息应显示分组 `test`。**选好后不要再改。**
3. 点「开始」后照下面的节奏做，全程保持棋盘完整在左目画面里，距离 15–40 cm：
   - 0–8 s：完全静止；
   - 之后自由混合运动约 160 s：左右、上下、前后平移（行程尽量做到 10–20 cm），夹杂点头、摇头、绕光轴转；快慢交替，但画面里的棋盘不能糊；
   - 每 10–15 s 明显停住 1–2 s（心里默数「一、二」）；
   - 最后 10 s：完全静止。
4. 点「停止」。程序自动检查并写入 `datasets/collection_log.csv`，看最后一行：
   - 合格条件：棋盘可用率 ≥70%（此前 `traj_20260924_144019` 就是 69% 不合格）、重投影中位数 ≤0.4 px、无丢帧；
   - 不合格：这一段作废，不能挑选重录去替换已合格的段；在清单 `notes` 注明原因后再录一段新的。
5. **不要打开、不要计算任何 T 段的模型误差。** 只看采集程序自带的质量检查。

### 1d. 四段都合格后

```powershell
python tools/build_pose_gt.py
python tools/export_gt_raw.py
python tools/prepare_trajectory_data.py
```

1. `prepare_trajectory_data.py` 的输出必须显示 `prospective_test_status.complete=true`、`accepted=4`。
2. 生成一次预测（输出文件名不得覆盖已冻结的 `session_cv_gate_benchmark.json`）：

   ```powershell
   python tools/eval_session_cv.py --seeds 0 1 --cv-tag cvg --baseline-tag cvx --output datasets/final_predictions.json
   ```

3. 仅运行一次：

   ```powershell
   python tools/final_prospective_evaluation.py run --predictions datasets/final_predictions.npz
   ```

4. `FINAL_EVALUATION_COMPLETE.lock` 出现后禁止继续调参；若发现代码错误，签署修订说明并采集新的测试段。

## 2. 精密位移台/千分尺验证

器材：

- 三轴精密滑台或可溯源千分尺，分辨率 ≤0.01 mm；
- 校准证书或量块核查记录；
- 刚性相机夹具；
- 固定 GP050 棋盘和稳定照明。

步骤：

```powershell
python tools/reference_stage_validation.py init
```

按 CSV 顺序完成 57 个 8–12 s 静止录制。第二遍按反方向移动，用于暴露回差。真实 X/Y/Z 读数必须来自滑台，不得从相机结果反推。填入会话名后：

```powershell
python tools/build_pose_gt.py
python tools/export_gt_raw.py
python tools/reference_stage_validation.py analyze
```

门限：留出 RMSE ≤1.0 mm、最大误差 ≤2.0 mm、各轴尺度误差 ≤1%、串轴 RMSE ≤0.5 mm、重复性 p95 ≤0.5 mm。分析只拟合刚体旋转和平移，不拟合尺度。

### 2b. 无位移台替代：数显游标卡尺一维验证

器材：0–150 mm 数显游标卡尺（分辨率 0.01 mm，附出厂合格证或用量块核查一次），双面胶/小夹子，相机固定支架。

1. 相机刚性固定在桌面上不动；把 GP050 棋盘贴在卡尺**活动量爪**上，卡尺主尺用重物或胶带固定。
2. 生成计划：`python tools/reference_stage_validation.py init --mode caliper`
   （`datasets/reference_caliper_plan.csv`：X/Y/Z 三个方向 × 3 次往返，每次在 0、5、10、15、20、30、40 mm 停顿）。
3. 每个方向装夹一次：X = 卡尺平行图像水平方向，Y = 平行图像竖直方向，Z = 沿光轴（棋盘朝镜头移动）。**同一方向的 3 次往返之间不得重新装夹**。
4. 每次往返录一段连续录像：在每个读数处停 ≥5 s，读数以卡尺显示为准；两个读数之间的移动用 1.5–3 s 匀速完成，中途不要停。
5. 把会话名、卡尺编号、分辨率填入 CSV，然后：

   ```powershell
   python tools/build_pose_gt.py
   python tools/export_gt_raw.py
   python tools/reference_stage_validation.py analyze --mode caliper
   ```

分析先对位置做 ±0.7 s 滑动中位数滤波（去掉 PnP 在停顿中约 1 mm 的离散跳变），再按滤波后逐帧步长 <0.2 mm 切出停顿段，只保留计划中数量的最长段；每个方向只拟合位移方向和原点，不拟合尺度；方向由其余往返估计，留出往返用于评分。门限与位移台相同。合成数据自检（24 组，含 0–20% 跳变帧）：真实尺度全部通过（留出 RMSE ≤0.02 mm），注入 2% 尺度误差全部检出（2.0%）并判为不通过。局限：各方向分别装夹，不检验轴间正交性。

### 2c. 仿真已知位移检验（已完成，替代实物位移台）

```powershell
python tools/sim_reference_validation.py --mc 50
```

用数字孪生按 2b 的卡尺协议渲染棋盘（标定畸变、传感器噪声、光照闪烁），走与真实参考**同一套**角点检测和 PnP（`build_pose_gt.py`），再用 2b 的分析评分。结果写在 `datasets/sim_reference_validation.json`，并已写入论文第 V 节：

| 工作距离 | 条件 | 留出 RMSE | 最大轴尺度误差 | 最大串轴误差 |
|---|---|---|---|---|
| 250 mm | 内参精确 | 0.73 mm | 2.1% | 0.88 mm |
| 350 mm | 内参精确 | 1.20 mm | 2.5% | 1.86 mm |
| 250 / 350 mm | 内参按标定标准差抽样 50 次 | 中位 1.0 / 1.4 mm | — | — |

结论：3 mm 节距小棋盘的 PnP 参考存在约 2–2.5% 的系统性尺度误差（重复性仅约 0.01 mm），达不到 1% 目标；折算到测试集平均约 50 mm 的 3 s 位移约 1.2 mm，只有模型误差的约 3%，不影响方法排序，但不能声称亚毫米级参考精度。150 mm 距离下 Y 方向 40 mm 行程超出视场，未计入；实物检验应在 250–350 mm 做。

仿真中还发现：棋盘移动时 PnP 位置会出现约 1 mm 的跳变。后续检验表明它与 `cornerSubPix` 的终止精度无关（收紧到 0.001 px、100 次迭代后结果不变），也与 PnP 求解器无关（IPPE、LM 精修结果相同）；主要来源是渲染边缘过于锐利（10–90% 边缘宽度 0.8 px，真实图像约 3.9 px）。加入与真实边缘宽度匹配的光学模糊（σ≈1.5 px）后，150 mm 处跳变基本消失；250–350 mm 处小棋盘的 PnP 仍会偶发不稳定。因此仿真结果对光学模糊建模很敏感，只能给出参考误差的范围。

局限：这是流程级检验，不是 SI 可溯源检验；棋盘平面度、卷帘快门、真实离焦和真实内参误差都未包含。就绪检查中该项因此为警告而非通过；做完 2b 的实物卡尺检验可升级为通过。

## 3. 双目外参

根因已查明：右目 `cam1/times.txt` 比右目图像多约 400 行，行号并不对应图像编号；旧工具按时间戳配对，导致左右图像错配（拟合 2.99 px）。左右图像按相同编号成对保存（两路棋盘中心轨迹相关系数 0.85/0.97，残余异步约 0.52 帧）。`calibrate_stereo_extrinsic.py --quasi-static` 已改为按编号配对、只取两路角点都近乎静止的帧、并用初值 + 联合重投影优化替代发散的 `cv2.stereoCalibrate`：基线 46.3 mm、滚转 10.2°、拟合 0.58 px，但留出旋转 1.47°、平移 4.8 mm，仍未通过门限。右目不参与参考位姿和 IMU 结果，该项已降为警告，不再阻塞投稿。若要通过，使用采集程序第 12 步重新录制：

- 每个棋盘姿态静止 2 s；
- 远/中/近、图像四角、俯仰/偏航/横滚；
- 至少 25 个静止姿态。

```powershell
python tools/calibrate_stereo_extrinsic.py --quasi-static --motion-px 1 --session datasets/<stereo_ext_session>
```

只有 `passed=true` 才能替换名义外参。必须同时满足拟合 RMS ≤0.6 px、外极线 p95 ≤1.0 px、留出旋转 RMSE ≤0.5°、留出平移 RMSE ≤2 mm。

## 4. 相机—IMU 杆臂

```powershell
python tools/register_lever_arm_measurement.py init
```

测量“USB IMU 敏感中心 → 左目光心”向量，表达在左相机坐标系（+x 向右、+y 向下、+z 向前）。至少 5 次拆装/重复，记录工具编号、分辨率和镜头表面到光心的图纸修正。完成后：

```powershell
python tools/register_lever_arm_measurement.py analyze
```

机械实测值用于最终配置；网络学习值只作独立一致性比较，禁止反向校准机械测量。

注意：`coordinate_frames.json` 中现存的 `T_C0_Iusb` 平移 (5.5, −1.5, 168.3) mm 不是实测值，与网络学到的约 38 mm 杆臂矛盾，不得引用。

### 4a. 机械实测（主值，约 15 分钟，钢直尺或卡尺均可）

1. 把整机放在桌上，镜头朝前（朝向平时拍摄的棋盘方向）。站在相机**后方**顺着视线看：
   向右 = +x，向下 = +y，向前（朝被拍物）= +z。
2. 定两个点：
   - **相机点**：左目镜头前端面的圆心（镜头直径 5.5 mm，取圆心）；
   - **IMU 点**：USB IMU 模块内芯片的位置。没有数据手册时取模块外壳的几何中心（长宽高各取一半），在 `notes` 写「外壳中心」。
3. 符号规则：向量 = 相机点 − IMU 点。
   - 相机在 IMU **右边** → x 为正，左边为负；
   - 相机在 IMU **下方** → y 为正，上方为负；
   - 相机在 IMU **前方**（更靠近被拍物）→ z 为正，后方为负。
4. 分别沿 x、y、z 三个方向量两点间的距离（尺子与该方向平行，读数估读到 0.5 mm），按上面的规则加符号，填入 `datasets/lever_arm_measurements.csv` 的 `imu_to_camera_x/y/z_mm`。
5. `optical_center_correction_mm`：光心在镜头前端面之后多少毫米。有镜头资料就填资料值，程序会自动从 z 中减去；没有就填 0，并在 `notes` 写「光心按前端面，±2 mm」。
6. `instrument_id` 填尺子型号，`resolution_mm` 填 0.5（钢直尺）或 0.02（卡尺）。
7. 每次把尺子拿开重新放，共量 5 次。然后：

```powershell
python tools/register_lever_arm_measurement.py analyze
```

参照：网络学到的值是 (−22.6, −5.2, −29.7) mm，意思是相机在 IMU 左侧约 23 mm、上方约 5 mm、后方约 30 mm。按实物如实测量，这个值只作事后对照，不得用于修正实测。

### 4b. 数据估计（独立交叉验证，可选）

`tools/estimate_lever_arm.py` 用刚体运动学（比力 = 相机加速度 − 重力 + 杆臂引起的切向/向心加速度）在真实录像上直接解杆臂，不经过网络。仿真验证（注入已知杆臂）：

| 录制方式 | PnP 噪声 | 各轴 RMSE (mm) |
|---|---|---|
| 现有 rigid 段（±20° 慢转） | 1 mm / 0.3° | 4.9 / 28.6 / 6.7 |
| ±30° 中速（峰值约 95°/s） | 1 mm / 0.3° | 3.7 / 5.9 / 8.9 |
| ±20° 慢转 | 0.3 mm / 0.2° | 2.8 / 3.7 / 1.7 |

现有两段 rigid 录像的激励不够，95% 置信区间宽达 ±50 mm 以上，**不能用**。如需数据估计，用采集程序第 9 步专门录一段「杆臂激励」：棋盘距镜头 15–20 cm（越近 PnP 越准），点头、摇头、绕光轴转各 ±30°，每次约 1 s 往返、连续做 10 次，峰值不超过约 100°/s（画面不糊），轴与轴之间停 3 s，共 90 s；然后：

```powershell
python tools/build_pose_gt.py
python tools/export_gt_raw.py
python tools/estimate_lever_arm.py real --sessions <新录的 rigid 会话名>
```

预期精度 5–9 mm，只用作 4a 的交叉验证，不替代实测。

## 5. 相对 6-DoF 评测

```powershell
python tools/eval_pure_imu_pose.py --predictions <prediction_bundle.npz>
```

该评测用第一帧参考位姿固定坐标规约，后续姿态来自陀螺积分、平移来自 IMU 网络；不会再使用图像。它是相对 6-DoF 轨迹，不是绝对姿态或参考真值。当前历史测试结果：姿态 RMSE 4.51°、3-s 旋转 RPE 3.48°、最佳融合平移 ATE 80.59 mm。

## 6. 公开发布

```powershell
python tools/build_release_bundle.py
```

上传 `release/EGO_Mo_v1.0.0.zip` 到公开代码仓库并连接 Zenodo。原始录像应作为独立数据集发布，保留 `DATA_MANIFEST.json` 的会话结构和 SHA-256。获得 DOI 后更新 `CITATION.cff`、Cover Letter 和论文数据可用性声明。

## 7. 投稿

逐项完成 `paper/SUBMISSION_CHECKLIST_TIM.md`。当前稿 14 页，按 2026 TIM 规则预计 6 页超页：非 IMS 会员 US$1,590，IMS 会员 US$1,320，均未含税；最终金额以录用后排版页数为准。

## 8. v3 方法迭代记录（模型冻结之后；只用验证集与已有折模型，不读封存数据）

```powershell
python tools/v3_experiments.py --configs base bias_local beta5 nll_main nogate_beta5 nogate --seeds 0 1
python tools/v3_scale_sweep.py --configs base nogate nll_main
python tools/v3_mixture_eval.py
python tools/v3_uncertainty.py
```

| 改动 | 验证误差 (mm) | 回归斜率 x/y/z | 验证 ATE (mm) | 结论 |
|---|---|---|---|---|
| 冻结配置重训（门控） | 29.49 | 0.48/0.56/0.23 | 126.4 | 基线 |
| 停顿插值陀螺零偏 | 29.84 | 0.46/0.51/0.24 | 146.4 | 变差；多数录像停顿不足 |
| 近似均方误差（β=5 cm） | 30.41 | 0.42/0.49/0.27 | 114.6 | 收缩不变 |
| NLL 为主损失 | 29.92 | 0.43/0.47/0.29 | 108.7 | 收缩不变 |
| 去门控 | 30.95 | 0.43/0.49/0.19 | 94.4 | 轨迹最好 |
| 事后放大 α=1.5（门控） | 31.06 | — | 152.2 | 误差与轨迹都变差 |
| 门控+非门控混合（验证） | 29.59 | — | 103.5 | 折中 |
| 门控+非门控混合（8 折 CV / 历史测试） | 24.63 / 43.89 | — | 测试 97.95 | CV 小幅改进，测试不胜 IMUNet |

不确定度：单一 κ、按 σ 分层、分裂保形、按幅值缩放四种校准都修不好测试集的覆盖不足（保形在跨验证录像上命中名义值 68.7%/95.3%，测试集 1σ 仅 49.9%），需要更多样的校准数据。

结论：收缩是信息不足下条件均值的表现，换损失或事后放大都无法消除；下一步只能从采集端补充信息（高停顿率、30 fps 参考、K–O 数据）。

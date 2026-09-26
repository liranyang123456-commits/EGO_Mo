# 投稿前实验证据补齐流程

研究协议 ID：`85393aba60ecdc40`

## 1. 四段前瞻性测试

模型已于 2026-09-26 冻结（`datasets/final_model_manifest.json`，清单哈希
`e50df3c9…`）：16 个 `physnet_cvg_*` 门控 PhysNet、16 个 `imunet_cvx_*`
IMUNet，以及生成预测、标签和评测的 12 个代码/配置文件。此后不得修改这些文件，
否则 `run` 会拒绝执行。若在看到任何 T 指标之前还想加 K–O 训练数据重训，只能删除清单后重新冻结。

1. 在至少两天、两个操作者或握法下采集 T1–T4（每段约 180 s，每 10–15 s 停 1–2 s，
   行程 10–20 cm，工作距离 15–40 cm）。采集程序自动写入协议 ID。
2. `prepare_trajectory_data.py` 必须显示 `prospective_test_status.complete=true`。
3. 生成一次预测（输出文件名不得覆盖已冻结的 `session_cv_gate_benchmark.json`）：

   ```powershell
   python tools/eval_session_cv.py --seeds 0 1 --cv-tag cvg --baseline-tag cvx --output datasets/final_predictions.json
   ```

4. 仅运行一次：

   ```powershell
   python tools/final_prospective_evaluation.py run --predictions datasets/final_predictions.npz
   ```

5. `FINAL_EVALUATION_COMPLETE.lock` 出现后禁止继续调参；若发现代码错误，签署修订说明并采集新的测试段。

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
4. 每次往返录一段连续录像：在每个读数处停 ≥3 s，读数以卡尺显示为准，移动过程中不要停顿。
5. 把会话名、卡尺编号、分辨率填入 CSV，然后：

   ```powershell
   python tools/build_pose_gt.py
   python tools/export_gt_raw.py
   python tools/reference_stage_validation.py analyze --mode caliper
   ```

分析自动切分静止段（1 s 窗口内偏离中位数 <0.3 mm），每个方向只拟合位移方向和原点，不拟合尺度；方向由其余往返估计，留出往返用于评分。门限与位移台相同。合成数据自检：真实尺度时留出 RMSE 0.019 mm 通过；注入 2% 尺度误差时三轴均检出 2.0% 并判为不通过。局限：各方向分别装夹，不检验轴间正交性。

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

实测方法（用第 2b 节同一把卡尺，约 15 分钟）：从 USB IMU 模块中心量到左目镜头前端面中心，按左相机坐标系（+x 右、+y 下、+z 前）记录带符号的三个分量（IMU→相机方向，与 CSV 定义一致）；IMU 敏感中心按模块数据手册的芯片位置修正（无手册时取外壳几何中心，并在 `notes` 注明）；光心在镜头前端面之后，按镜头图纸或厂家标称填 `optical_center_correction_mm`（无资料时填 0 并注明，不确定度按 ±2 mm 记）。重复 5 次，每次重新放置卡尺。

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

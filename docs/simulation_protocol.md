# EGO_Mo 双目相机 + IMU 数字孪生数据集

## 1. 原则

棋盘 GP050 的尺寸已知（12×9 格、3 mm 方格），因此仿真真值使用精确的程序化几何，不从照片重建尺寸。照片和真实录像只用于拟合：

- 相机内参、畸变、图像噪声、运动模糊和光照范围；
- USB IMU 的采样率、量化、白噪声、静态偏置和随机游走；
- IMU 到相机的刚性旋转；
- 真实运动的速度、加速度和转角范围。

`datasets/sim_assets/gp050_board/gp050_board.obj` 是可导入 Blender/Unreal 的毫米尺度棋盘模型，坐标原点与真实 PnP 一致：第一个内角点。

## 2. 当前实现

双击桌面 **EGO_Mo 仿真**，或运行：

```powershell
E:\EGO_Mo\run_simulator.bat
```

界面提供：

- 左右目实时渲染；
- 三维相机轨迹和棋盘；
- 200 Hz 加速度、陀螺曲线；
- lateral / depth / circle / mixed / random spline 基础运动；
- slow_far（长距离慢速）、aggressive_6dof（剧烈六自由度）、
  pure_rotation、translation_xyz、pitch_sweep、stop_go、spiral；
- 平移、深度、转角、速度、光照、噪声、杆臂参数；
- 固定棋盘或运动棋盘（后者只作为 OOD/负样本）；
- 时间滑块和播放；
- 一键导出兼容数据集。
- “真实 / 网络轨迹”页：真实轨迹、捷联、Ridge、zero-shot、真实训练、微调和最终混合轨迹；
- “方法比较”页：分别查看真实验证、真实测试、仿真测试和仿真 OOD；
- “数据集与物理检查”页：数据段数、split、噪声来源和审计结果；
- 真实分布、高速、深度推进、棋盘运动 OOD、静态标定五种参数预设。

界面支持两套导出后端：

- OpenCV CPU：快速预览和大批量基础数据；
- Blender Eevee：真实三维材质、阴影和光照，支持 GPU 渲染。

分辨率支持 640×480、1280×720、1920×1080 和 2560×1440（2K/QHD），默认 720p。

## 3. 物理模型

### 3.1 运动和真值

在 200 Hz 时间轴上生成平滑的 \(SE(3)\) 相机轨迹。棋盘坐标系固定时，真值为：

\[
R_{BC}=R_{WB}^{T}R_{WC},\qquad
p_{BC}=R_{WB}^{T}(p_{WC}-p_{WB}).
\]

右目默认使用相同旋转和相机 X 轴方向 46.5 mm 基线。这个外参目前仍是名义值，不能用于最终论文数据，必须先重新做真实双目标定。

### 3.2 IMU

IMU 姿态：

\[
R_{WI}=R_{WC}R_{CI}.
\]

加速度计输出比力：

\[
f_I=R_{WI}^{T}(\ddot p_{WI}-g_W).
\]

角速度由相邻 \(R_{WI}\) 的对数映射计算。IMU 杆臂会自动产生切向和向心加速度。

噪声配置 `datasets/sim_assets/imu_profile_usb.json` 来自
`rigid_20260924_142902` 开头 5 秒的真实静止数据：

- 采样率约 200.01 Hz；
- 加速度计逐样本白噪声约 0.0014/0.0015/0.0030 g；
- 陀螺逐样本白噪声约 0.029/0.137/0.018 °/s；
- 量化、1 秒偏置随机游走和静态陀螺偏置均写入配置。
- 三轴白噪声和偏置随机游走使用完整协方差矩阵，不再假设轴间独立；
- IMU 时间戳按实测约 0.68 ms 标准差产生不规则采样；
- 每段同时导出 noisy 和 ideal IMU，自动检查所有轴是否符合刚体运动学。

可运行 `tools/fit_imu_sim_profile.py` 重新拟合。

### 3.3 相机

左右目分别读取真实内参和畸变。渲染包含：

- 棋盘三维透视；
- 左右目基线；
- 径向/切向畸变；
- 空间光照梯度和时间闪烁；
- 暗角、图像噪声、运动模糊；
- 组织风格低频背景纹理。

## 4. 导出结构

```text
sim_YYYYMMDD_HHMMSS_sSEED/
  cam0/images/000000.jpg
  cam0/times.txt
  cam1/images/000000.jpg
  cam1/times.txt
  imu_stream.csv
  imu_bt.csv
  ground_truth.npz
  session_meta.json
```

`ground_truth.npz` 同时保存：

- `R, p`：相机在棋盘坐标系中的精确姿态；
- `R_W_C, p_W_C`：世界坐标系相机姿态；
- `R_W_B, p_W_B`：棋盘姿态；
- 相机和 IMU 时间戳。

无界面批量生成：

```powershell
D:\anaconda\python.exe tools\generate_synthetic.py `
  --duration 60 --motion random_spline --seed 1001
```

## 5. 数据集设计

不要随机拆窗口，必须按完整仿真序列和随机种子拆分：

- synthetic-train：运动类型、光照、噪声范围内随机化；
- synthetic-val：未见过的种子和参数组合；
- synthetic-test-ood：更强噪声、棋盘运动、部分出画、极端光照；
- real-test：真实预先封存录像，绝不参与仿真参数选择。

建议第一轮每种运动生成 100 段 × 30 秒，合计约 4.2 小时。图像可以按 15 fps 导出，IMU 保持 200 Hz。训练时先做 synthetic pretrain，再用真实训练段 fine-tune；论文必须同时报告 synthetic、real validation 和 sealed real test。

当前已生成并审计：

- 95 段 IMU+真值序列（60 train、15 val、15 test、5 OOD）；
- 20 段 OpenCV 双目代表序列；
- 5 段 Blender 1080p 代表序列；
- 五种运动均覆盖，所有 split 的 seed 无重叠；
- 95/95 IMU 序列、25/25 双目序列通过自动检查；
- 棋盘检测率中位数 100%，最低 50%；
- 检测角点相对精确真值的 RMSE 中位数 0.252 px、p90 0.842 px。

后续扩展语料增加到 12 种运动，共 156 段 IMU+真值序列，全部通过物理和 split 审计。长距离慢速段可覆盖约 0.8 m，剧烈 6DoF 段覆盖最高约 8 g 和 1400 °/s（仍在 ±16 g、±2000 °/s 量程内）。扩展语料没有改善当前两段验证集，因此不替换最终模型；它保留用于未来采集到对应真实主题后的再微调，避免为了现有验证分布删除有效极端运动。

第一次联合训练暴露了域差距：初版仿真转动过多、平移加速度过小。最终语料把朝向跟随降到 0.02–0.15，并使用低转角为主、少量高转角的混合分布。真实与调节后仿真的 IMU 百分位见 `synthetic-training-results.canvas.tsx`。

生成和审计命令：

```powershell
D:\anaconda\python.exe tools\generate_synthetic_corpus.py
$root = Get-Content datasets\synthetic_corpus\latest.txt
D:\anaconda\python.exe tools\audit_synthetic_corpus.py $root
```

## 6. 当前不能忽略的两个参数

1. **右目外参**：现有名义 46.5 mm 与真实 PnP 不一致。最终仿真前需重新完成双目外参标定。
2. **IMU 杆臂**：当前默认 0 mm。请测量左目光心到 USB IMU 中心的 X/Y/Z 距离并填入界面；旋转运动时该参数会显著影响加速度。

## 7. Blender 后端

已经安装并接入 Blender 5.2.1 LTS。界面选择 **Blender Eevee** 后，导出过程会：

- 在 Blender 中建立毫米尺度三维棋盘；
- 建立组织风格程序化材质和平面；
- 使用左右目独立内参、相机位姿和名义双目基线；
- 使用面光源、补光、阴影和 AgX 高对比色彩管理；
- 渲染后再施加真实镜头畸变；
- 保持和 OpenCV 后端相同的 IMU、真值与目录格式。

720p、1080p 和 2K 双目单帧均已通过导出与棋盘检测检查。后续还可增加真实组织网格、rolling shutter、曝光时间、方向性运动模糊和 BlenderProc 域随机化。

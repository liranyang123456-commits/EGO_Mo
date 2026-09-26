# EGO_Mo 采集套件

内镜视场双目相机 + **USB 维特 IMU** + **蓝牙 BWT901CL / WTSDCL** 同步采集，用来做 ego-motion、VIO 和三维重建。

## 从哪里迁来

对话 `1d0149c5-d486-4837-9fe3-e6198126bdfb` 对应旧工作区 `E:\elsarticle-templateCMBP_IMU_Camera`。那套设备的界面程序在：

`D:\reloc3r\stereo_endoscope_capture_gui.py`

已迁到本目录并重写：双 IMU、断线重连、录像写盘与抓图分离、蓝牙 200Hz。

## 启动

```bat
E:\EGO_Mo\run_capture.bat
```

或：

```bat
python E:\EGO_Mo\capture_app.py
```

依赖：`pip install -r requirements.txt`（本机已有 OpenCV / pyserial / Pillow / bleak）。

## 设备

| 设备 | 本机识别 | 数据率 |
|---|---|---|
| USB 维特（CP210x） | `COM3` @ 921600 | 已约 200Hz，启动时再写 200Hz |
| 蓝牙 IMU | BLE 名 `WTSDCL`，地址 `F9:5A:7B:FD:65:A0` | 协议 `0x55 0x61`（28 字节），命令 `0x0B` 后约 200Hz |
| 内镜相机 | DirectShow 索引 0/1/2 | 目标 1080p@30，Hub 不够用 720p |

产品页写的是 **BWT901CL（经典蓝牙 2.0 / 115200 SPP）**。当前电脑配对到的是 **WTSDCL（BLE 5 服务 FFE5/FFE4/FFE9）**，没有出现 SPP COM 口。界面两种都支持：有 COM 走 115200，有 BLE 走 GATT。

## 界面改动（稳定性）

- 相机抓图线程和 AVI/JPEG/CSV 写盘线程分开，写盘卡住时丢的是队列而不是实时预览
- USB / BLE IMU 各自线程，串口或 GATT 掉线自动重连并重新下发 200Hz
- 相机连续读失败会自动重开
- 门禁不再卡死主界面
- 单目也能过门禁；USB 与 BLE 可只开一路
- 时间戳按 200Hz 回推，避免串口/BLE 粘包把一堆样本打成同一时刻

## 一次采集目录

`E:\EGO_Mo\datasets\<kind>_YYYYMMDD_HHMMSS\`

- `cam0/video.avi`、`cam0/images/000000.jpg`、`cam0/times.txt`
- `imu_stream.csv`：USB IMU（兼容旧 reloc3r 字段）
- `imu_bt.csv`：蓝牙 IMU
- `frames.csv`：每帧对齐两路 IMU 下标与时间差
- `session_meta.json`、`checkerboard.yaml`

## 三系标定顺序（先做这个）

坐标系：`I_usb`（绑在双目上）→ `I_bt`（蓝牙）→ `C0/C1`（左右光心）→ `O`（棋盘）。
双目光心基线 **46.5 mm** = 外壳外侧 52 mm − 镜头直径 5.5 mm。

1. 扫描 → 指定左右目 → 打开相机+双 IMU → 门禁
2. **双IMU静止对齐**：两枚贴紧、同向，整套不动 30–45 s → `R(I_usb←I_bt)`（重力+磁力）
3. **双IMU动态对齐**：仍贴在一起，平移并绕三轴转 45–90 s → 手眼精化旋转（文档里的 AX=XB）
4. **双目内参**：蓝牙拿开，GP050 慢扫
5. **棋盘+蓝牙IMU**：蓝牙粘在棋盘上，双目里多姿态 2–3 min → `T(C0←I_bt)` + PnP
6. **三系投影**：合成 `T(C0←I_usb)`、`T(C1←C0)`，写入 `datasets/rig_state.json`

然后再采 EgoMotion / 三维重建。参考 `E:\IMU_Camera_MR_Navigation_System(2).docx` 与旧工作 `1d0149c5` 的 PnP / R_CI。

## 坐标系投影模型（不要单独 MLP 当主模型）

手眼 AX=XB **能解常值旋转**。解不了的是时延、滤波滞后、磁场畸变、安装微弹性。网络只学残差：

`R_C = R_geo @ R_res`，`t_C = t_geo + R_geo t_res`

| 结构 | 用途 |
|---|---|
| 单独 flatten MLP | 消融基线。丢掉时间顺序，欧拉角会跳，不作为主模型 |
| 双流 GRU | 与论文 GRU6D 同族，200Hz 短窗样本效率高 |
| **GRU + 交叉注意力** | 推荐。Transformer 只用在 USB↔蓝牙对齐，不替代时序编码 |

实现：`ego_capture/mapping/`。有棋盘+双 IMU 窗口标签后再训。

## 数字孪生与合成数据

双击桌面 **EGO_Mo 仿真**，或运行：

```bat
E:\EGO_Mo\run_simulator.bat
```

界面可配置三维运动、双目相机、光照、图像噪声、IMU 噪声和棋盘运动，并显示左右目、三维轨迹和 200 Hz IMU 曲线。支持 720p、1080p、2K/QHD，以及 OpenCV CPU / Blender Eevee 两套导出后端。导出结构与真实采集相近，另带精确 `ground_truth.npz`。

无界面批量生成：

```bat
D:\anaconda\python.exe tools\generate_synthetic.py --duration 60 --motion random_spline --seed 1001
```

详细物理模型、参数来源和数据集划分见 `docs/simulation_protocol.md`。

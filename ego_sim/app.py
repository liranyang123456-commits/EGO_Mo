#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PySide6 visual console for synthetic stereo/IMU generation."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import cv2
import numpy as np
from matplotlib import rcParams
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from .core import SimConfig, StereoRenderer, build_sequence, export_sequence

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False


class ExportWorker(QtCore.QObject):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, sequence, output: Path) -> None:
        super().__init__()
        self.sequence = sequence
        self.output = output

    @QtCore.Slot()
    def run(self) -> None:
        try:
            path = export_sequence(
                self.sequence,
                self.output,
                lambda done, total: self.progress.emit(int(round(done * 100 / max(total, 1)))),
            )
            self.finished.emit(str(path))
        except Exception:
            self.failed.emit(traceback.format_exc())


class SimulationWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EGO_Mo 数字孪生数据生成器")
        self.resize(1560, 940)
        self.sequence = None
        self.renderer = None
        self._thread = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._advance)
        self._build_ui()
        self.generate_preview()

    def _spin(self, low, high, value, suffix="", decimals=1):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(low, high)
        box.setValue(value)
        box.setDecimals(decimals)
        box.setSuffix(suffix)
        box.setSingleStep(0.1 if decimals else 1)
        return box

    def _build_ui(self) -> None:
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QHBoxLayout(root)

        controls = QtWidgets.QGroupBox("仿真参数")
        controls.setMaximumWidth(330)
        form = QtWidgets.QFormLayout(controls)
        self.motion = QtWidgets.QComboBox()
        self.motion.addItems([
            "mixed", "lateral", "depth", "circle", "random_spline",
            "slow_far", "aggressive_6dof", "pure_rotation",
            "translation_xyz", "pitch_sweep", "stop_go", "spiral",
        ])
        self.resolution = QtWidgets.QComboBox()
        self.resolution.addItems([
            "1280×720（720p）",
            "1920×1080（1080p）",
            "2560×1440（2K/QHD）",
            "640×480（快速预览）",
        ])
        self.backend = QtWidgets.QComboBox()
        self.backend.addItems(["OpenCV CPU", "Blender Eevee"])
        self.backend.setToolTip("预览始终使用快速 OpenCV；导出时可选择 Blender Eevee。")
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([
            "真实分布", "高速运动", "前后推进", "长距离慢速",
            "剧烈 6DoF", "纯旋转", "俯仰扫描", "棋盘运动 OOD", "静态标定",
        ])
        self.preset.currentIndexChanged.connect(self.apply_preset)
        self.duration = self._spin(2, 300, 12, " s")
        self.fps = self._spin(5, 60, 30, " fps")
        self.imu_hz = self._spin(50, 1000, 200, " Hz", 0)
        self.seed = QtWidgets.QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(7)
        self.translation = self._spin(0, 600, 30, " mm")
        self.depth = self._spin(100, 1500, 150, " mm")
        self.depth_change = self._spin(0, 1000, 20, " mm")
        self.rotation = self._spin(0, 180, 8, "°")
        self.tracking_gain = self._spin(0, 1, 0.35, "×", 2)
        self.speed = self._spin(0.02, 6, 1.0, "×", 2)
        self.light = self._spin(0.3, 2.0, 1.0, "×")
        self.flicker = self._spin(0, 0.5, 0.08, "", 2)
        self.noise = self._spin(0, 4, 1.0, "×")
        self.image_noise = self._spin(0, 15, 2.0, " px")
        self.board_motion = QtWidgets.QCheckBox("棋盘运动（负样本/OOD）")
        self.board_motion.setToolTip("棋盘自身运动不会被相机 IMU 观测到，不能作为正常 ego-motion 训练标签。")
        self.board_motion_mm = self._spin(0, 40, 8, " mm")
        self.lever_x = self._spin(-200, 200, 0, " mm")
        self.lever_y = self._spin(-200, 200, 0, " mm")
        self.lever_z = self._spin(-200, 200, 0, " mm")
        rows = [
            ("参数预设", self.preset), ("运动类型", self.motion),
            ("分辨率", self.resolution), ("导出后端", self.backend),
            ("时长", self.duration), ("相机帧率", self.fps), ("IMU 频率", self.imu_hz),
            ("随机种子", self.seed), ("平移幅度", self.translation), ("基准距离", self.depth),
            ("深度变化", self.depth_change), ("独立转角", self.rotation),
            ("朝向跟随", self.tracking_gain), ("运动速度", self.speed),
            ("照明", self.light), ("光照闪烁", self.flicker), ("IMU 噪声", self.noise),
            ("图像噪声", self.image_noise), ("棋盘幅度", self.board_motion_mm),
            ("IMU 杆臂 X", self.lever_x), ("IMU 杆臂 Y", self.lever_y), ("IMU 杆臂 Z", self.lever_z),
        ]
        for label, widget in rows:
            form.addRow(label, widget)
        form.addRow(self.board_motion)

        self.preview_btn = QtWidgets.QPushButton("生成 / 刷新预览")
        self.preview_btn.clicked.connect(self.generate_preview)
        self.export_btn = QtWidgets.QPushButton("导出兼容数据集")
        self.export_btn.clicked.connect(self.export_dataset)
        self.play_btn = QtWidgets.QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        form.addRow(self.preview_btn)
        form.addRow(self.export_btn)
        form.addRow(self.play_btn)
        self.progress = QtWidgets.QProgressBar()
        form.addRow(self.progress)
        self.status = QtWidgets.QLabel("准备中")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        layout.addWidget(controls)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        center = QtWidgets.QWidget()
        center_layout = QtWidgets.QVBoxLayout(center)
        views = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.left_view = QtWidgets.QLabel("左目")
        self.right_view = QtWidgets.QLabel("右目")
        for view in (self.left_view, self.right_view):
            view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            view.setMinimumSize(420, 280)
            view.setStyleSheet("background:#11151d;color:#c9d2e3;border:1px solid #3b4455;")
            views.addWidget(view)
        center_layout.addWidget(views, 3)
        timeline = QtWidgets.QHBoxLayout()
        timeline.addWidget(QtWidgets.QLabel("时间"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.render_index)
        timeline.addWidget(self.slider)
        self.time_label = QtWidgets.QLabel("0.00 s")
        timeline.addWidget(self.time_label)
        center_layout.addLayout(timeline)

        self.figure = Figure(facecolor="#151922")
        self.canvas = FigureCanvasQTAgg(self.figure)
        center_layout.addWidget(self.canvas, 2)
        self.tabs.addTab(center, "仿真预览")

        trajectory_tab = QtWidgets.QWidget()
        trajectory_layout = QtWidgets.QVBoxLayout(trajectory_tab)
        trajectory_controls = QtWidgets.QHBoxLayout()
        trajectory_controls.addWidget(QtWidgets.QLabel("数据集"))
        self.trajectory_dataset = QtWidgets.QComboBox()
        self.trajectory_dataset.addItems(["真实固定测试", "仿真测试"])
        self.trajectory_dataset.currentIndexChanged.connect(self.plot_trajectory_comparison)
        trajectory_controls.addWidget(self.trajectory_dataset)
        trajectory_controls.addWidget(QtWidgets.QLabel("显示方法"))
        self.trajectory_methods = QtWidgets.QListWidget()
        self.trajectory_methods.setMaximumHeight(92)
        friendly = [
            ("ground_truth", "真实轨迹", True),
            ("zero", "零运动", False),
            ("ridge", "线性 Ridge", True),
            ("zero_shot", "纯仿真 Zero-shot", False),
            ("real_only", "仅真实训练", True),
            ("fine_tuned", "仿真预训练 + 微调", True),
            ("blend", "最终混合模型", True),
            ("ronin_fine", "RoNIN-ResNet 微调", False),
            ("imunet_real", "IMUNet 真实训练", True),
            ("architecture_blend", "跨架构验证集成", True),
            ("strapdown", "捷联积分", False),
        ]
        for key, label, checked in friendly:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            self.trajectory_methods.addItem(item)
        self.trajectory_methods.itemChanged.connect(self.plot_trajectory_comparison)
        trajectory_controls.addWidget(self.trajectory_methods, 1)
        trajectory_layout.addLayout(trajectory_controls)
        self.trajectory_figure = Figure(facecolor="#151922")
        self.trajectory_canvas = FigureCanvasQTAgg(self.trajectory_figure)
        trajectory_layout.addWidget(self.trajectory_canvas, 1)
        self.trajectory_table = QtWidgets.QTableWidget()
        self.trajectory_table.setMaximumHeight(190)
        trajectory_layout.addWidget(self.trajectory_table)
        self.tabs.addTab(trajectory_tab, "真实 / 网络轨迹")

        benchmark_tab = QtWidgets.QWidget()
        benchmark_layout = QtWidgets.QVBoxLayout(benchmark_tab)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("比较数据集"))
        self.benchmark_dataset = QtWidgets.QComboBox()
        self.benchmark_dataset.addItems(["真实验证", "真实固定测试", "仿真测试", "仿真 OOD"])
        self.benchmark_dataset.currentIndexChanged.connect(self.fill_benchmark_table)
        row.addWidget(self.benchmark_dataset)
        row.addStretch()
        benchmark_layout.addLayout(row)
        self.benchmark_table = QtWidgets.QTableWidget()
        benchmark_layout.addWidget(self.benchmark_table)
        benchmark_layout.addWidget(QtWidgets.QLabel("公开方法参考（协议不同，不参与同表排名）"))
        self.literature_table = QtWidgets.QTableWidget()
        self.literature_table.setMaximumHeight(180)
        benchmark_layout.addWidget(self.literature_table)
        self.benchmark_note = QtWidgets.QLabel(
            "直接跑模型 = 纯仿真 Zero-shot；微调 = 仿真预训练后使用真实训练段微调。"
            "RoNIN/TLIO 的公开权重属于行人运动和不同坐标协议，不能伪装成同协议数值。"
        )
        self.benchmark_note.setWordWrap(True)
        benchmark_layout.addWidget(self.benchmark_note)
        self.tabs.addTab(benchmark_tab, "方法比较")

        dataset_tab = QtWidgets.QWidget()
        dataset_layout = QtWidgets.QVBoxLayout(dataset_tab)
        self.dataset_text = QtWidgets.QTextEdit()
        self.dataset_text.setReadOnly(True)
        dataset_layout.addWidget(self.dataset_text)
        self.tabs.addTab(dataset_tab, "数据集与物理检查")
        self.setStyleSheet(
            "QMainWindow,QWidget{background:#202634;color:#e8edf7;}"
            "QGroupBox{font-weight:bold;border:1px solid #465166;margin-top:8px;padding-top:8px;}"
            "QPushButton{background:#315e9f;padding:8px;border-radius:3px;}"
            "QPushButton:disabled{background:#4b5160;}"
            "QComboBox,QSpinBox,QDoubleSpinBox{background:#171c27;padding:3px;}"
            "QTableWidget,QListWidget,QTextEdit{background:#151922;gridline-color:#465166;}"
        )
        self._load_evaluation_views()

    @QtCore.Slot()
    def apply_preset(self) -> None:
        if not hasattr(self, "translation"):
            return
        index = self.preset.currentIndex()
        presets = [
            ("mixed", 30, 150, 20, 8, 0.35, 1.0, 1.0, 0.08, False),
            ("mixed", 65, 180, 55, 18, 0.20, 2.5, 1.4, 0.14, False),
            ("depth", 15, 190, 75, 5, 0.12, 1.4, 1.0, 0.06, False),
            ("slow_far", 350, 450, 400, 4, 0.08, 0.12, 1.0, 0.03, False),
            ("aggressive_6dof", 120, 240, 120, 65, 0.30, 3.5, 1.8, 0.18, False),
            ("pure_rotation", 5, 170, 0, 90, 0.0, 1.3, 1.2, 0.08, False),
            ("pitch_sweep", 10, 170, 0, 45, 0.0, 0.8, 1.0, 0.05, False),
            ("random_spline", 45, 180, 45, 15, 0.30, 2.0, 2.5, 0.25, True),
            ("lateral", 0, 160, 0, 0, 0.0, 0.2, 1.0, 0.0, False),
        ]
        motion, trans, depth, depth_change, rotation, tracking, speed, noise, flicker, board = presets[index]
        self.motion.setCurrentText(motion)
        self.translation.setValue(trans)
        self.depth.setValue(depth)
        self.depth_change.setValue(depth_change)
        self.rotation.setValue(rotation)
        self.tracking_gain.setValue(tracking)
        self.speed.setValue(speed)
        self.noise.setValue(noise)
        self.flicker.setValue(flicker)
        self.board_motion.setChecked(board)

    def config(self) -> SimConfig:
        resolutions = [(1280, 720), (1920, 1080), (2560, 1440), (640, 480)]
        width, height = resolutions[self.resolution.currentIndex()]
        return SimConfig(
            duration_s=self.duration.value(),
            camera_fps=self.fps.value(),
            imu_hz=self.imu_hz.value(),
            width=width,
            height=height,
            seed=self.seed.value(),
            motion=self.motion.currentText(),
            translation_mm=self.translation.value(),
            depth_mm=self.depth.value(),
            depth_change_mm=self.depth_change.value(),
            rotation_deg=self.rotation.value(),
            tracking_gain=self.tracking_gain.value(),
            speed=self.speed.value(),
            light_level=self.light.value(),
            light_flicker=self.flicker.value(),
            noise_scale=self.noise.value(),
            image_noise_std=self.image_noise.value(),
            board_motion=self.board_motion.isChecked(),
            board_motion_mm=self.board_motion_mm.value(),
            imu_lever_arm_mm_x=self.lever_x.value(),
            imu_lever_arm_mm_y=self.lever_y.value(),
            imu_lever_arm_mm_z=self.lever_z.value(),
            render_backend="blender" if self.backend.currentIndex() == 1 else "opencv",
        )

    @QtCore.Slot()
    def generate_preview(self) -> None:
        try:
            self.sequence = build_sequence(self.config())
            self.renderer = StereoRenderer(self.sequence.config)
            self.slider.setRange(0, len(self.sequence.poses.t) - 1)
            self.slider.setValue(0)
            self._plot()
            self.render_index(0)
            gyro = np.linalg.norm(self.sequence.imu_usb[:, 3:6], axis=1)
            lin = np.linalg.norm(self.sequence.imu_usb[:, 0:3], axis=1)
            self.status.setText(
                f"{len(self.sequence.poses.t)} 个 IMU 样本；"
                f"最大角速度 {gyro.max():.1f}°/s；加速度模长 {lin.min():.2f}–{lin.max():.2f} g。"
            )
        except Exception as exc:
            self.status.setText(str(exc))
            QtWidgets.QMessageBox.critical(self, "生成失败", traceback.format_exc())

    def _plot(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(121, projection="3d")
        p = self.sequence.poses.p_W_C * 1000.0
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#5aa9ff", linewidth=1.4, label="camera")
        bw = self.sequence.config.board_cols * self.sequence.config.square_mm / 2
        bh = self.sequence.config.board_rows * self.sequence.config.square_mm / 2
        ax.plot([-bw, bw, bw, -bw, -bw], [-bh, -bh, bh, bh, -bh], [0] * 5, color="#f2c94c", label="board")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title("3D camera trajectory")
        ax.legend(loc="upper right")
        ax.set_facecolor("#151922")

        ax2 = self.figure.add_subplot(122)
        t = self.sequence.poses.t
        acc = self.sequence.imu_usb[:, 0:3]
        gyro = self.sequence.imu_usb[:, 3:6]
        ax2.plot(t, acc[:, 0], label="acc x (g)", linewidth=0.8)
        ax2.plot(t, acc[:, 1], label="acc y (g)", linewidth=0.8)
        ax2.plot(t, acc[:, 2], label="acc z (g)", linewidth=0.8)
        ax2b = ax2.twinx()
        ax2b.plot(t, gyro[:, 0], "--", label="gyro x", linewidth=0.7, alpha=0.75)
        ax2b.plot(t, gyro[:, 1], "--", label="gyro y", linewidth=0.7, alpha=0.75)
        ax2b.plot(t, gyro[:, 2], "--", label="gyro z", linewidth=0.7, alpha=0.75)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Acceleration (g)")
        ax2b.set_ylabel("Angular velocity (deg/s)")
        ax2.set_title("Synthetic 200 Hz IMU — all axes")
        lines = ax2.lines + ax2b.lines
        ax2.legend(lines, [line.get_label() for line in lines], fontsize=6, ncol=2)
        ax2.grid(alpha=0.2)
        ax2.set_facecolor("#151922")
        self.figure.tight_layout()
        self.canvas.draw()

    def _load_evaluation_views(self) -> None:
        self._trajectory_summary = {}
        datasets = Path(__file__).resolve().parents[1] / "datasets"
        trajectory_summary = datasets / "trajectory_comparison" / "summary.json"
        if trajectory_summary.is_file():
            self._trajectory_summary = json.loads(trajectory_summary.read_text(encoding="utf-8"))
        benchmark_path = datasets / "method_benchmark_20260924.json"
        self._benchmark = (
            json.loads(benchmark_path.read_text(encoding="utf-8"))
            if benchmark_path.is_file() else {"datasets": {}}
        )
        external_root = datasets / "external_benchmark"
        external_test = external_root / "test_benchmark.json"
        self._external_test = (
            json.loads(external_test.read_text(encoding="utf-8"))
            if external_test.is_file() else {}
        )
        self._external_val = {}
        for arch in ("ronin_resnet", "ronin_lstm", "tlio_resnet", "imunet"):
            path = external_root / f"{arch}_s0.json"
            if path.is_file():
                self._external_val[arch] = json.loads(path.read_text(encoding="utf-8"))
        optimized_path = external_root / "optimized_ensemble.json"
        self._optimized_architecture = (
            json.loads(optimized_path.read_text(encoding="utf-8"))
            if optimized_path.is_file() else {}
        )
        self.plot_trajectory_comparison()
        self.fill_benchmark_table()
        self.fill_literature_table()
        self.fill_dataset_summary()

    def _selected_trajectory_methods(self) -> list[str]:
        selected = []
        for i in range(self.trajectory_methods.count()):
            item = self.trajectory_methods.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return selected

    @QtCore.Slot()
    def plot_trajectory_comparison(self) -> None:
        if not hasattr(self, "trajectory_figure"):
            return
        case = "real_test" if self.trajectory_dataset.currentIndex() == 0 else "synthetic_test"
        path = Path(__file__).resolve().parents[1] / "datasets" / "trajectory_comparison" / f"{case}.npz"
        if not path.is_file():
            return
        data = np.load(path)
        selected = self._selected_trajectory_methods()
        labels = {
            "ground_truth": "真实轨迹",
            "zero": "零运动",
            "ridge": "线性 Ridge",
            "zero_shot": "纯仿真 Zero-shot",
            "real_only": "仅真实训练",
            "fine_tuned": "仿真预训练 + 微调",
            "blend": "最终混合",
            "ronin_fine": "RoNIN-ResNet 微调",
            "imunet_real": "IMUNet 真实训练",
            "architecture_blend": "跨架构验证集成",
            "strapdown": "捷联积分",
        }
        colors = {
            "ground_truth": "#ffffff", "zero": "#8892a6", "ridge": "#f2c94c",
            "zero_shot": "#d66efd", "real_only": "#56b4e9",
            "fine_tuned": "#37c978", "blend": "#ff7f50", "strapdown": "#ef5350",
            "ronin_fine": "#00acc1", "imunet_real": "#ab47bc",
            "architecture_blend": "#ff4081",
        }
        self.trajectory_figure.clear()
        ax3 = self.trajectory_figure.add_subplot(221, projection="3d")
        axes = [
            self.trajectory_figure.add_subplot(222),
            self.trajectory_figure.add_subplot(223),
            self.trajectory_figure.add_subplot(224),
        ]
        t = data["t"] - data["t"][0]
        for method in selected:
            if method not in data.files:
                continue
            p = data[method] * 1000.0
            mask = np.isfinite(p).all(axis=1)
            width = 2.4 if method == "ground_truth" else 1.2
            ax3.plot(
                p[:, 0], p[:, 1], p[:, 2],
                label=labels[method], color=colors[method], linewidth=width,
            )
            for axis, name in enumerate(("X", "Y", "Z")):
                axes[axis].plot(t[mask], p[mask, axis], color=colors[method], linewidth=width, label=labels[method])
                axes[axis].set_ylabel(f"{name} (mm)")
                axes[axis].set_xlabel("Time (s)")
                axes[axis].grid(alpha=0.18)
                axes[axis].set_facecolor("#151922")
        ax3.set_xlabel("X (mm)")
        ax3.set_ylabel("Y (mm)")
        ax3.set_zlabel("Z (mm)")
        ax3.set_title("真实轨迹与网络预测轨迹")
        ax3.legend(fontsize=7)
        ax3.set_facecolor("#151922")
        if selected:
            axes[0].legend(fontsize=6, ncol=2)
        self.trajectory_figure.tight_layout()
        self.trajectory_canvas.draw()
        self._fill_trajectory_table(case, labels)

    def _fill_trajectory_table(self, case: str, labels: dict[str, str]) -> None:
        metrics = self._trajectory_summary.get(case, {}).get("metrics", {})
        columns = ["方法", "有效节点", "ATE RMSE (mm)", "ATE 平均 (mm)", "ATE p90 (mm)"]
        self.trajectory_table.setColumnCount(len(columns))
        self.trajectory_table.setHorizontalHeaderLabels(columns)
        self.trajectory_table.setRowCount(len(metrics))
        for row, (method, values) in enumerate(metrics.items()):
            entries = [
                labels.get(method, method),
                values["nodes"],
                values["ate_rmse_mm"],
                values["ate_mean_mm"],
                values["ate_p90_mm"],
            ]
            for col, value in enumerate(entries):
                self.trajectory_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(value)))
        self.trajectory_table.resizeColumnsToContents()

    @QtCore.Slot()
    def fill_benchmark_table(self) -> None:
        keys = ("real_validation", "real_test", "synthetic_test", "synthetic_ood")
        key = keys[self.benchmark_dataset.currentIndex()]
        metrics = self._benchmark.get("datasets", {}).get(key, {})
        labels = {
            "zero_motion": "零运动",
            "strapdown_zero_velocity": "捷联积分（零初速）",
            "ridge_real_fitted": "真实数据 Ridge",
            "synthetic_zero_shot": "纯仿真 Zero-shot",
            "real_only_network": "仅真实训练",
            "synthetic_pretrain_real_finetune": "仿真预训练 + 真实微调",
            "validation_blend": "最终验证集混合",
        }
        order = list(labels)
        external_rows = []
        external_labels = {
            "ronin_resnet": "RoNIN-ResNet",
            "ronin_lstm": "RoNIN-LSTM",
            "tlio_resnet": "TLIO-ResNet",
            "imunet": "IMUNet",
        }
        if key in {"real_test", "synthetic_test"}:
            for arch, label in external_labels.items():
                block = self._external_test.get(arch, {})
                for mode, suffix in (("zero_shot", "direct"), ("fine_tuned", "fine-tuned")):
                    if key in block.get(mode, {}):
                        external_rows.append((f"{label} ({suffix})", block[mode][key]))
        elif key == "real_validation":
            for arch, label in external_labels.items():
                block = self._external_val.get(arch, {})
                for metric_key, suffix in (
                    ("synthetic_zero_shot_real_val", "direct"),
                    ("fine_tuned_real_val", "fine-tuned"),
                ):
                    if metric_key in block:
                        external_rows.append((f"{label} ({suffix})", block[metric_key]))
        if key == "real_validation" and "validation" in self._optimized_architecture:
            external_rows.append(("跨架构验证集成", self._optimized_architecture["validation"]))
        elif key == "real_test" and "test" in self._optimized_architecture:
            external_rows.append(("跨架构验证集成", self._optimized_architecture["test"]))
        columns = ["方法", "样本", "平均误差 (mm)", "快运动误差 (mm)", "预测中位数 (mm)", "R²"]
        self.benchmark_table.setColumnCount(len(columns))
        self.benchmark_table.setHorizontalHeaderLabels(columns)
        self.benchmark_table.setRowCount(len(order) + len(external_rows))
        for row, method in enumerate(order):
            value = metrics.get(method, {})
            entries = [
                labels[method], value.get("n", "-"), value.get("err_mm", "-"),
                value.get("fast_mm", "-"), value.get("pred_med_mm", "-"), value.get("r2", "-"),
            ]
            for col, entry in enumerate(entries):
                self.benchmark_table.setItem(row, col, QtWidgets.QTableWidgetItem(str(entry)))
        for offset, (label, value) in enumerate(external_rows, start=len(order)):
            entries = [
                label, value.get("n", "-"), value.get("err_mm", "-"),
                value.get("fast_mm", "-"), value.get("pred_med_mm", "-"), value.get("r2", "-"),
            ]
            for col, entry in enumerate(entries):
                self.benchmark_table.setItem(offset, col, QtWidgets.QTableWidgetItem(str(entry)))
        self.benchmark_table.resizeColumnsToContents()

    def fill_literature_table(self) -> None:
        columns = ["方法", "输入/运动", "论文指标", "与本项目关系"]
        rows = [
            ["IONet (AAAI 2018)", "手机 IMU / 行人步行", "约 90% 时间误差 < 2 m（2 min）", "运动先验和尺度完全不同"],
            ["RoNIN (ICRA 2020)", "手机 IMU / 行人步行", "seen 1-min RTE 约 2.63 m", "公开权重不能直接迁移到相机慢速运动"],
            ["TLIO (RA-L 2020)", "头戴 IMU / 行走", "1-s 位移误差 σ≈51/51/13 mm", "结构最接近，但坐标系和训练域不同"],
            ["Ours direct", "本设备仿真预训练", "真实测试 90.80 mm，R²=-2.097", "真正的 zero-shot 直接运行"],
            ["Ours fine-tuned/blend", "仿真预训练 + 真实微调", "真实测试 45.45 mm，R²=0.166", "同设备、同标签、同 3-s 协议"],
        ]
        self.literature_table.setColumnCount(len(columns))
        self.literature_table.setHorizontalHeaderLabels(columns)
        self.literature_table.setRowCount(len(rows))
        for r, values in enumerate(rows):
            for c, value in enumerate(values):
                self.literature_table.setItem(r, c, QtWidgets.QTableWidgetItem(str(value)))
        self.literature_table.resizeColumnsToContents()

    def fill_dataset_summary(self) -> None:
        datasets = Path(__file__).resolve().parents[1] / "datasets"
        audits = sorted((datasets / "synthetic_corpus").glob("corpus_*/audit.json"))
        summaries = []
        for path in audits[-4:]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summaries.append((path.parent.name, payload["summary"]))
        profile_path = datasets / "sim_assets" / "imu_profile_usb.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.is_file() else {}
        split_path = datasets / "trajectory_split_20260924.json"
        split = json.loads(split_path.read_text(encoding="utf-8")) if split_path.is_file() else {}
        lines = [
            "EGO_Mo 数据集与物理检查",
            "=" * 48,
            "",
            f"真实数据：train={len(split.get('train', []))}，val={len(split.get('val', []))}，test={len(split.get('test', []))}",
            "真实测试整段封存，不参与模型和仿真参数选择。",
            "",
            "IMU 数字孪生参数",
            f"来源：{profile.get('source_session', '-')}",
            f"采样率：{profile.get('sample_rate_hz', 0):.3f} Hz",
            f"采样间隔标准差：{profile.get('sample_dt_std_ms', 0):.3f} ms",
        ]
        for name, summary in summaries:
            lines += [
                "",
                f"合成语料 {name}",
                f"  IMU：{summary['imu_passed']}/{summary['imu_sequences']} 通过",
                f"  视觉：{summary['visual_passed']}/{summary['visual_sequences']} 通过",
                f"  棋盘检测率中位数：{summary.get('detection_rate_median')}",
                f"  角点真值 RMSE：{summary.get('corner_rmse_px_median')} px",
                f"  split seed 重叠：{summary['seed_overlap']}",
            ]
        lines += [
            "",
            "物理约束",
            "  specific force = R_WI^T (a_world - gravity)",
            "  gyro = Log(R_i^T R_i+1) / dt",
            "  已启用不规则采样、三轴噪声协方差、偏置随机游走、量化和杆臂。",
            "",
            "注意：棋盘运动 OOD 对相机 IMU 不可观测，只用于失败检测。",
        ]
        self.dataset_text.setPlainText("\n".join(lines))

    @staticmethod
    def _pixmap(image: np.ndarray, target: QtWidgets.QLabel) -> QtGui.QPixmap:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimage = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
        return QtGui.QPixmap.fromImage(qimage).scaled(
            target.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

    @QtCore.Slot(int)
    def render_index(self, index: int) -> None:
        if self.sequence is None or self.renderer is None:
            return
        k = min(max(index, 0), len(self.sequence.poses.t) - 1)
        pose = self.sequence.poses
        gyro = float(np.linalg.norm(self.sequence.imu_usb[k, 3:6]))
        left = self.renderer.render(pose.R_W_C[k], pose.p_W_C[k], pose.R_W_B[k], pose.p_W_B[k], k, False, gyro)
        right = self.renderer.render(pose.R_W_C[k], pose.p_W_C[k], pose.R_W_B[k], pose.p_W_B[k], k, True, gyro)
        self.left_view.setPixmap(self._pixmap(left, self.left_view))
        self.right_view.setPixmap(self._pixmap(right, self.right_view))
        self.time_label.setText(f"{pose.t[k]:.2f} s")

    def _advance(self) -> None:
        value = self.slider.value() + max(1, int(self.config().imu_hz / self.config().camera_fps))
        if value > self.slider.maximum():
            value = 0
        self.slider.setValue(value)

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("播放")
        else:
            self.timer.start(max(20, int(1000 / self.config().camera_fps)))
            self.play_btn.setText("暂停")

    def export_dataset(self) -> None:
        if self.sequence is None:
            self.generate_preview()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "选择数据集根目录",
            str(Path(__file__).resolve().parents[1] / "datasets" / "synthetic"),
        )
        if not directory:
            return
        self.export_btn.setEnabled(False)
        self.progress.setValue(0)
        self._thread = QtCore.QThread(self)
        self._worker = ExportWorker(self.sequence, Path(directory))
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.finished.connect(self._export_done)
        self._worker.failed.connect(self._export_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    @QtCore.Slot(str)
    def _export_done(self, path: str) -> None:
        self.export_btn.setEnabled(True)
        self.status.setText(f"已导出：{path}")
        QtWidgets.QMessageBox.information(self, "导出完成", path)

    @QtCore.Slot(str)
    def _export_failed(self, error: str) -> None:
        self.export_btn.setEnabled(True)
        self.status.setText("导出失败")
        QtWidgets.QMessageBox.critical(self, "导出失败", error)

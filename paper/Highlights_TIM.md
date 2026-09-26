# Technical highlights for PeerTrack / cover-letter preparation

- IMU-only inference estimates 3-D camera displacement from continuous 200-Hz streams; chessboard poses are used only for calibration, supervision and evaluation.
- PhysNet combines analytic anchor-frame pre-integration, dense supervision, a stillness gate and a learned lever arm with 0.42 M parameters (1/7 of adapted IMUNet).
- PhysNet lowers the synthetic-test error of adapted IMUNet by 24%; on real data all learned models beat zero motion, but PhysNet, IMUNet and their blends do not differ significantly.
- Every estimator is shrunk toward zero (slope 0.2-0.5), consistent with an initial velocity the window cannot determine; a 10-fps reference cannot measure that term's share of the error.
- A simulated known-displacement test bounds the chessboard reference at 1.4-2.5% scale error (depending on the modelled optical blur), at most 3% of the model error.

These bullets are an internal submission aid. IEEE TIM does not request a
separate highlights file in its current author instructions.

# TIM manuscript restructuring notes

## Style references read

- Zimmer et al., "Digital Twins for Building Pseudo-Measurements," IEEE TIM, 2025.
  - Structure: problem, theoretical background, proposed architecture, field case study, short/long-term results, conclusion.
- Evans et al., "Pattern Matters," IEEE TIM, 2025.
  - Explicit contribution bullets, controlled physical-factor ablations, multi-architecture verification, reproducibility statement.
- Liu et al., "DUET," IEEE TIM, 2023.
  - Kinematic formulation before network, two public datasets, baseline table, bias-domain discussion, window-length ablation, conclusion and limitations.
- Zeinali et al., "IMUNet," IEEE TIM, 2024.
  - Dataset protocol, common metrics across datasets, architecture/efficiency comparison, edge-device latency.
- Fu et al., "Hybrid Model-Driven Spectroscopic Network," IEEE TIM, 2023.
  - Physical simulation + experimental data, domain generalization, in-situ validation.

OA/accepted PDFs used for internal style study are stored in `paper/literature/`.

## Structure (revision 2, 2026-09-25)

1. Introduction: four gaps, three research questions (RQ2 now asks for the representation and the information limit), five contributions.
2. Related work by reference measurement, inertial calibration, and measurement digital twins.
3. Measurement framework and information boundaries (fixed split + leave-one-session-out + sealed test; extra-train never scored).
4. Physical instrumentation, timing, calibration, real dataset incl. four rigidity recordings as training-only data.
5. Reference uncertainty methodology.
6. Camera-IMU rigidity and label-leakage tests.
7. Parameter-provenance digital twin, now with measured board-to-gravity orientation and real-trajectory replay.
8. PhysNet: anchor-frame pre-integration, velocity-integration output, dense supervision, learnable lever arm, training protocols, trajectory formation (single-edge and dense-stream graph).
9. Experimental protocol: data inventory, adapted public architectures, block-bootstrap metrics, ablation matrix.
10. Results: reference, twin, multi-dimensional comparison (Table III), leave-one-session-out (Fig. 6), information limit (Table IV, Fig. 7), learned lever arm, trajectory ATE.
11. Discussion by research question, measurement implications (statistical power of one test session, lever-arm self-check), next measurements (capture protocol v2).
12. Conclusion.

## Numbers used in revision 2 and their sources

- Synthetic direct comparison: `datasets/physnet_v1/eval_synonly_181622.json`, `datasets/external_benchmark/test_benchmark.json`, `datasets/external_benchmark/zero_shot_synthetic_ood.json`.
- Fixed-split validation/test: `datasets/physnet_v1/eval_wd1.json`, `datasets/external_benchmark/optimized_ensemble.json`, `datasets/method_benchmark_20260924.json`.
- Leave-one-session-out, calibration, sealed test, ATE: `datasets/session_cv_benchmark.json` (tag `cvx`: 8 folds x 2 seeds, extra-train = 4 rigid sessions from `trajectory_split_20260924.json`).
- Ablations: `datasets/physnet_v1/metrics_*_s{0,1}.json`; information-limit statistics: `paper/figures/information_limit.json` (written by `tools/make_paper_figures.py`).
- Trajectories: `datasets/trajectory_comparison/summary.json` (rebuilt with `--extra-edges datasets/session_cv_benchmark.npz`).

## Length

Revision 2 compiles to 11 pages (previous draft: 8). Candidates for trimming if an 8-page budget is required: horizon figure (text already carries the numbers), framework figure height, domain-gap figure, trajectory figure (one row), and the related-work section.

## Remaining author-side information

- Authors, affiliations, IEEE membership, corresponding author.
- Funding and manuscript dates.
- Author biographies (and photos for the final accepted version).
- Exact camera product/model; mechanical verification of the learned 38-mm lever arm.
- Independent optical-tracker validation and manufacturer tolerance for board pitch/flatness.
- Results of the v2 acquisition round (`docs/collection_protocol.md`): several sealed test sessions will replace the single-session block-bootstrap intervals.

## Round 3 (A/B/C) additions

New sections and numbers, with sources:

- learning.tex: zero-velocity detection and structural update (eq. gate), per-prediction uncertainty. Source: tools/train_physnet.py --still-weight/--gate/--zupt/--nll-weight.
- results.tex: Table V (zero-velocity and uncertainty variants), stillness detector quality, Fig. 8 (ZUPT bound + uncertainty calibration), cross-window fusion negative result, gate local/trajectory trade-off. Sources: datasets/physnet_a/metrics_*.json, datasets/zupt_bound.json, datasets/uncertainty_val.json, datasets/fusion_sweep_val.txt, datasets/session_cv_gate_benchmark.json.
- Table III updated with the gated PhysNet fold ensemble and both blends (cv-tag cvg against baseline-tag cvx).
- Summary of the whole round: datasets/physnet_round3_20260925.json.

Current length: 13 pages. Trim candidates in order: Fig. 4 (horizon, numbers are in the text), Fig. 3 or Fig. 2 (one of the two audit figures), the related-work section, Table I, and one row of Fig. 9.

## Round 4: narrative repositioned to "measurement chain + information limit + acquisition design" (2026-09-25)

Title: "Audited Inertial Camera-Motion Measurement and Its Information Limit".

The spine is now the information in the chain, not the network ranking.

- framework.tex: new Section III-B "What Is Observable Inside One Window" derives the decomposition (eq. decomp) of the measurand into an inertially measured double integral and an unobservable initial velocity, and states the two ways to change it (longer context: statistical; a zero-velocity epoch: exact, limited by attitude). The results sections are forward-referenced from here.
- introduction.tex: four research questions (reference / information limit / which acquisition property moves it / which representation reaches it at what cost) and five contributions led by the limit analysis and the acquisition-design result.
- results.tex reordered: A reference, B twin audit, C observable horizon and the information limit (horizon ablation + 40-variant saturation + decomposition + the negative fusion result), D what a detected pause is worth and the resulting acquisition requirement (detector quality + ZUPT bound + pause statistics + the 4-6/min target), E estimators at the limit (multi-dimensional table framed as cost-to-reach-the-limit), F uncertainty, G lever arm, H trajectory and the operating-point choice.
- experiments.tex: new subsection "Measurements That Do Not Involve the Estimator" so the limit and ZUPT-bound experiments are identified as protocol characterization, not model results.
- discussion.tex: restructured to the four research questions; adds attitude quality as the second lever on the dominant error term.
- abstract.tex, conclusion.tex: lead with the limit and the acquisition parameter; the estimator is presented as reaching the limit at the lowest cost.

Length: 14 pages. Trim order if an 8-page budget is required: Fig. 4 (horizon, numbers in text), one of the two reference/twin audit figures, Section II (related work) compressed to one paragraph per theme, Table I, one row of Fig. 9, and the earlier Conv-BiGRU row of Table III.

## Round 5: authors, related work, and three new/redrawn figures (2026-09-25)

- main.tex: author block filled in (Ranyang Li, Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan). Affiliations, IEEE membership grades, corresponding author, manuscript dates and funding remain as TODO because they are not in the repository.
- related.tex rewritten from 3 to 5 subsections and from about 320 to about 1150 words:
  A Reference measurements for camera motion (robot kinematics, optical trackers, endoscopic datasets, visual-inertial benchmarks and their calibration, planar-target conditioning).
  B Inertial navigation and the observability of translation (strapdown error propagation, Allan variance, pre-integration, then the ZUPT line of work) -- this subsection is the literature anchor of the information-limit spine.
  C Learned inertial odometry and learned calibration (IONet, RIDI, RoNIN, OxIOD, TLIO, IDOL, IMUNet, DUET, gyro denoising, AI-IMU, legged) with an explicit statement of what this literature does not decompose.
  D Uncertainty of a learned measurement (heteroscedastic likelihood, deep ensembles, calibration, block bootstrap).
  E Digital twins and simulation for measurement.
  refs.bib grew from 25 to 45 entries; 46 references are cited.
- Fig. 1 redrawn (tools/make_paper_figures.py::framework): three bands, four colour-coded chains, the decomposition equation as the centre band with braces marking which term each path supplies, per-source audits, and the sealed-test barrier. Caption rewritten accordingly.
- Fig. 4 new (::architecture): PhysNet block diagram. Band 1 input window with a still phase, deterministic anchor-frame pre-integration with its equations, the trunk with layer list, and the three heads; band 2 the structural kinematics (gate, integration with lever arm, uncertainty); band 3 the output displacement stream with the dense supervision points, the anchor and the 3-s target.
- Fig. 2 new (::dataset): (a)-(c) real left-camera frames from three recordings with detected corners, (d) a Blender 2K render, (e) the 200-Hz stream with detected pauses shaded, (f) the reference speed with the two stillness thresholds, (g) a full reference trajectory coloured by speed, (h) the measurand distribution per split. This is the dataset/qualitative display the reviewers would otherwise ask for.
- abstract.tex shortened from about 330 to about 250 words.

Length: 16 pages, 11 figures, 5 tables, 46 references. This is now clearly over an 8-page TIM budget. Trim order: (1) Fig. 6 horizon (numbers are in the text), (2) Fig. 3 domain gap or Fig. 5 reference characterization, (3) Table I, (4) one row of Fig. 11, (5) the earlier Conv-BiGRU row of Table V, (6) compress related work B and E by one paragraph each, (7) merge Fig. 9 into Table V.

## Round 6: trimming for the IEEE TIM page budget (2026-09-25)

Target agreed with the authors: as close to the editorial norm as possible while accepting a modest overlength charge. 17 pages -> 14 pages.

Author block: Ranyang Li (corresponding, lry@haut.edu.cn), Nan Wei, Zhipeng Lin, Wufeng Liu, Chao Fan; affiliations and the six funding grants taken from the authors' BMC Medical Imaging submission. NSFC grant number corrected to 62402163. Biographies stubbed for all five authors. A Reproducibility section was added before the references.

What was removed or converted:
- Figures 11 -> 8. Deleted: horizon ablation, domain gap, and the four-panel method benchmark (all three only restated numbers that are in the text or in Table IV). Converted from double-column to single-column with stacked panels: information limit, ZUPT/uncertainty, trajectory. The trajectory figure now shows the sealed real test only; the synthetic ATE is given in the caption. The dataset figure lost its trajectory panel and is now two rows.
- Tables 5 -> 4. The instrument-parameter table was folded into a sentence in Section IV-A.
- References 46 -> 35. Removed as least load-bearing: IPPE, Ligorio and Sabatini, SPSVO, C3VD, TUM VI, extended Kalibr, OxIOD, IDOL, legged-robot inertial odometry, OpenShoe, VINS-Mono.
- Prose 10178 -> 9419 words. The discussion no longer restates result numbers, the training protocols and the information-boundary list became running text, the ablation matrix in Section IX is now a pointer to the two ablation tables, "Why the representation matters" was merged into the pre-integration subsection, and the uncertainty and lever-arm subsections were merged.

To go below 14 pages, content has to be dropped rather than compressed. Remaining options, in the order I would take them:
1. Move the twin section (Section VII) and its audit paragraph to supplementary material, keeping three sentences in the experimental protocol: about 1 page.
2. Drop the per-prediction uncertainty subsection and Fig. 6(b), keeping one sentence and the coverage numbers: about 0.5 page.
3. Drop Fig. 7 (session CV) and report the eight fold errors as a table row: about 0.2 page.
4. Drop the earlier Conv-BiGRU and RoNIN-LSTM rows from Table IV and the corresponding sentences: about 0.2 page.
5. Shorten Section V (reference uncertainty) to the operational statement and move the Monte Carlo and holdout detail to supplementary material: about 0.6 page.
Items 1, 2 and 5 weaken the instrumentation framing that the current positioning rests on, so 14 pages with an overlength charge looks like the better trade.

## Round 7: everything restored, every figure single-column (2026-09-26)

The authors asked not to delete figures and to place them all in a single column.

- The three figures deleted in round 6 are back: horizon ablation, domain gap, four-panel method benchmark. Table I (instrument parameters) is back as well.
- All eleven figures are now single-column and drawn at 3.45 in native width, so no figure is scaled down by LaTeX and no label loses legibility. Four figures had to be redrawn for the narrow format: reference uncertainty and domain gap as two narrow side-by-side panels, the method benchmark as a 2x2 grid, and the session cross-validation as horizontal bars with the session names on the y axis.
- The three diagrams were redesigned vertically: the framework as five stacked bands (instrument, the two measurement paths, the decomposition, twin and estimator, audits and evaluation), PhysNet as six stacked bands (input, pre-integration, trunk, heads, structural kinematics, output stream), and the corpus figure as four image panels in two rows plus three full-width plots.
- _box() no longer uses an absolute body offset, which was what made the text overflow the smaller boxes.
- Float placement changed from [t] to [tb] for the tall figures; with [t] only, seven figures were deferred to the end and produced almost empty pages (a first attempt at this configuration came out at 18 pages).
- Table V (the multi-dimensional comparison) stays double-column: it has ten columns and cannot be set in one.

Result: 15 pages, 11 figures, 5 tables, 35 references. That is one page more than the round-6 version that had three figures deleted and three wide figures, and three pages more than the 12-page target. The extra length buys back the three figures and the instrument table.

## Round 8: keep everything, condense on-objective prose (2026-09-26)

Nothing was deleted. The five candidates the authors asked to keep are all still in the paper: the digital-twin section, the per-prediction uncertainty subsection with Fig. 6(b), the session cross-validation figure, the early Conv-BiGRU and RoNIN-LSTM rows of Table V, and the reference-uncertainty section.

Condensation instead of deletion:
- Subsection headings 46 -> 42. Section V went from five subsections to three (model and Type-A merged; corner propagation merged with tilt-translation coupling). Section VI went from four to two (the rigidity test and its acceptance criteria merged, the gyroscope-drift paragraph folded into one sentence of the same subsection). Section VII merged the scene/motion domain with the twin audit. Section VIII merged per-prediction uncertainty into the output subsection and protocols with trajectory formation. Each heading costs about two lines in two-column layout.
- Figure area, with no content removed: information limit, ZUPT/uncertainty and trajectory changed from two stacked panels to two side-by-side panels, 3.55-3.86 in tall -> 2.05-2.26 in. The per-axis correlations and slopes moved from the Fig. 5(a) legend into the text, where they already appeared, so the legend is now just the three axis colours.

Result: 14 pages, 11 figures, 5 tables, 35 references, no scaled-down artwork (every figure is drawn at 3.4-3.6 in native width; the framework and PhysNet diagrams are drawn at 2.87 in and are scaled up, not down).

Same as the round-6 count that had three figures deleted, now with all of them present.

## Round 9: merged result displays, no double-column float left (2026-09-26)

The authors asked whether the many experimental results could be merged and shown single-column.

Merged figures (11 -> 9):
- Fig. 6 "observability.pdf" merges the former horizon ablation and the two information-limit panels into one three-panel figure: (a) validation R2 versus horizon, (b) residual against the unobservable term, (c) the RMS of that term versus context length. All three belong to Section X-C, so one caption now serves them.
- Fig. 8 "estimators.pdf" merges the former four-panel method benchmark with the session cross-validation bar chart into one five-panel figure: (a)-(c) synthetic-only models, (d) sealed-test CV ensembles, (e) the per-session view. All five belong to Section X-E.
- tools/make_paper_figures.py: horizon(), information_limit(), benchmark_v2() and session_cv() were replaced by observability() and estimators(); _drift_statistics() now holds the shared computation. The unused PDFs were deleted from paper/figures.

Merged and split tables (5 -> 5, but nothing double-column):
- Table III now carries every PhysNet ablation in two labelled groups: A for capacity, loss, augmentation, pretraining and data, B for the zero-velocity and uncertainty variants with seed counts and spreads. The former Table IV disappeared as a separate float.
- The former double-column Table V was split by evaluation domain into two single-column tables: Table IV for synthetic accuracy, synthetic-to-real transfer, parameter count and latency, and Table V for the real-data columns (validation, cross-validation, sealed test, ATE). This also let the parameter and latency numbers move from running text into a table, and it matches the narrative, whose first paragraph is synthetic and second is real.

The paper now contains no figure* or table*: every float is single-column, and no artwork is scaled down by LaTeX.

Result: 14 pages, 9 figures, 5 tables, 35 references. Float count dropped from 16 to 14 while all content was kept.

## Round 10: Figure 1 rewritten in symbols (2026-09-26)

Figure 1 carried too much prose. It is now symbol-driven and 2.97 in tall instead of 3.40 in, with the same information flow:
- Instrument band: the three raw signals as symbols, {I_L, I_R} / f_I, omega_I / B, omega_B, each with a three-word note.
- Measurement paths: the optical path as T_BC <- PnP(I_L) with its two numeric gates, the inertial path as the two pre-integration equations.
- Decomposition band: unchanged equation, but the two prose annotations became "unobservable / prior or pause" and "measured / +- b, +- s, +- dtheta x g".
- Twin: Sigma, b~RW, Q, dt, g_B and the replay arrow to {f, omega}_sim.
- PhysNet: the gate, the integration with the lever arm, and the two outputs d(t_b), u(d).
- Audits and evaluation: the four quantities each, as symbols (u_A, Sigma_corner, s_p, r_axis) and metrics (e_p, R2, ATE).
- The two-line colour legend was shortened to one line each and centred.

The caption now expands the symbols, so the figure carries notation and the caption carries the words. Page count stays at 14.

## Round 11: final layout/structure pass (2026-09-26)

- Figure 1: reduced to 2.99 in high, replaced almost all prose inside the boxes by symbols/formulas, tightened body line spacing from 1.35 to 1.15, increased every arrow head from mutation scale 9 to 13 and the minimum shaft width to 1.15, and retained 2-3% vertical gaps between boxes. The caption now expands the notation.
- Figure 2: every acquisition domain now contains a four-frame montage (16 image examples in panels a-d instead of one per domain); the remaining panels show the 200-Hz stream, reference speed/stillness labels, and the measurand distribution. Empty image-panel area has been removed.
- Figure 3: the actual PySide6 digital-twin workbench is shown above the acceleration and angular-rate domain audit, so the implementation is visible without adding another float.
- Figure 9: expanded to six real-trajectory views: XY, XZ, YZ, X(t), Z(t), and the position-error CDF. Each includes Reference/Ridge/IMUNet/PhysNet/blend; ATE appears in the first legend.
- Subsections reduced to 25 total: Discussion 3->2, Digital Twin 3->2, Reference Uncertainty 3->2. The existing consolidated structure (Framework 2, System 2, Experiments 2, Learning 3, Results 5) is retained.
- Abstract rewritten into problem -> proposed framework -> proposed PhysNet -> limit analysis -> quantitative results -> conclusion. Contribution bullets rewritten in active voice: "We establish / We derive and measure / We propose / We build and validate".
- Review manuscript omits author biographies and photographs; sections/biographies.tex remains for the accepted final version. Affiliations, corresponding author and funding remain on page 1.
- Review output: paper/main_review.pdf and paper/main.pdf, 14 pages, no compilation error, undefined citation or overfull box.

## Round 12: submission-level audit (2026-09-26)

Figures/layout:
- Fig. 1 arrows: default arrow-head mutation scale 9 -> 13; minimum shaft width 1.15; box-body line spacing 1.35 -> 1.15; symbols replace prose.
- Fig. 2: four-frame montage per acquisition domain (16 image examples total); panel/caption mismatch fixed.
- Fig. 3: full PySide6 workbench screenshot regenerated after loading Microsoft YaHei into the off-screen Qt process, so the left parameter panel is visible and Chinese labels render correctly; the GUI is merged with the two domain-audit panels.
- Fig. 9: expanded to six sealed-real views (XY/XZ/YZ, X(t), Z(t), position-error CDF) for Reference, Ridge, IMUNet, PhysNet and blend.

Structure/prose:
- Subsections reduced to 25: Discussion 3->2, Digital Twin 3->2, Reference Uncertainty 3->2 (plus earlier consolidations).
- Abstract rewritten as problem/method/analysis/results/conclusion. Contributions use active verbs: We establish, derive and measure, propose, build and validate.
- Discussion reduced and separated into reference / information-and-acquisition / estimator-and-twin findings; Conclusion reduced to two result-centred paragraphs.
- All red TODOs removed from the review manuscript. Author biographies remain outside the review build.

Mathematical corrections:
- Frame convention R_XY defined; R_CI direction made explicit; the learned output explicitly limited to 3-D translation rather than full learned SE(3) pose.
- World-frame convention and Log^vee operator defined in the twin; quantizers/noise/random-walk variables defined.
- Window-mean subtraction corrected from an unbiased "gravity estimate" to gravity-plus-mean-acceleration removal.
- The velocity integral made a signed discrete operator for samples before and after the anchor; lever-arm direction defined; endpoint double weighting in the dense loss disclosed.
- Log-variance renamed eta to avoid collision with dynamic acceleration ell; units and calibrated total uncertainty defined.
- Pose-graph row scaling corrected in the paper from w||r||^2 to alpha^2||r||^2, matching the implementation.
- R2, fastest-third definition and unaligned anchor-only ATE defined.
- The information-limit claim corrected from "40% of the residual" to 16% of mean error / about 30% of squared error.
- Tilt-sensitivity units defined as mm/deg and px/deg.

Verification:
- tools/verify_manuscript.py added. It asserts every headline number in Tables IV/V, information-limit statistics, uncertainty coverage, confidence intervals and trajectory ATE against the stored JSON artefacts and checks every LaTeX label/citation. Current result: PASS.
- refs.bib contains exactly the 33 cited works: no missing or unused entry. Crossref resolves 29 DOI entries; four proceedings/arXiv entries without DOI were checked by venue/arXiv metadata. datasets/reference_audit.json records all 33 as verified.
- Final build: paper/main_submission.pdf and paper/main.pdf, 14 pages; no compilation error, undefined citation or overfull box.

## Round 13: submission audit after independent technical reviews (2026-09-26)

Figure/layout:
- Fig. 2 top image area rebuilt with nested grids; each of the four domains now shows a larger four-frame montage while the lower plots retain independent spacing.
- Fig. 3 now uses paper/figures/sim_gui_full.png, captured after loading Microsoft YaHei into the off-screen Qt process. The complete left parameter panel, tabs, stereo views, 3-D path and IMU plot are visible. It is merged with the acceleration/angular-rate domain audit.
- Fig. 9 expanded to six real-trajectory views: XY, XZ, YZ, X(t), Z(t), position-error CDF for Reference, Ridge, adapted IMUNet, non-gated PhysNet and their non-gated blend.

Mathematical/claim corrections from independent audit:
- The context-mean velocity is now called an empirical low-frequency proxy, not the initial velocity itself; longer context is no longer claimed to make v(t_a) observable. "Information limit" was downgraded throughout to a protocol-dependent observability floor / empirical error floor.
- The stillness mechanism is called a soft stillness-conditioned velocity gate, not a true ZUPT state reset. A stationary epoch supplies a boundary condition subject to detection/attitude/bias errors.
- The stationary-boundary experiment now reports its actual sample support: n=1 below 2 s (0.7/1.2 mm) and pooled 2-4 s errors of 53.9/104.1 mm. The long-span divergence is attributed to time-varying attitude and nonconstant sensor errors; a constant tilt component can be absorbed by the fitted bias. "Accelerometer only" was removed because an attitude trajectory is supplied.
- The empirical proxy reduces mean error by 16%, equivalent to about 30% in squared error; the earlier unsupported "40% of residual" wording was removed.
- Fixed frame conventions, the undefined O frame, Log^vee, gravity/noise/quantizer definitions, signed integration around the anchor, log-variance symbol collision (eta vs ell), uncertainty units, R2/ATE definitions, tilt units, pose-graph row-weight convention, and interpolation wording.
- Explicitly states that the learned output is 3-D translation/displacement, not full learned SE(3), and that reference orientation is used only for translation-isolated trajectory evaluation.

Result/protocol corrections:
- Fig. 8 now reads session_cv_gate_benchmark.json and matches the gated numbers; gate PhysNet wins 6/8 folds, not 5/8.
- Table V separates equal gate blend (42.21 mm, R2=.276) from calibrated gate blend (42.29 mm, R2=.279, fast=72.10). No row combines metrics from different outputs.
- Gated PhysNet vs zero CI corrected to [2.16, 10.80] mm; large-motion R2 range corrected to .17-.62; PhysNet latency is called lower than IMUNet, not globally lowest (RoNIN-LSTM is .19 ms but collapses).
- Fig. 9/caption identifies its PhysNet/blend trajectories as the non-gated ATE-selected operating point.
- Group A/B of Table III explicitly disclose different training pools; six unfiltered rigidity recordings vs four verified recordings are not presented as a pure architecture ablation.
- The historical test recording is now called held-out, not sealed. The paper discloses that it was excluded from fitting/initial selection but reused for post-selection exploratory diagnostics. Future recordings are called prospectively sealed.
- tools/verify_manuscript.py now checks equal/calibrated blend separation, both confidence intervals, 6/8 fold wins and pooled stationary-boundary values. PASS.

References:
- Current refs.bib contains exactly the 31 cited records, no unused entry. All 31 verified: 29 DOI records resolve in Crossref; Kendall/Gal and Lakshminarayanan et al. verified by NeurIPS volume/pages.
- Removed withdrawn MITI arXiv citation; corrected RoNIN author order, full IMUNet title, Zimmer pages, Titterton publisher; added DOI/volume/pages to the verifiable records. Guo classification-calibration paper removed from the regression-uncertainty claim.
- datasets/reference_audit.json regenerated; all 31 pass.

Final outputs: paper/main.pdf and paper/main_submission.pdf, 14 pages. tools/verify_manuscript.py PASS; LaTeX has no error, undefined citation or overfull box.

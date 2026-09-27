# IEEE TIM submission checklist — EGO_Mo

Status values: `[x]` complete, `[ ]` blocked or requires author action.

## Scientific evidence

- [x] Estimator comparison uses the existing real recordings only. Further
  handheld collection with a continuously visible board is not part of this
  submission.
- [x] Acquisition (stop rate, label rate, ideal IMU) is studied in a twin
  matched to the measured motion and sensor domain
  (`datasets/twin_realistic_summary.json`). The twin does not replace the
  real comparison.
- [x] Disclose that the historical test is held out but not prospectively sealed.
- [x] Evaluate initially anchored pure-IMU relative 6-DoF pose.
- [ ] Compare the chessboard reference with a physical displacement standard
  before any reference-grade claim.
- [ ] Enter at least three independent mechanical lever-arm measurements.
  The learned lever arm is not a calibration.

## Manuscript integrity

- [x] IEEE two-column regular-paper format; figures and tables inline.
- [x] PDF is self-contained, unencrypted, and below 20 MB.
- [x] Author biographies/photos omitted from the review manuscript.
- [x] No undefined citation or label, and no red placeholder. One 1.6-pt overfull in the ablation table was removed by tightening column spacing.
- [x] All 35 cited references are in `refs.bib` with no unused entry. DOIs were checked against the published records; four proceedings entries without a DOI (NeurIPS, MIDL) were checked by venue, volume and pages.
- [x] `tools/verify_manuscript.py` passes.
- [x] Scope explicitly states translation/displacement, not learned absolute SE(3).
- [x] Disclose the held-out recording as exploratory. No further real cohort
  is pending for this submission; stop-rate results are from the matched twin.

## PeerTrack metadata

- [ ] ORCID for Ranyang Li.
- [ ] ORCID for Nan Wei.
- [ ] ORCID for Zhipeng Lin.
- [ ] ORCID for Wufeng Liu.
- [ ] ORCID for Chao Fan.
- [ ] Confirm every author's institutional e-mail and affiliation.
- [x] Corresponding author: Ranyang Li (`lry@haut.edu.cn`).
- [x] Funding numbers included on page 1.
- [ ] Choose traditional publication or optional open access.
- [ ] Enter TIM classifications/keywords in PeerTrack.
- [ ] Obtain approval from all co-authors for the final PDF.

## Page charges (2026 policy)

- [x] Regular-paper free limit: 8 published pages.
- [x] Current review PDF: 15 pages (the production length can change).
- [ ] Identify the author/institution with financial authority.
- [ ] Accept/upload the mandatory overlength agreement in PeerTrack.
- [ ] Budget, before tax:
  - non-IMS rate: 6 × US$265 = **US$1,590**;
  - IMS-member rate: 6 × US$220 = **US$1,320**.
- [ ] If choosing open access, separately budget the current APC shown by TIM
  (US$2,800 for papers submitted in 2026, plus tax); verify again on submission day.

## Files

- [x] Main review PDF: `paper/main_submission.pdf`.
- [x] LaTeX source and embedded figures.
- [x] Optional cover-letter template.
- [x] Software/result release ZIP with SHA-256 manifest.
- [x] Repository URL: https://github.com/liranyang123456-commits/EGO_Mo (public).
- [ ] Zenodo DOI.
- [ ] Separate raw-data archive and DOI, or a justified controlled-access statement.
- [x] `CITATION.cff` repository URL set; DOI pending.
- [ ] Update the cover letter’s repository/DOI placeholder.

## Current TIM instructions checked

- One complete PDF with inline figures/tables; do not upload figures separately.
- Regular paper minimum 5 pages; no submission maximum.
- Overlength charges begin after 8 published pages.
- Cover letter is optional and should be used only for exceptional disclosure.
- Graphical abstract is optional and must be peer reviewed if supplied.
- Author biographies are not required in the review manuscript.

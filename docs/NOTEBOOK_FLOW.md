# Assignment Notebook Flow

`notebooks/WiFi_CSI_HPE_A2_Independent.ipynb` is the public, self-contained narrative. It calls
shared modules and reads saved manifests/results; it does not reproduce a second,
divergent implementation.

| Section | Evidence | Source of truth |
|---|---|---|
| 0. Reproducibility | environment, seed, paths, data versions | config + manifests |
| 1.1 Wi-Pose | counts, action distribution, CSI, 18 AlphaPose sequence | inventory/audit |
| 1.2 WiMANS | domain/band/person/action distribution and label limits | inventory/audit |
| 2. Preprocessing | raw/clean CSI, named mapping, canonicalization, split | processed report |
| 3. Model | tensor interfaces, architecture, loss, metrics | shared modules |
| 4. Condition A | configuration, epoch curves, final metrics | source-only results |
| 5. Condition B | SSL/fine-tune configuration and results | SSL results |
| 6. Evaluation | predictions, error analysis, target-domain caveats | frozen outputs |
| 7. Reflection | limitations, decisions, implementation record | project docs |

Every figure title identifies dataset, split, condition, sample/recording ID, time
range, label provenance, and preprocessing version where applicable. AlphaPose labels
and CSI-model predictions cannot share an ambiguous “ground truth” label.

## Section 6 execution

Section 6 starts from a fresh kernel: it validates and reloads both selected
`best.pt` files, constructs only the frozen evaluation inputs, and reads/writes
versioned evidence under `results/evaluation`.  Native Wi-Pose labels support the
primary pose-accuracy claims.  The 594 held-out WiMANS `empty_room` recordings
contain one participant—the name identifies the room configuration—and AlphaPose
video output is reported only as a **video-derived pseudo-reference**.

The section records coverage before agreement metrics, uses identical frames for
both conditions, reports confidence and ±2-frame alignment sensitivity, and shows
separate Condition A and Condition B performance subsections. Static examples and
a full-recording HTML5 animation use the same corrected upright display axes. It never
trains, resumes, selects, or overwrites a model.

The executed evidence contains 17,645 Wi-Pose test samples and 52,684 valid
AlphaPose pseudo-reference frames from all 594 WiMANS recordings. AlphaPose image
coordinates are rotated into the Wi-Pose stored-camera frame before metrics are
calculated. Condition B improves NME on Wi-Pose (`0.202268` versus `0.206182`) and
WiMANS pseudo-reference agreement (`0.336049` versus `0.353495`), while the notebook
retains the pseudo-reference and synchronization limitations.

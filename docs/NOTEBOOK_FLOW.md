# Assignment Notebook Flow

`notebooks/WiFi_CSI_HPE_A2.ipynb` is the public, self-contained narrative. It calls
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

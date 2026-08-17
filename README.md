```bash
cd /home/zealot/ResarchWork/PackDT/packetDT
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh statedt configs/datasets/cic_ids_2017.yaml
```

StateDT implements decision-sufficient state synthesis for a single, unchanged
decision tree. It supports the explicitly registered monotonic `max`,
non-negative `counter`, and non-negative `sum` feature semantics. A run writes
`compiler.json`, `state_report.json`, and a readable `state_report.txt` beside
the normal `model.pkl` and `metrics.csv` artifacts.

StateDT's former `feature_selection.scaling_aware`, `validation_folds`, and
`max_f1_drop` configuration keys are accepted for compatibility but deprecated
and unused. StateDT does not perform epsilon/F1-loss or capacity-aware model
selection; exact predicate, prediction, and decision-path agreement is required.

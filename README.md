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

StateDT does not perform epsilon/F1-loss or capacity-aware model selection;
exact predicate, prediction, and decision-path agreement is required.

StateDT flow state uses the compiler-emitted `state_layout`: synthesized
feature fields, a 16-bit fingerprint, one initiator-direction bit, and one
valid bit. A two-choice miss returns an explicit collision-fallback status; it
never classifies with an occupied entry belonging to another fingerprint. All
P4 targets expose allocation, fingerprint-mismatch, collision, and fallback
counters.

## Research status

StateDT is an active research prototype. The semantic compiler and generated
state layout are shared by the BMv2, Tofino, and Xilinx implementations. Do not
interpret logical state estimates as measured hardware savings until the
applicable target compiler report confirms the allocation.

The DSN research plan and claim ledger are in `docs/dsn_readiness.md`. The
working presentation is in `slides/dsn_statedt.md`.

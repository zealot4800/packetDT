# SpliDT Minimal

Minimal runner for SpliDT/CAP and the baseline models used in this artifact:
LEO, NetBeacon, and IIsy.

## Prerequisites

- Linux or macOS shell environment
- Python 3.10 recommended
- Preprocessed dataset files under `dataset/`
- Local HyperMapper checkout available at `hypermapper-src/`

This checkout expects these two paths to exist:

```bash
dataset -> ../SpliDT-Artifact-NSDI26/dataset
hypermapper-src -> ../SpliDT-Artifact-NSDI26/hypermapper
```

If those symlinks are missing, create them from the repository root:

```bash
ln -s ../SpliDT-Artifact-NSDI26/dataset dataset
ln -s ../SpliDT-Artifact-NSDI26/hypermapper hypermapper-src
```

The config files point to preprocessed pickle files such as:

```text
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p1.pkl
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p2.pkl
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p3.pkl
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p4.pkl
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p5.pkl
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p6.pkl
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p7.pkl
dataset/CIC-IDS-2017-PCAPS1-f10/dataset_df_p1024.pkl
```

IIsy also needs `dataset_df_p0.pkl`. If that file is not present, `run.sh iisy` and
`run.sh all` skip IIsy automatically.

## Setup

From the repository root:

```bash
cd /home/zealot/SplitDT/SpliDT-Minimal
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.10` is not available but your default `python3` is Python 3.10, use:

```bash
python3 -m venv .venv
```

## Run

The recommended entry point is `run.sh`.

`dataset/`, `hypermapper-src/`, `logs/`, `results/`, and `models/` are local
or generated paths and are ignored by Git. Do not commit those directories.

Show all supported commands:

```bash
./run.sh --help
```

Run the default SpliDT/CAP experiment:

```bash
source .venv/bin/activate
./run.sh splidt
```

By default this uses:

```text
configs/cic-ids-2017-c10-bo.yml
```

Run SpliDT/CAP with a specific config:

```bash
./run.sh splidt configs/cic-ids-2018-c10-bo.yml
```

Run a baseline model:

```bash
./run.sh leo configs/cic-ids-2017-c10-bo.yml
./run.sh netbeacon configs/cic-ids-2017-c10-bo.yml
./run.sh iisy configs/cic-ids-2017-c10-bo.yml
```

Run all models for one config:

```bash
./run.sh all-models configs/cic-ids-2017-c10-bo.yml
```

Run SpliDT/CAP for every local config:

```bash
./run.sh all-datasets
```

Run SpliDT/CAP, LEO, NetBeacon, and IIsy for every local config:

```bash
./run.sh all
```

Available configs:

```text
configs/cic-ids-2017-c10-bo.yml
configs/cic-ids-2018-c10-bo.yml
configs/cic-iot-2023-c4-bo.yml
configs/cic-iot-2023-c32-bo.yml
configs/cic-iomt-2024-c19-bo.yml
```

## Direct Python Commands

You can also run the scripts directly:

```bash
python src/train.py --config configs/cic-ids-2017-c10-bo.yml
python src/leo.py --config configs/cic-ids-2017-c10-bo.yml
python src/netbeacon.py --config configs/cic-ids-2017-c10-bo.yml
python src/iisy.py --config configs/cic-ids-2017-c10-bo.yml
```

`src/train.py` follows the selected `operational_mode` in the config file. The
provided configs currently enable HyperMapper mode:

```yaml
operational_mode:
  single_run: False
  bruteforce: False
  hypermapper: True
```

To run one fixed SpliDT configuration, edit the config so exactly one mode is
enabled, for example:

```yaml
operational_mode:
  single_run: True
  bruteforce: False
  hypermapper: False
```

## Outputs

Runs write logs here:

```text
logs/
```

Experiment outputs are written here:

```text
results/
```

Saved best model artifacts are written here:

```text
models/<dataset>/<model>/<selector>/
```

For example:

```text
models/CIC-IDS-2017-PCAPS1-f10/splidt/best_by_f1/
models/CIC-IDS-2017-PCAPS1-f10/leo/best_by_flows/
```

## Troubleshooting

If installation fails with `./hypermapper-src` not found, check the symlink:

```bash
ls -la hypermapper-src
find -L hypermapper-src -maxdepth 2 -type f | head
```

If a run fails with a missing dataset pickle, check the dataset path in the config:

```yaml
dataset:
  path: "dataset"
  name: "CIC-IDS-2017-PCAPS1-f10"
  destination: "."
```

Then verify the expected files:

```bash
find -L dataset/CIC-IDS-2017-PCAPS1-f10 -maxdepth 1 -name 'dataset_df_p*.pkl' | sort
```

If you want to use a different Python executable with `run.sh`, set `PYTHON`:

```bash
PYTHON=.venv/bin/python ./run.sh splidt configs/cic-ids-2017-c10-bo.yml
```

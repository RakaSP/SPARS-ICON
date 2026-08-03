# SPARS Paper Experiments

This repository contains the experiment configuration and visualization code used for the paper.

## Setup

For simulator installation, dependencies, workload formats, platform formats, and general usage, please refer to the **SPARS-PUB** simulator repository:

`https://github.com/RakaSP/SPARS-Pub`

## Run the Paper Experiments

The complete experiment configuration used for the paper is already defined in:

```text
RunSPARSOuter.py
```

Run it with:

```bash
python RunAll.py
```

An optional number of worker cores can be provided:

```bash
python RunSPARSOuter.py 24
```

## Generate the Paper Figures

After the experiments finish, open and run:

```text
PaperVis.ipynb
```

The notebook reads the generated results and creates the figures used in the paper.

## License

MIT License.

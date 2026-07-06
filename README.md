# JianduReg

This repository provides a minimal reproducibility package for the paper:

**A Fracture Surface Guided Multistage Registration Framework for Bamboo Slip Point Clouds**

The package includes one testing script, one real scanned Jiandu fracture-surface pair, and one synthetic bamboo-slip fracture-surface pair. It is intended to help reviewers and readers inspect the input data format, run the registration pipeline, and reproduce the basic evaluation procedure.

## Repository Structure


JianduReg/
├── README.md
├── requirements.txt
├── test.py
├── data/
│   ├── real_pair/
│   │   ├── fracture_A.ply
│   │   ├── fracture_B.ply
│   │   └── gt.json
│   └── synthetic_pair/
│       ├── fracture_A.ply
│       ├── fracture_B.ply
│       └── gt_transform.json
└── results/

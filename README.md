# Deep Time Series Anomaly Detection Based on Abnormal Priors
This repository provides the implementation of the _CutAddPaste: Time Series Anomaly Detection Utilizing Abnormal Knowledge_ method, called _CutAddPaste_ bellow. 

The implementation uses the Merlion libraries.

## Installation
This code is based on `Python 3.8`, all requires are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```
This repository already includes the merlion's data loading package `ts_datasets`.
Download the data and unzip it in `data/iops_competition` and `data/ucr respectively`.
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.

## Repository Structure

### `conf`
This directory contains experiment parameters for all models on IOpsCompetition, UCR datasets.

### `models`
Source code of OCSVM, DeepSVDD, COCA, AOC, NCAD models.

### `result`
Directory where the experiment result is saved.

## Usage
```
# CutAddPaste Method
python cutAddPaste.py --selected_dataset UCR --device cuda --seed 2
python cutAddPaste.py --selected_dataset IOpsCompetition --device cuda --seed 2

# CutAddPaste Variants
python cutAddPaste_AddAug.py --selected_dataset UCR --device cuda --seed 2
python cutAddPaste_CutAug.py --selected_dataset UCR --device cuda --seed 2
python cutAddPaste_Diff.py --selected_dataset UCR --device cuda --seed 2
python cutAddPaste_PointAug.py --selected_dataset UCR --device cuda --seed 2
python cutAddPaste_Same.py --selected_dataset UCR --device cuda --seed 2


# Baseline training
# model_name: IsolationForest, RandomCutForest, SpectralResidual, LSTMED, DAGMM, OCSVM, DeepSVDD
python baseline.py --dataset UCR --model <model_name>  --debug

# ITDBERT (THNS24)
# Place THNS24_2010.csv and THNS24_2011.csv under data/itdbert/
python baselines.py --selected_dataset THNS24 --selected_model ITDBERT --device cuda --seed 6
# COCA 
python coca.py --selected_dataset UCR --device cuda --seed 2
python coca.py --selected_dataset IOpsCompetition --device cuda --seed 2
# NCAD 
python ncad.py --selected_dataset UCR --device cuda --seed 2
python ncad.py --selected_dataset IOpsCompetition --device cuda --seed 2
# AOC 
python aoc.py --selected_dataset UCR --device cuda --seed 2
python aoc.py --selected_dataset IOpsCompetition --device cuda --seed 2
# TS_AD training
python ts_ad_main.py --training_mode self_supervised --selected_dataset IOpsCompetition --device cuda --seed 5
python ts_ad_main.py --training_mode anomaly_detection --selected_dataset IOpsCompetition --device cuda --seed 5

```


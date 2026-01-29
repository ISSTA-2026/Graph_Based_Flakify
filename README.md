# Graph-Based Flakify

**Graph-Based Flakify** is a multimodal framework for flaky test detection that overcomes the limitations of traditional sequence-based approaches. By synthesizing **semantic representations** from Pre-trained Language Models with **structural representations** derived from Code Property Graph (CPG), our approach effectively captures the complex, non-local dependencies inherent in non-deterministic tests.

Unlike text-only baselines, Graph-Based Flakify utilizes a **Multi-Head Cross-Attention** mechanism to dynamically align semantic tokens with graph topology. This synergy significantly enhances detection accuracy. Experimental results on the FlakeFlagger dataset demonstrate the efficacy of our approach: the model achieves an F1-score of 77.2% in 10-fold cross-validation and 30.5% in per-project prediction, outperforming state-of-the-art baselines by 7.2% and 9.1%, respectively.

The proposed approach combines:
- **Sequence-based semantic models** (CodeBERT, GraphCodeBERT, UniXcoder)
- **Graph-based structural models** (RGCN)
- **Multimodal fusion models** with concatenation or cross-attention

---

## Directory Structure

```
Graph-Based-Flakify/
├── results/
│   ├── cross-validation/       # Results for 10-fold CV experiments
│   └── per-project-prediction/ # Results for per-project prediction tasks
├── run_files/                  
│   ├── models/                 # Model architecture definitions
│   └── run_*.py                # Execution scripts for training & evaluation
├── zips/                       # Compressed raw datasets (*.zip)
├── node_embedding.py           # Script for node embedding
├── parse.py                    # Script for parsing source code into CPG
├── export_to_dot.py            # Script for exporting CPG
└── extract_label_code.py       # Script for preparing CPG nodes token
```

---

## Environment

The following versions were used in our experiments:

- CUDA Version: 12.6
- Python: 3.8.10  
- imbalanced_learn: 0.12.4  
- numpy: 1.24.4  
- pandas: 2.0.3  
- transformers: 4.35.0  
- torch: 2.4.0  
- scikit_learn: 1.3.2  
- dgl: 2.4.0+cu121
- Joern: 4.0.314

---

## Dataset Preparation
To reproduce the experiments, please follow the steps below to prepare the dataset.

### 1. Clone Repository and Retrieve Data
First, clone this repository. Note that large dataset files are managed via Git LFS, so please ensure you install LFS and pull the actual data.

```bash
git clone https://github.com/ISSTA-2026/Graph_Based_Flakify.git
cd Graph_Based_Flakify
git lfs install
git lfs pull
```

---

### 2. Extract Pre-processed Data

Extract all dataset archives located in the `zips/` directory to the project root.

```bash
unzip './zips/*.zip' -d .
```

Upon successful extraction, the following seven directories will be created in the root directory:

* `dataset/`: Contains the base CSV datasets including source code and labels.

* `cpg_bins_FlakeFlagger/`, `cpg_bins_IDoFT/`: The results of parsing Java source code into binary CPG format using Joern.

* `dot_outputs_FlakeFlagger/`, `dot_outputs_IDoFT/`: The CPG exported in DOT format from the binary files.

* `node_token_outputs_FlakeFlagger/`, `node_token_outputs_IDoFT/`: Extracted node labels and code tokens required for graph construction.

---

### 3. Generate Node Embeddings
Run the embedding script to initialize the CPG nodes using LLM features.
Specify the target dataset name (`FlakeFlagger` or `IDoFT`) as an argument.

```bash
python node_embedding.py <dataset>
```

This process generates the final graph dataset with initialized features, which will be saved in the `saved_graphs_<dataset>` directory (e.g., `saved_graphs_FlakeFlagger`).

### (Optional) Construct CPG Data from Scratch

If you wish to regenerate the intermediate CPG data (the contents of the `zips`) from scratch, please execute the following scripts in order. Replace `<dataset>` with either `FlakeFlagger` or `IDoFT`.

Note: These scripts require Joern. Please download and install it from the official Joern website before execution.

1. Parse Source Code: Parses raw Java files and generates the `cpg_bins` directory.
```bash
python parse.py <dataset>
```

2. Export to DOT: Exports the binary CPGs to DOT format, generating the `dot_outputs` directory.
```bash
python export_to_dot.py <dataset>
```

3. Extract Tokens: Extracts node information and generates the `node_token_outputs` directory.
```bash
python extract_label_code.py <dataset>
```

---

## Execution
The framework supports two evaluation protocols: 10-fold Cross-Validation and Per-Project Prediction.
All execution scripts are located in the `run_files/` directory. The model definitions required for each execution script are stored in the `run_files/models/` directory.

**1. Cross-Validation Experiments**

Perform 10-fold cross-validation on the specified dataset.

**Arguments:**
* `<dataset>`: Choose between `FlakeFlagger` or `IDoFT`.

* `<model>`: Choose the model architecture (`CodeBERT`, `GraphCodeBERT`, `UniXcoder`).

* `<fusion>`: Choose the fusion strategy (`only-concat` or `cross-attention`).

**Fine-Tuning (Sequence Baselines)**

* CodeBERT / UniXcoder:

Since these two models share the same execution script, please specify either `CodeBERT` or `UniXcoder` for the `<model>` argument.
```bash
python run_files/run_FineTuning_CodeBERT_UniXcoder.py <dataset> <model>
# Example: python run_files/run_FineTuning_CodeBERT_UniXcoder.py FlakeFlagger CodeBERT
```

* GraphCodeBERT:
```bash
python run_files/run_FineTuning_GraphCodeBERT.py <dataset>
```

**R-GCN (Structural Baseline)**

* R-GCN:

Specify the LLM used for node initialization.
```bash
python run_files/run_RGCN.py <dataset> <model>
# Example: python run_files/run_RGCN.py FlakeFlagger UniXcoder
```

**Proposed Method (Multimodal)**

* Graph-Based Flakify:

Specify the LLM and the fusion strategy.
```bash
python run_files/run_multimodal.py <dataset> <model> <fusion>
# Example: python run_files/run_multimodal.py FlakeFlagger UniXcoder cross-attention
```

Upon execution, the results will be saved in the `results/` directory. The best model checkpoint for each fold is saved as a `.pt` file, and the performance metrics are logged in a `.csv` file.

**2. Per-Project Prediction**

Evaluate the model's generalization capability on unseen projects using the FlakeFlagger dataset. Note that the dataset argument is omitted as it is fixed to FlakeFlagger for this setting.

**Fine-Tuning (Sequence Baselines)**

* CodeBERT / UniXcoder:
```bash
python run_files/run_FineTuning_CodeBERT_UniXcoder_per.py <model>
```

* GraphCodeBERT:
```bash
python run_files/run_FineTuning_GraphCodeBERT_per.py
```

**R-GCN (Structural Baseline)**

* R-GCN:
```bash
python run_files/run_RGCN_per.py <model>
```


**Proposed Method (Multimodal)**

* Graph-Based Flakify:
```bash
python run_files/run_multimodal_per.py <model> <fusion>
```

Similar to cross-validation, the best models and result CSVs for each project will be saved in the `results/` directory.

---
# ITU BDS MLOPS'25 - Project

This repository contains our implementation of a fully reproducible end-to-end MLOps pipeline developed for the ITU BDS MLOps 2025 course.
The goal was to transform the original exploratory notebook into a modular, automated ML system using **DVC** for data management, **MLflow** for experiment tracking, and **GitHub Actions + Dagger** for CI/CD automation.
  
  
## Project Structure
```bash
Happy-days-forked/  
│
├──.github/workflows/pipeline.yml  # GitHub Actions workflow for automated pipeline execution
│
├── artifacts/                     # Model artifacts, metrics & outputs  
│   ├── model_selection.json       # Summary of the best model  
│   ├── lr_metrics.json            # Metrics from the Linear Regression run  
│   ├── xgboost_metrics.json       # Metrics from the XGBoost run  
│   ├── columns_list.json          # Column names used for training  
│   └── date_limits.json           # Date filtering boundaries  
│  
├── notebooks/                     # Original notebooks provided for the project  
│   ├── main.ipynb  
│   ├── model_inference.py  
│   └── requirements.txt  
│  
├── src/  
│   ├── data/                      # Data loading, cleaning, preprocessing  
│   ├── models/                    # Training scripts for LR and XGBoost  
│   └── pipeline/                  # Pipeline orchestration & model selection  
│  
├── .gitignore  
├── README.md  
└── requirements.txt               # Project dependencies needed to run the pipeline  
  ```
  
## Pipeline Overview
### The pipeline performs the following steps:
- Load & preprocess data (cleaning, feature generation, train–test split)
- Generate features + train/test split
- Train two models: **Linear Regression** and **XGBoost**
- Log every experiment to **MLflow** (params, metrics, artifacts)
- Select the best model based on evaluation metrics
- Export artifacts to the artifacts/ folder
    

### How to Run the Pipeline Locally (Manual Execution):
1. Clone the repository
  ```bash
  git clone https://github.com/aniela2906/Happy-days-forked.git
  cd Happy-days-forked
  ```
2. (Recommended) Create a virtual environment
   -> once in the repo folder :  
   *(WINDOWS):* 
  ```bash
  python -m venv clean_env
  clean_env\Scripts\activate
  ```
3. Install dependencies
  ```bash
  pip install -r requirements.txt
  ```
4. Run the pipeline 
  ```bash
  python -m src.pipeline.train
  ```
  This command runs:
- preprocessing  
- feature extraction  
- **LR** training  
- **XGBoost** training  
- model selection  
- **MLflow** logging  
- artifact export  

5. When the run finishes, you will find updated artifacts in:
 ```bash
  artifacts/
│
├── model_selection.json   # Best model + F1 score + MLflow Run ID
├── lr_metrics.json
├── xgboost_metrics.json
├── columns_list.json
└── date_limits.json
 ```

Additionally, all experiments will be tracked in:
  ```bash
  mlruns/
  ```
You can explore results with **mlflow ui**. 

## Continuous Integration (CI) with GitHub Actions
To ensure full reproducibility and automation, the project includes a GitHub Actions workflow located at:
  ```bash
  .github/workflows/pipeline.yml
  ```
On every push to the **main** branch, GitHub Actions **automatically**:  
1. Checks out the repository  
2. Sets up a clean Python environment  
3. Installs all dependencies from *requirements.txt*  
4. Runs the full ML pipeline *(python -m src.pipeline.train)*  
5. Collects results from the run, including:  
- MLflow experiment data  
- model artifacts (metrics, trained models, JSON summaries)
6. Uploads artifacts as downloadable files in GitHub Actions UI  
  
This guarantees that the pipeline executes identically on any machine, is fully reproducible, and requires no manual intervention. The instructor can download all results directly from GitHub Actions.
  
### Files produced by CI *(downloadable artifacts)*
After each workflow run, GitHub Actions exposes two ZIP packages: 
```bash
model-artifacts.zip       # all model outputs and metrics
mlflow-runs.zip           # full MLflow tracking directory
```
    
These files can be downloaded from the **Actions → Run → Artifacts section**.


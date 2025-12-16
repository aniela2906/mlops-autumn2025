# ITU BDS MLOPS'25 - Project Happy Days

This repository contains our implementation of a fully reproducible end-to-end MLOps pipeline developed for the ITU BDS MLOps 2025 course. The goal was to transform the original exploratory notebook into a modular, automated ML system using **DVC** for data management, **MLflow** for experiment tracking, and **GitHub Actions + Dagger** for CI/CD automation.


### Repository Structure 

```text
Happy-days-forked/
│
├──.dvc                         # Internal DVC directory storing configuration and metadata for data versioning.
│ 
├── .github/
│   └── workflows/
│       └── ci_pipeline.yml     # CI/CD Orchestration: Executes Dagger, then uploads the generated artifacts.
│
├── data/
│   └── raw_data.csv.dvc        # DVC Pointer
│
├── src/                        # Core Python Pipeline
│   ├── data_prep.py            # Phase 1: Executes 'dvc pull' and performs data cleaning.
│   ├── train.py                # Phase 2: Training, MLflow run logging, and local model saving.
│   └── deploy.py               # Phase 3: Model selection, registration to the 'mlruns' directory.
│
├── .dvcignore                  
│
├──go.mod                       # Go file that defines the module and required dependencies
│
├──go.sum                       # Go file that ensures continuity and integrity of dependencies.
│
├── main.go                     # Orchestrator (Dagger): Defines the pipeline steps and exports the generated folders.
├── requirements.txt            # Dependencies: Defines the Python environment for the Dagger container.
└── README.md
```
## How To Run the Project on Github UI

**1. Open the Actions tab**  
Navigate to your GitHub repository and click on the **Actions tab**.  

**2. Select the workflow**  
From the left-hand menu, choose *“MLOps Pipeline”*.  

**3. Trigger the workflow**  
Click **Run workflow** at the top of the page to start the pipeline on a GitHub-hosted runner.  

**4. Monitor execution**  
Follow the progress of the job directly in the workflow run view.  

**5. Retrieve results**  
Once the run completes, open the workflow’s **Summary** page to download the generated artifacts.  


## How To Run the Project Locally 

The project can be executed locally using the Dagger orchestrator.
All pipeline steps are run inside a container and do not require a local Python environment.  
  
### Prerequisites:
Ensure the following tools are installed:  
  **Docker Desktop:** Must be running!     
  **Go:** Required to run the Dagger orchestrator (`main.go`).  
  **Dagger CLI:** The command-line interface for running the Dagger pipeline.  
  
You can verify the installation with:  
```bash
docker version
go version
dagger version
```


**1. Clone the repository**
```bash
git clone <repository-url>
cd <root of the repository>
```

**2. Execute the MLOps Pipeline:**
  
```bash
go run main.go
```
This command:  
    
- starts the Dagger engine    
- builds a containerized Python environment  
- pulls the dataset using DVC   
- runs data preparation, training, and deployment stages  
- logs experiments using MLflow  
- exports the generated artifacts and MLflow runs to the local filesystem  
  
**3. Verify the output**  
  
After a successful run, the following directories will be created:  

* **`mlruns/`:** Contains the local MLflow Tracking data and the Model Registry.
* **`artifacts/`:** Contains the final generated files, including `model.pkl`, `train_data_gold.csv`, and `model_results.json` etc.



## Authors

Earth Vangwithayakul eava@itu.dk - pecxpecx 

Aniela Marta Ciecierska anci@itu.dk - aniela2906 

Viktor Ulitin viul@itu.dk - viul1488 



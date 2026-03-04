GRaNDe- **Gaussian Rank-Based Neighborhood Degree** for
Graph Neural Networks in Image Classification:

Repository for the Scientific Initiation project that explores the develop of novel degree measure that integrates neighborhood
ranking with Gaussian distance weighting to better capture node
importance in Graph Neural Networks for image classification tasks.

## 📝 Project Summary

In image classification, graphs are usually constructed from feature similarity, where treating all neighbors equally may overlook
important variations in relevance. Motivated by this research
gap, we propose GRaNDe (Gaussian Rank-based Neighborhood
Degree).

To achieve this, we implemented a avarage dissimilarity based in Gaussian Similarity. Specifically, it computes the mean of the inverse
Gaussian similarities derived from the distances between q and
the samples in its neighborhood, where the averaging factor
corresponds to the neighborhood size. Then, this score is added to the degree centrality.

## 📂 Project Structure

The code has been modularized to ensure clarity and maintainability, following the structure below:

```
/
├── data/                    # Folder for storing datasets 
├── main.py                  # Main script to run experiments
├── config.py                # Control panel with hyperparameters
├── data_loader.py           # Module for loading data
├── gcn_model.py             # Module with the GCN model definition and class
├── utils.py                 # Helper functions (K-Fold, BallTree, etc.)
└── README.md                # This file
```

## ⚙️ Installation and Setup

Follow the steps below to set up the environment and run the project.

**1. Clone the repository:**
```bash
git clone 
```

**2. Create and activate a virtual environment (recommended):**
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the dependencies:**
```bash
pip install -r requirements.txt
```

## How to Run an Experiment

All experiments are controlled and executed from the `main.py` script. To configure a run, open `main.py` and edit the "Control Panel" at the beginning of the `main()` function.

**Example of the Control Panel in `main.py`:**
```python
def main():
    # --- EXPERIMENT CONTROL PANEL ---
    features_extractors= ['resnet'] 
    DATASET_TO_USE= 'flowers'
    GCN_TYPE_TO_USE= 'sgc_grande' # appnp_grande ou gcn_degree, sgc_grande ou sgc_degree
    NEURONS_LIST= [256]
    # ------------------------------------
    
    # ... rest of the code ...
```

After setting the desired parameters, run the script from the terminal (in the project's root folder):
```bash
python main.py
```

## 📊 Results

Experiments were conducted on the **Flowers-17** dataset, using features extracted by different CNN architectures. The table below summarizes the main accuracy results (mean ± standard deviation over 10-fold cross-validation).

| Feature Extractor | GNN Model    | Final Accuracy          |
|-------------------|--------------|-------------------------|
| ResNet            | APPNP-GRaNDe | **86.19% ± 0.121%** |
| ResNet            | SGC-GRaNDe   | **85.11% ± 0.021%** |

## 🎓 Author and Acknowledgements

* **Author:** R. M. Duarte
* **Advisor:** L. P. Valem

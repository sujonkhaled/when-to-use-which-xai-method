# When to Use Which Explainable AI Method?
## A Cross-Domain Study and Practitioner Decision Framework

Paper: "When to Use Which Explainable AI Method?: A Cross-Domain Study and Practitioner Decision Framework"  
Authors: Khaled Mahmud Sujon et al.  
Affiliation: Monash University, Australia  

---

## Repository Structure

```
/
├── NB1_Healthcare_PIMA_Final.py           ← D1: PIMA Diabetes
├── NB2_Cybersecurity_UNSW_Final.py        ← D2: UNSW-NB15
├── NB3_Finance_GermanCredit_Final.py      ← D3: German Credit
├── NB4_Education_StudentPerformance_Final.py ← D4: Student Performance
├── NB5_Image_ChestXray_Final.py           ← D5: Chest X-Ray Pneumonia
└── README.md
```

---

## Datasets

All datasets are publicly available:

| Domain | Dataset | Source |
|--------|---------|--------|
| D1 Healthcare | PIMA Indians Diabetes | https://archive.ics.uci.edu/dataset/34 |
| D2 Cybersecurity | UNSW-NB15 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| D3 Finance | German Credit | https://doi.org/10.24432/C5NC77 |
| D4 Education | Student Performance | https://doi.org/10.24432/C5TG7T |
| D5 Medical Imaging | Chest X-Ray Pneumonia | https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia |

---

## Requirements

```bash
pip install shap lime dice-ml alibi xgboost lightgbm \
            tensorflow opencv-python-headless scikit-image \
            scikit-learn pandas numpy matplotlib scipy
```

---

## How to Run

Each notebook is self-contained. Run on **Google Colab**:

1. Open a new Colab notebook
2. Copy and paste the `.py` file contents
3. Run Cell 1 (install) → restart runtime → run all remaining cells
4. All figures saved as `.pdf` and tables as `.csv`

D2 Cybersecurity: Download UNSW-NB15 from Kaggle before running  
D5 Image: Upload `kaggle.json` when prompted in Cell 3

---

 
---

## Citation

```bibtex
@article{sujon2025when,
  title   = {When to Use Which Explainable {AI} Method?: 
             A Cross-Domain Study and Practitioner 
             Decision Framework},
  author  = {Sujon, Khaled Mahmud and others},
  journal = {[Journal Name]},
  year    = {2025}
}
```

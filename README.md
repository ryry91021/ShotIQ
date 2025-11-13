# 🏀 NBA Shot Success Prediction

This project applies machine learning to predict the success of NBA shots using historical shot data. It combines thousands of individual game files into a single dataset and trains a classifier to estimate the probability that a shot will be made based on shot location, player identity, and game context.

---

## 📊 Dataset

- **Source:** [Kaggle - NBA Shots Dataset (2001–Present)](https://www.kaggle.com/datasets/techbaron13/nba-shots-dataset-2001-present)
- **Files:** 4,000+ CSVs containing individual shot data
- **Key Features:**
  - `player`, `team`, `shotX`, `shotY`, `distance`
  - `quarter`, `time_remaining`, `shot_type`
  - `made` (target variable)

---

## 🧠 Model & Methodology

- Data is loaded and combined using `pandas`
- Missing or invalid entries are removed
- Categorical features (`player`, `shot_type`) are one-hot encoded
- A **Random Forest Classifier** is trained to predict `made` (0 or 1)
- The model outputs a **make probability** based on player and shot location

### Why Random Forest?
Random Forests handle both numerical and categorical data well, model non-linear relationships, and offer feature importance for analysis.

---

## ⚙️ Features

- Predicts the probability of a shot going in based on:
  - Shot location (`shotX`, `shotY`)
  - Shot distance
  - Player identity
  - Shot type, quarter, and time remaining
- Supports user input through:
  - `ipywidgets` for Jupyter Notebooks
  - (Optional) Streamlit UI for web-based interaction

---

## 🚀 How to Use

### 1. Clone the repository & run script
```bash
git clone https://github.com/ryry91021/shotIQ.git
cd shotIQ
```
### 2. Download dataset
```bash
./scripts/download_nba_data
```
### 3. Run Python Program
```bash
python main.py
```

---

## 📈 Future Work

- Visual shot heatmaps per player
- Incorporate defender proximity and game score
- Streamlit or web-based deployment
- Player-specific model tuning

---

## 📄 License

This project is for educational and research purposes only. Data © NBA.


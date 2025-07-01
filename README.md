# Heart Disease Prediction using Decision Trees & Random Forest

This project is part of **Task 5** of my **AI & ML Internship at Elevate Labs**.  
It focuses on predicting heart disease using Decision Trees and Random Forest, with data exploration and feature importance analysis.

---

## Objective

- Perform **EDA** (Exploratory Data Analysis) with histograms, correlation matrix, pairplots.
- Build a **Decision Tree Classifier** to predict heart disease.
- Prevent overfitting by limiting tree depth.
- Build a **Random Forest Classifier** to improve accuracy.
- Analyze **feature importances** and use **cross-validation** for robust evaluation.

---

## Dataset

- **File:** `heart.csv`
- Contains patient attributes like:
  - `age`, `sex`, `cp` (chest pain type), `thalach` (max heart rate), `chol` (cholesterol), etc.
- **Target:** `target`
  - `1` indicates presence of heart disease
  - `0` indicates absence of heart disease

---

## Tools & Libraries Used

- `Pandas`, `NumPy` – data manipulation
- `Matplotlib`, `Seaborn` – visualizations
- `Scikit-learn` – models & metrics

---

## Steps Performed

1. **Exploratory Data Analysis**
   - Plotted **histograms** for all features to understand distributions.
   - Created a **correlation matrix heatmap** to see feature relationships.
   - Used a **pairplot** to inspect pairwise distributions.

2. **Decision Tree Classifier**
   - Trained a decision tree on 80% training data.
   - Visualized the full decision tree using `plot_tree`.
   - Limited depth to `max_depth=3` to control overfitting.

3. **Random Forest Classifier**
   - Built a random forest with 100 estimators.
   - Compared accuracy and printed confusion matrix, precision, recall, and F1-scores.

4. **Feature Importance**
   - Extracted feature importances from the Random Forest and plotted a bar chart.

5. **Cross-Validation**
   - Performed 5-fold cross-validation to verify the model's stability.

---

## Sample Output

- Decision Tree Accuracy: 0.80
- Decision Tree (max_depth=3) Accuracy: 0.79
- Random Forest Accuracy: 0.85
- Feature Importances:
- cp 0.18
- thalach 0.14
- oldpeak 0.13
- ca 0.12
- slope 0.11

---

Average CV Score: 0.83

---

## Key Insights

- Random Forest performed better than a single decision tree, reducing overfitting.
- `cp` (chest pain type), `thalach` (max heart rate), `oldpeak` and `ca` were top contributors.
- Cross-validation showed consistent results with low variance.

---

## Project Structure

Heart-Disease-Prediction-Tree-Forest
│
├── heart.csv # Dataset
├── heart_prediction.py # Python implementation of the models
├── README.md # This overview file

---

## Author

- **Yogesh Rajput**

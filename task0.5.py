# --------------------------------------------------------
# 📚 Import libraries
# --------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------------------------------------------
# 🗂 Load dataset
# --------------------------------------------------------
df = pd.read_csv("heart.csv")
print(df.head())
print(df.info())

# --------------------------------------------------------
# 📊 EDA: Histograms
# --------------------------------------------------------
df.hist(figsize=(20,10))
plt.tight_layout()
plt.show()

# --------------------------------------------------------
# 📊 Correlation Heatmap
# --------------------------------------------------------
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Matrix for heart.csv")
plt.show()

# --------------------------------------------------------
# 📊 Pairplot
# --------------------------------------------------------
sns.pairplot(df.sample(200)) # use sample to avoid overload
plt.show()

# --------------------------------------------------------
# 🔀 Split data
# --------------------------------------------------------
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42, 
                                                    stratify=y)

# --------------------------------------------------------
# 🌳 Decision Tree
# --------------------------------------------------------
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)

y_pred_tree = dtree.predict(X_test)
print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# --------------------------------------------------------
# 🌳 Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dtree, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

# --------------------------------------------------------
# 🌳 Overfitting control with max_depth
dtree_depth = DecisionTreeClassifier(max_depth=3, random_state=42)
dtree_depth.fit(X_train, y_train)
y_pred_depth = dtree_depth.predict(X_test)
print("\nDecision Tree (max_depth=3) Accuracy:", accuracy_score(y_test, y_pred_depth))

# --------------------------------------------------------
# 🌲 Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# --------------------------------------------------------
# 🔥 Feature Importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances:\n", importances)

plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importances from Random Forest")
plt.show()

# --------------------------------------------------------
# 🔄 Cross-Validation
scores = cross_val_score(rf, X, y, cv=5)
print("\nCross-Validation Scores:", scores)
print("Average CV Score:", np.mean(scores))

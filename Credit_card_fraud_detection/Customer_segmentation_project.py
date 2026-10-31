import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\sunil\Downloads\bank_customers.csv")
print(df.head())
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 30)
print(df.info())
print(df.describe())

#Clean the data
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

print("Missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
df["Total_Trans_Amt"] = df["Total_Trans_Amt"].fillna(df["Total_Trans_Amt"].median())

df = df.drop(columns=["Customer_ID"])
print("Shape after cleaning:", df.shape)

# 'Unknown' categories: kept as their own category (decision: dropping them would lose ~10-15% of rows)

for c in ["Marital_Status", "Education_Level", "Income_Category"]:
    print(df[c].value_counts(normalize=True).round(3), "\n")

#Exploratory data analysis
num_cols = df.select_dtypes("number").columns
df[num_cols].hist(bins=30, figsize=(14, 9))
plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 7))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation heatmap"); plt.show()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="Card_Category", ax=ax[0])
sns.countplot(data=df, x="Gender", ax=ax[1])
sns.countplot(data=df, x="Attrition_Flag", ax=ax[2])
plt.tight_layout(); plt.show()
print("Churn rate:", round((df["Attrition_Flag"] == "Attrited Customer").mean() * 100, 1), "%")

#feature engineering
df["Avg_Trans_Value"] = df["Total_Trans_Amt"] / df["Total_Trans_Ct"]

features = ["Customer_Age", "Months_on_Book", "Total_Relationship_Count", "Months_Inactive_12_mon",
            "Credit_Limit", "Avg_Utilization_Ratio", "Total_Trans_Amt", "Total_Trans_Ct", "Avg_Trans_Value"]
df[features].describe().T

#fix skew and scale
X = df[features].copy()
for c in ["Credit_Limit", "Total_Trans_Amt", "Avg_Trans_Value"]:
    X[c] = np.log1p(X[c])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled.shape

#choose the number of clusters
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_scaled)
    inertia.append(km.inertia_)
    sil.append(silhouette_score(X_scaled, km.labels_, sample_size=3000, random_state=42))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow method (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), sil, "o-"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
plt.show()
print(dict(zip(ks, np.round(sil, 3))))

#so k=4 is best fitting here
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_scaled)
df["Segment"] = km.labels_
df["Segment"].value_counts().sort_index()

#visualize the segments
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
plt.figure(figsize=(7, 5))
sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=df["Segment"], cmap="viridis", s=6)
plt.legend(*sc.legend_elements(), title="Segment")
plt.title(f"Segments in PCA space (2D explains {pca.explained_variance_ratio_.sum():.0%} of variance)")
plt.show()

#profile and name each segement
profile = df.groupby("Segment")[features].mean().round(1)
profile["Size_%"] = (df["Segment"].value_counts(normalize=True) * 100).round(1)
profile["Churn_%"] = df.groupby("Segment")["Attrition_Flag"].apply(
    lambda s: (s == "Attrited Customer").mean() * 100).round(1)
profile.T

rel = (df.groupby("Segment")[features].mean() / df[features].mean()).round(2)
plt.figure(figsize=(10, 4))
sns.heatmap(rel, annot=True, cmap="RdYlGn", center=1)
plt.title("Segment average vs overall average (1.0 = average)")
plt.show()

#name each segemnt
names = {}
# Rule-based naming so labels survive a re-run
p = df.groupby("Segment")[features].mean()
names[p["Months_Inactive_12_mon"].idxmax()] = "Dormant"
rest = p.drop(index=list(names))
names[rest["Credit_Limit"].idxmax()] = "Premium active"
rest = p.drop(index=list(names))
names[rest["Avg_Utilization_Ratio"].idxmax()] = "Revolvers"
rest = p.drop(index=list(names))
names[rest.index[0]] = "Newcomers"

df["Segment_Name"] = df["Segment"].map(names)
df.groupby("Segment_Name").agg(
    customers=("Segment", "size"),
    churn_pct=("Attrition_Flag", lambda s: round((s == "Attrited Customer").mean() * 100, 1)),
    avg_limit=("Credit_Limit", "mean"),
    avg_trans=("Total_Trans_Ct", "mean")).round(1)

#stability check and limitations
km2 = KMeans(n_clusters=K, n_init=10, random_state=7).fit(X_scaled)
print("Adjusted Rand Index vs a different seed:", round(adjusted_rand_score(km.labels_, km2.labels_), 3))


#part B churn prediction
#define targets and features
y = (df["Attrition_Flag"] == "Attrited Customer").astype(int)

num_feats = ["Customer_Age", "Months_on_Book", "Total_Relationship_Count", "Months_Inactive_12_mon",
             "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal", "Avg_Utilization_Ratio",
             "Total_Trans_Amt", "Total_Trans_Ct", "Avg_Trans_Value"]
cat_feats = ["Gender", "Marital_Status", "Education_Level", "Income_Category", "Card_Category", "Segment_Name"]

X_all = df[num_feats + cat_feats]
print("Churn rate:", round(y.mean() * 100, 1), "%")

#train test split
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42)
print(X_train.shape, X_test.shape, y_train.mean().round(3), y_test.mean().round(3))

#preprocessing+model
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

pre = ColumnTransformer([
    ("num", StandardScaler(), num_feats),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats)])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=42)}

pipes = {n: Pipeline([("pre", pre), ("model", m)]) for n, m in models.items()}

#cross validate then evaluate on test set
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score,
                             recall_score, f1_score, confusion_matrix, classification_report)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
rows = []
for name, pipe in pipes.items():
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    rows.append({"Model": name, "CV ROC-AUC": round(cv_auc, 3),
                 "Test ROC-AUC": round(roc_auc_score(y_test, prob), 3),
                 "Test PR-AUC": round(average_precision_score(y_test, prob), 3),
                 "Precision": round(precision_score(y_test, pred), 3),
                 "Recall": round(recall_score(y_test, pred), 3),
                 "F1": round(f1_score(y_test, pred), 3)})
results = pd.DataFrame(rows).set_index("Model")
results

#confusion matrix and curves for the best model
best_name = results["Test PR-AUC"].idxmax()
best = pipes[best_name]
prob = best.predict_proba(X_test)[:, 1]
print("Best model:", best_name)
print(classification_report(y_test, (prob >= 0.5).astype(int), target_names=["Stay", "Churn"]))

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ConfusionMatrixDisplay.from_predictions(y_test, (prob >= 0.5).astype(int), ax=ax[0], colorbar=False)
RocCurveDisplay.from_predictions(y_test, prob, ax=ax[1])
PrecisionRecallDisplay.from_predictions(y_test, prob, ax=ax[2])
plt.tight_layout(); plt.show()

#choose a threshold using business cost
COST_MISSED_CHURNER = 5000   # assumed lifetime value lost per churner we fail to flag (Rs)
COST_RETENTION_CALL = 300    # assumed cost of an offer/call to a flagged customer (Rs)
SAVE_RATE = 0.30             # assumed share of flagged churners we actually retain

thresholds = np.arange(0.05, 0.96, 0.01)
net = []
for t in thresholds:
    flagged = prob >= t
    tp = ((flagged) & (y_test == 1)).sum()
    value = tp * SAVE_RATE * COST_MISSED_CHURNER - flagged.sum() * COST_RETENTION_CALL
    net.append(value)
best_t = thresholds[int(np.argmax(net))]

plt.figure(figsize=(7, 4))
plt.plot(thresholds, net); plt.axvline(best_t, color="red", ls="--")
plt.xlabel("Threshold"); plt.ylabel("Net benefit (Rs, test set)")
plt.title(f"Best threshold = {best_t:.2f}"); plt.show()

pred_t = (prob >= best_t).astype(int)
print(f"At threshold {best_t:.2f}: precision {precision_score(y_test, pred_t):.2f}, "
      f"recall {recall_score(y_test, pred_t):.2f}")

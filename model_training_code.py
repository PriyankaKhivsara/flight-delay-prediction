
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, precision_recall_curve
import joblib
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

file_path = 'C:/Users/SOURABH/Desktop/DBDA/Project/flight_2019_2023_data.csv'

# Load the CSV
df = pd.read_csv(file_path)

# Convert DELAY_DUE_WEATHER to 'Yes'/'No'
df['DELAY_DUE_WEATHER_YN'] = df['DELAY_DUE_WEATHER'].apply(lambda x: 'Yes' if pd.notna(x) and x != 0 else 'No')

# Count total records
total_flights = len(df)

# Count flights delayed due to weather
weather_delayed_flights = len(df[df['DELAY_DUE_WEATHER_YN'] == 'Yes'])

# Calculate percentage
weather_delay_percent = (weather_delayed_flights / total_flights) * 100

# Get 'Yes' delayed flights
weather_delayed = df[df['DELAY_DUE_WEATHER_YN'] == 'Yes']

# Get same number of 'No' (not delayed)
weather_not_delayed = df[df['DELAY_DUE_WEATHER_YN'] == 'No'].sample(n=len(weather_delayed), random_state=42)

# Combine and shuffle
stratified_df = pd.concat([weather_delayed, weather_not_delayed]).sample(frac=1, random_state=42).reset_index(drop=True)

# 1. Clean data  by removing irrelevant columns
delay_columns = [
    'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
    'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
]
stratified_df = stratified_df.drop(columns=delay_columns, errors='ignore')

# 2. Feature engineering function 
def create_features(df):
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'])
    df['MONTH'] = df['FL_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['FL_DATE'].dt.dayofweek
    df['HOUR_OF_DAY'] = df['CRS_DEP_TIME'] // 100

    season_map = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }
    df['SEASON'] = df['MONTH'].map(season_map)

    bins = [0, 600, 1200, 1800, 2400]
    labels = ['Early_Morning', 'Morning', 'Afternoon', 'Evening']
    df['TIME_CATEGORY'] = pd.cut(df['CRS_DEP_TIME'], bins=bins, labels=labels, right=False)

    severity_map = {'Light': 1, 'Moderate': 2, 'Heavy': 3, 'Severe': 4, 'Unknown': 1}
    df['SEVERITY_SCORE'] = df['Severity'].map(severity_map)

    df['PRECIP_CAT'] = pd.cut(
        df['Precipitation(in)'],
        bins=[-1, 0.01, 0.1, 0.5, float('inf')],
        labels=['None', 'Light', 'Moderate', 'Heavy']
    )

    df['HEAVY_RAIN'] = ((df['Type'] == 'Rain') & df['Severity'].isin(['Heavy', 'Severe'])).astype(int)
    df['SNOW_STORM'] = ((df['Type'] == 'Snow') & df['Severity'].isin(['Moderate', 'Heavy', 'Severe'])).astype(int)

    return df

stratified_df = create_features(stratified_df.copy())

features = [
    'ORIGIN', 'DEST', 'CRS_DEP_TIME', 'MONTH', 'DAY_OF_WEEK', 'HOUR_OF_DAY',
    'SEASON', 'TIME_CATEGORY', 'Type', 'Severity', 'SEVERITY_SCORE',
    'Precipitation(in)', 'PRECIP_CAT', 'HEAVY_RAIN', 'SNOW_STORM'
]

X = stratified_df[features]
y = stratified_df['DELAY_DUE_WEATHER_YN'].map({'Yes': 1, 'No': 0})

categorical_cols = ['ORIGIN', 'DEST', 'Type', 'Severity', 'SEASON', 'TIME_CATEGORY', 'PRECIP_CAT']
encoder = ce.TargetEncoder(cols=categorical_cols)
X_encoded = encoder.fit_transform(X, y)

numerical_cols = ['CRS_DEP_TIME', 'Precipitation(in)', 'MONTH', 'DAY_OF_WEEK', 'HOUR_OF_DAY', 'SEVERITY_SCORE']
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

rfe_selector = RFE(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    n_features_to_select=12,
    step=0.1
)
rfe_selector.fit(X_train, y_train)

selected_features = X_train.columns[rfe_selector.support_]
print("\nSelected Features (Top 12):", list(selected_features))

categorical_selected = [col for col in categorical_cols if col in selected_features]
numerical_selected = [col for col in numerical_cols if col in selected_features]

encoder_12 = ce.TargetEncoder(cols=categorical_selected)
X_train_sel_enc = encoder_12.fit_transform(X_train[selected_features], y_train)
X_test_sel_enc = encoder_12.transform(X_test[selected_features])

scaler_12 = StandardScaler()
X_train_sel_enc[numerical_selected] = scaler_12.fit_transform(X_train_sel_enc[numerical_selected])
X_test_sel_enc[numerical_selected] = scaler_12.transform(X_test_sel_enc[numerical_selected])

model = XGBClassifier(
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.7,
    n_estimators=1200,
    reg_alpha=1.0,
    reg_lambda=1.5,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight
)

print("\nTraining XGBoost with 12 selected features...")
model.fit(X_train_sel_enc, y_train, eval_set=[(X_test_sel_enc, y_test)], verbose=100)

y_proba = model.predict_proba(X_test_sel_enc)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
high_precision_thresholds = thresholds[precision[:-1] >= 0.70]
if len(high_precision_thresholds) > 0:
    best_idx = np.argmax(recall[:-1][precision[:-1] >= 0.70])
    optimal_threshold = high_precision_thresholds[best_idx]
else:
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal Threshold: {optimal_threshold:.4f}")

y_pred = (y_proba >= optimal_threshold).astype(int)
print(f"\nModel F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"Model AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Delayed', 'Delayed']))

class FlightDelayPredictor:
    def __init__(self, encoder, scaler, model, selected_features, threshold):
        self.encoder = encoder
        self.scaler = scaler
        self.model = model
        self.selected_features = selected_features
        self.threshold = threshold

    def predict_proba(self, X):
        X_sel = X[self.selected_features]
        X_encoded = self.encoder.transform(X_sel)
        numerical_selected = [col for col in ['CRS_DEP_TIME', 'Precipitation(in)', 'MONTH', 'DAY_OF_WEEK', 'HOUR_OF_DAY', 'SEVERITY_SCORE'] if col in self.selected_features]
        X_encoded[numerical_selected] = self.scaler.transform(X_encoded[numerical_selected])
        return self.model.predict_proba(X_encoded)

    def predict(self, X, severity_adjustment=True):
        proba = self.predict_proba(X)[:, 1]
        if severity_adjustment and 'HEAVY_RAIN' in X.columns and 'SNOW_STORM' in X.columns:
            heavy_rain = X['HEAVY_RAIN'].iloc[0]
            snow_storm = X['SNOW_STORM'].iloc[0]
            if heavy_rain or snow_storm:
                adjusted_threshold = max(0.3, self.threshold * 0.7)
                return (proba >= adjusted_threshold).astype(int), proba
        return (proba >= self.threshold).astype(int), proba

final_predictor = FlightDelayPredictor(
    encoder=encoder_12,
    scaler=scaler_12,
    model=model,
    selected_features=list(selected_features),
    threshold=optimal_threshold
)

joblib.dump(final_predictor, 'flight_delay_predictor.pkl')

print("\nSaved final_predictor pipeline to 'flight_delay_predictor.pkl'.")

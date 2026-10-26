import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

# ===================== LOAD DATA =====================
df = pd.read_csv('sleep_health_dataset.csv')

X = df.drop(['felt_rested', 'sleep_disorder_risk', 'cognitive_performance_score'], axis=1)
y = df['felt_rested'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===================== PREPROCESS =====================
def preprocess_train(X, y):
    df = X.copy()
    y = y.copy()

    df.drop(['person_id', 'occupation', 'country'], axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # Outlier removal
    NUMERIC_COLS = df.select_dtypes(include=['number']).columns

    Q1 = df[NUMERIC_COLS].quantile(0.25)
    Q3 = df[NUMERIC_COLS].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    mask = pd.Series(True, index=df.index)
    for col in NUMERIC_COLS:
        mask &= (df[col] >= lower_bound[col]) & (df[col] <= upper_bound[col])

    df = df[mask]
    y = y[mask]

    # Encoding
    df = pd.get_dummies(df, drop_first=True)
    train_columns = df.columns

    # Scaling
    scaler = StandardScaler()
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])

    return df, y, scaler, train_columns


def preprocess_test(X, y, scaler, train_columns):
    df = X.copy()
    y = y.copy()

    df.drop(['person_id', 'occupation', 'country'], axis=1, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    NUMERIC_COLS = df.select_dtypes(include=['number']).columns

    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=train_columns, fill_value=0)

    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    return df, y


# ===================== PREPROCESS DATA =====================
X_train_prep, y_train, scaler, train_cols = preprocess_train(X_train, y_train)
X_test_prep, y_test = preprocess_test(X_test, y_test, scaler, train_cols)

# ===================== PIPELINE WITH GRIDSEARCHCV =====================
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('variance_threshold', VarianceThreshold(threshold=0.1)),
    ('pca', PCA(n_components=0.95)),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['rbf', 'poly'],
    'pca__n_components': [0.80, 0.95],
    'variance_threshold__threshold': [0.05, 0.1]
}

#print("Running GridSearchCV...")

#grid_search = GridSearchCV(
#    svm_pipeline,
#    param_grid,
#    cv=5,
#    scoring='accuracy',
#    n_jobs=-1,
#    verbose=1
#)

#grid_search.fit(X_train_prep, y_train)

#print(f"\nBest parameters: {grid_search.best_params_}")
#print(f"Best CV score: {grid_search.best_score_:.2f}")

#best_model = grid_search.best_estimator_
#best_test_acc = best_model.score(X_test_prep, y_test)
#print(f"Best test accuracy: {best_test_acc:.2f}")

# Save model
#joblib.dump(best_model, 'svm_pipeline_model.joblib')
#print("Model saved to 'svm_pipeline_model.joblib'")


trained_model = joblib.load('svm_pipeline_model_final.joblib')
test_acc = trained_model.score(X_test_prep, y_test)
print(f"Trained model test accuracy: {test_acc:.2f}")


# ===================== STREAMLIT APP =====================
##   streamlit run "d:\Studing\data computation\Final Project\Final\DataComputationProj.py"

import streamlit as st 
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report
 
st.set_page_config(page_title="Sleep Health SVM", layout="wide")
st.title("Sleep Health — SVM Classifier")
 
 
# ---- Cache the heavy work so it only runs once ----
 
@st.cache_data
def load_and_prepare():
    """Loads the CSV, splits, and preprocesses — runs only once."""
    _df = pd.read_csv('sleep_health_dataset.csv')
 
    _X = _df.drop(['felt_rested', 'sleep_disorder_risk', 'cognitive_performance_score'], axis=1)
    _y = _df['felt_rested'].astype('category')
 
    _X_train_raw, _X_test_raw, _y_train_raw, _y_test_raw = train_test_split(
        _X, _y, test_size=0.2, random_state=42
    )
 
    _X_train_prep, _y_train, _scaler, _train_cols = preprocess_train(_X_train_raw, _y_train_raw)
    _X_test_prep,  _y_test               = preprocess_test(_X_test_raw, _y_test_raw, _scaler, _train_cols)
 
    # Also keep the raw numeric columns for the "before" boxplot
    _num_cols_raw = _X_train_raw.drop(['person_id', 'occupation', 'country'], axis=1) \
                                .select_dtypes(include='number').columns.tolist()
 
    return (_X_train_raw, _X_train_prep, _X_test_prep,
            _y_train, _y_test, _scaler, _train_cols, _num_cols_raw)
 
 
@st.cache_resource
def load_model():
    """Loads the saved SVM pipeline — runs only once."""
    return joblib.load('svm_pipeline_model_final.joblib')
 
 
# ---------- load everything ----------
(X_train_raw, X_train_prep, X_test_prep,
 y_train, y_test, scaler, train_cols, num_cols_raw) = load_and_prepare()
 
model = load_model()
 
# ===================== TABS =====================
tab1, tab2, tab3 = st.tabs(["EDA & Preprocessing", "Model Results", "Predict"])
 
 
# ===================== TAB 1 — EDA =====================
with tab1:
 
    # --- Boxplot BEFORE outlier removal ---
    st.subheader("Boxplot — Before Outlier Removal")
    num_cols_raw_list = X_train_raw.drop(['person_id', 'occupation', 'country'], axis=1) \
                                   .select_dtypes(include='number').columns.tolist()
    fig_before = px.box(
        X_train_raw[num_cols_raw_list],
        title="All Numeric Features (before outlier removal)",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_before.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig_before, use_container_width=True)
 
    st.divider()
 
    # --- Boxplot AFTER outlier removal ---
    st.subheader("Boxplot — After Outlier Removal")
    num_cols_prep = X_train_prep.select_dtypes(include='number').columns.tolist()
    fig_after = px.box(
        X_train_prep[num_cols_prep],
        title="All Numeric Features (after outlier removal & scaling)",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_after.update_layout(height=500, xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig_after, use_container_width=True)
 
    st.divider()
 
    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap (Training Features)")
    corr = X_train_prep.corr()
    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
        aspect="auto"
    )
    fig_heat.update_layout(height=700)
    st.plotly_chart(fig_heat, use_container_width=True)
 
    st.divider()
 
    # --- Target class distribution ---
    st.subheader("Target Variable Distribution (felt_rested)")
    fig_dist = px.histogram(
        x=y_train.astype(str),
        labels={"x": "Felt Rested (0 = No, 1 = Yes)", "count": "Count"},
        title="Class Distribution in Training Set",
        color=y_train.astype(str),
        color_discrete_map={"0": "#EF553B", "1": "#00CC96"}
    )
    st.plotly_chart(fig_dist, use_container_width=True)
 
 
# ===================== TAB 2 — MODEL RESULTS =====================
with tab2:
 
    y_pred = model.predict(X_test_prep)
 
    # Accuracy metric
    test_acc = model.score(X_test_prep, y_test)
    st.metric("✅ Test Accuracy", f"{test_acc:.2%}")
 
    st.divider()
 
    # --- Confusion Matrix ---
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Label", y="True Label"),
        x=["Not Rested (0)", "Rested (1)"],
        y=["Not Rested (0)", "Rested (1)"],
        title="Confusion Matrix — Test Set"
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    st.caption(
        "Each cell shows how many test samples were classified into that category. "
        "The diagonal (top-left, bottom-right) are correct predictions."
    )
 
    st.divider()
 
    # --- Classification Report ---
    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)
 
 
# ===================== TAB 3 — PREDICT =====================
with tab3:
    st.subheader("Enter your data and the model will predict whether you'll feel rested")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Sleep Info**")
            sleep_duration_hrs          = st.number_input("Sleep Duration (hrs)", 3.0, 12.0, 7.0, step=0.1)
            sleep_quality_score         = st.number_input("Sleep Quality Score (1–10)", 1.0, 10.0, 5.0, step=0.1)
            rem_percentage              = st.number_input("REM % of Sleep", 10.0, 30.0, 20.0, step=0.5)
            deep_sleep_percentage       = st.number_input("Deep Sleep % of Sleep", 5.0, 30.0, 20.0, step=0.5)
            sleep_latency_mins          = st.number_input("Sleep Latency (mins)", 1, 60, 15)
            wake_episodes_per_night     = st.number_input("Wake Episodes per Night", 0, 10, 3)
            sleep_aid_used              = st.selectbox("Sleep Aid Used?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            weekend_sleep_diff_hrs      = st.number_input("Weekend vs Weekday Sleep Diff (hrs)", -1.0, 3.0, 1.2, step=0.1)

        with col2:
            st.markdown("**Lifestyle**")
            age                         = st.number_input("Age", 18, 80, 30)
            bmi                         = st.number_input("BMI", 16.0, 50.0, 25.0, step=0.1)
            caffeine_mg_before_bed      = st.number_input("Caffeine Before Bed (mg)", 0, 400, 0, step=10)
            alcohol_units_before_bed    = st.number_input("Alcohol Units Before Bed", 0.0, 6.0, 0.0, step=0.5)
            screen_time_before_bed_mins = st.number_input("Screen Time Before Bed (mins)", 0, 180, 60, step=5)
            exercise_day                = st.selectbox("Exercised Today?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            steps_that_day              = st.number_input("Steps That Day", 500, 20000, 7500, step=100)
            nap_duration_mins           = st.number_input("Nap Duration (mins)", 0, 120, 0, step=5)

        with col3:
            st.markdown("**Other Info**")
            stress_score                = st.number_input("Stress Score (1–10)", 1.0, 10.0, 5.0, step=0.1)
            work_hours_that_day         = st.number_input("Work Hours Today", 0.0, 18.0, 8.0, step=0.5)
            heart_rate_resting_bpm      = st.number_input("Resting Heart Rate (bpm)", 45, 99, 67)
            shift_work                  = st.selectbox("Shift Work?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            room_temperature_celsius    = st.number_input("Room Temperature (°C)", 15.0, 28.0, 20.0, step=0.5)
            gender                      = st.selectbox("Gender", ["Female", "Male", "Other"])
            chronotype                  = st.selectbox("Chronotype", ["Evening", "Morning", "Neutral"])
            mental_health_condition     = st.selectbox("Mental Health Condition", ["Anxiety", "Both", "Depression", "Healthy"])
            season                      = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])
            day_type                    = st.selectbox("Day Type", ["Weekday", "Weekend"])

        st.divider()
        predict_button = st.form_submit_button("Predict!", type="primary")

    if predict_button:
        input_df = pd.DataFrame([{  
            'age':                         age,
            'gender':                      gender,
            'bmi':                         bmi,
            'sleep_duration_hrs':          sleep_duration_hrs,
            'sleep_quality_score':         sleep_quality_score,
            'rem_percentage':              rem_percentage,
            'deep_sleep_percentage':       deep_sleep_percentage,
            'sleep_latency_mins':          sleep_latency_mins,
            'wake_episodes_per_night':     wake_episodes_per_night,
            'caffeine_mg_before_bed':      caffeine_mg_before_bed,
            'alcohol_units_before_bed':    alcohol_units_before_bed,
            'screen_time_before_bed_mins': screen_time_before_bed_mins,
            'exercise_day':                exercise_day,
            'steps_that_day':              steps_that_day,
            'nap_duration_mins':           nap_duration_mins,
            'stress_score':                stress_score,
            'work_hours_that_day':         work_hours_that_day,
            'chronotype':                  chronotype,
            'mental_health_condition':     mental_health_condition,
            'heart_rate_resting_bpm':      heart_rate_resting_bpm,
            'sleep_aid_used':              sleep_aid_used,
            'shift_work':                  shift_work,
            'room_temperature_celsius':    room_temperature_celsius,
            'weekend_sleep_diff_hrs':      weekend_sleep_diff_hrs,
            'season':                      season,
            'day_type':                    day_type,
            'person_id': 0, 'occupation': 'Unknown', 'country': 'Unknown'
        }])

        input_prep, _ = preprocess_test(input_df, pd.Series([0]), scaler, train_cols)
        prediction = model.predict(input_prep)[0]
        
        if prediction == 1:
            st.success("**Prediction: You WILL feel rested!**")
        else:
            st.error("**Prediction: You will NOT feel rested.**")



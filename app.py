import streamlit as st
import numpy as np
import pandas as pd
import joblib



st.set_page_config(page_title="🌱 Dry Bean Classifier", layout="centered")
st.title("🌱 Dry Bean Type Classifier 🫘")
st.markdown("""
<span style='font-size:1.2em'><b>Objective</b></span><br>
Manual classification of dry bean types in agricultural operations is labor-intensive, error-prone, and inefficient at scale. Automating this process using artificial intelligence can improve accuracy, reduce operational costs, and ensure consistent quality in packaging and distribution.<br><br>
The goal is to build a machine learning solution that accurately classifies bean types based on physical characteristics such as area, perimeter, shape, and compactness. This enables automation of quality control, reduction of manual labor and cost, and delivery of a scalable solution for real-time classification in industrial settings.<br><br>
The project demonstrates the application of supervised machine learning in Agri-tech and food processing industries for tangible business impact.<br>
<hr>
<span style='font-size:1.2em'>✨ Upload a CSV file or enter feature values to predict the bean class! ✨</span>
""", unsafe_allow_html=True)

# Load model and label encoder
def load_model():
    try:
        model = joblib.load("best_rf_model_tuned.pkl")
        st.success("Loaded Random Forest (SMOTE, Tuned) model.")
    except Exception:
        try:
            model = joblib.load("best_gb_model_tuned.pkl")
            st.success("Loaded Gradient Boosting (SMOTE, Tuned) model.")
        except Exception:
            model = joblib.load("best_bean_classifier_model.pkl")
            st.success("Loaded default best model.")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, label_encoder

model, label_encoder = load_model()

# Feature names (order must match training)
feature_names = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation',
    'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
    'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
]

def predict(features):
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)
    pred_label = label_encoder.inverse_transform(pred)[0]
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(arr)[0]
    elif hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'predict_proba'):
        proba = model.steps[-1][1].predict_proba(arr)[0]
    else:
        proba = None
    return pred_label, proba

# Input method selection
input_method = st.radio("Choose input method:", ("Manual Entry", "CSV Upload"))

if input_method == "Manual Entry":
    st.subheader("📝 Enter Feature Values")

    # Preloaded real sample values for each class
    class_samples = {
        'SEKER': [28395,610.291,208.1781167,173.888747,1.197191424,0.549812187,28715,190.1410973,0.763922518,0.988855999,0.958027126,0.913357755,0.007331506,0.003147289,0.834222388,0.998723889],
        'BARBUNYA': [41487,815.9,299.046841,177.0814896,1.688752684,0.805825541,42483,229.8323062,0.68917572,0.976555328,0.783155548,0.76854952,0.007208206,0.001551295,0.590668365,0.997492788],
        'BOMBAY': [114004,1279.356,451.3612558,323.7479961,1.394174671,0.696795094,115298,380.9913399,0.748986604,0.988776909,0.875280258,0.844094027,0.00395917,0.001239788,0.712494726,0.9933421],
        'CALI': [45504,793.417,295.4698306,196.3118225,1.505104618,0.747372158,45972,240.7020819,0.737779075,0.98981989,0.908356725,0.814641825,0.006493272,0.001764047,0.663641303,0.998849589],
        'DERMASON': [20420,524.932,183.601165,141.8862155,1.294002834,0.634654708,20684,161.2437642,0.790186518,0.987236511,0.931235461,0.878228437,0.008991242,0.003299358,0.771285188,0.99804522],
        'HOROZ': [33006,710.496,283.0203847,149.6237186,1.891547593,0.848828915,33354,204.9988888,0.635476232,0.989566469,0.821636048,0.724325525,0.008574816,0.001455927,0.524647467,0.992395653],
        'SIRA': [31519,676.641,255.0735621,157.80274,1.616407689,0.785662138,32065,200.3278244,0.758032708,0.982972088,0.865098731,0.785372748,0.008092692,0.001899224,0.616810353,0.997016997]
    }

    preload_class = st.selectbox("🔄 Preload values for class:", ["None"] + list(class_samples.keys()), index=0)

    if preload_class != "None":
        # Ensure all default values are float to avoid Streamlit warning
        default_values = [float(x) for x in class_samples[preload_class]]
    else:
        default_values = [0.0] * len(feature_names)

    cols = st.columns(4)
    user_input = []
    for i, fname in enumerate(feature_names):
        val = cols[i % 4].number_input(fname, value=float(default_values[i]), format="%.6f", key=fname)
        user_input.append(val)

    if st.button("🔮 Predict Bean Class"):
        pred_label, proba = predict(user_input)
        st.success(f"🎉 Predicted Bean Class: 🫘 **{pred_label}**")
        if proba is not None:
            proba_df = pd.DataFrame({
                'Class': label_encoder.classes_,
                'Probability': proba
            }).sort_values('Probability', ascending=False)
            st.write("📊 Prediction Probabilities:")
            st.dataframe(proba_df, hide_index=True)

elif input_method == "CSV Upload":
    st.subheader("📁 Upload CSV File")
    uploaded_file = st.file_uploader("📤 Choose a CSV file with the correct feature columns.", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if all(f in df.columns for f in feature_names):
            X = df[feature_names]
            preds = model.predict(X)
            pred_labels = label_encoder.inverse_transform(preds)
            df['Predicted_Class'] = pred_labels
            st.success(f"✅ Predictions complete for all {len(df)} rows!")
            st.dataframe(df)
            # Show probabilities for all rows (first 5 for preview)
            if hasattr(model, 'predict_proba') or (hasattr(model, 'steps') and hasattr(model.steps[-1][1], 'predict_proba')):
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X)
                else:
                    probas = model.steps[-1][1].predict_proba(X)
                proba_df = pd.DataFrame(probas, columns=label_encoder.classes_)
                st.write(f"📊 Prediction Probabilities (showing first 5 of {len(df)} rows):")
                st.dataframe(proba_df.head(), hide_index=True)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="bean_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"❌ CSV must contain columns: {feature_names}")

st.markdown("---")
st.markdown("""
📝 **Instructions:**

Made with ❤️ and Streamlit. 🫘
<br><br>
<span style='font-size:1.1em'><b>Developed by Ramasamy_A_Batch_11 for a Mini Project-Supervised ML</b></span>
""", unsafe_allow_html=True)
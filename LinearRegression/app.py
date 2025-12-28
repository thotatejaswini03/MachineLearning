import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config("Linear Regression", layout="centered")

# Load CSS
import os


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)
css_path = os.path.join(BASE_DIR, "style.css")

load_css(css_path)



# Title
st.markdown("""
<div class="card">
    <h1>Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Linear Regression.</p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Prepare data
X, y = df[["total_bill"]], df["tip"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - 2)

# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

x_line = np.linspace(
    df["total_bill"].min(),
    df["total_bill"].max(),
    100
).reshape(-1, 1)

x_line_scaled = scaler.transform(x_line)
y_line = model.predict(x_line_scaled)

ax.plot(x_line, y_line, linewidth=3)
ax.set_xlabel("Total Bill ($)", color="black")
ax.set_ylabel("Tip ($)", color="black")
ax.tick_params(axis="both", colors="black")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# Model parameters
st.markdown(f"""
<div class="card">
    <h3>Model Intercept & Coefficient</h3>
    <p>
        <b>Coefficient:</b> {model.coef_[0]:.3f}<br>
        <b>Intercept:</b> {model.intercept_:.3f}
    </p>
</div>
""", unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

bill = st.slider(
    "Total Bill ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="prediction-box">Predict Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)



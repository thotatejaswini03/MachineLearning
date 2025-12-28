import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Multiple Linear Regression",
    layout="centered"
)
import os


def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(__file__)
css_path = os.path.join(BASE_DIR, "style.css")

load_css(css_path)
st.markdown("""
<div class="card">
    <h1> Multiple Linear Regression </h1>
    <p>
        Predict <b>Tip Amount</b> using multiple features  
        (<b>Total Bill, Size, Day, Time & Smoker</b>)
    </p>
</div>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(" Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

df_encoded = df.copy()

df_encoded["sex"] = df_encoded["sex"].map({"Male": 1, "Female": 0})
df_encoded["smoker"] = df_encoded["smoker"].map({"Yes": 1, "No": 0})
df_encoded["time"] = df_encoded["time"].map({"Lunch": 0, "Dinner": 1})
df_encoded["day"] = df_encoded["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})

X = df_encoded[["total_bill", "size", "sex", "smoker", "day", "time"]]
y = df_encoded["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X.shape[1] - 1)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(" Actual vs Predicted Tips")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.7, color="#f97316")
ax.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="#a855f7", linewidth=2)

ax.set_xlabel("Actual Tip")
ax.set_ylabel("Predicted Tip")
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(" Model Performance")
c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="card">
    <h3> Model Coefficients</h3>
    <p>
        <b>Total Bill:</b> {model.coef_[0]:.3f}<br>
        <b>Size:</b> {model.coef_[1]:.3f}<br>
        <b>Sex:</b> {model.coef_[2]:.3f}<br>
        <b>Smoker:</b> {model.coef_[3]:.3f}<br>
        <b>Day:</b> {model.coef_[4]:.3f}<br>
        <b>Time:</b> {model.coef_[5]:.3f}<br><br>
        <b>Intercept:</b> {model.intercept_:.3f}
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader(" Predict Tip Amount")

bill = st.slider(" Total Bill ($)", float(df.total_bill.min()), float(df.total_bill.max()), 30.0)
size = st.slider(" Party Size", 1, 6, 2)
sex = st.selectbox(" Sex", ["Male", "Female"])
smoker = st.selectbox(" Smoker", ["Yes", "No"])
day = st.selectbox(" Day", ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox(" Time", ["Lunch", "Dinner"])

input_data = np.array([[
    bill,
    size,
    1 if sex == "Male" else 0,
    1 if smoker == "Yes" else 0,
    {"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3}[day],
    0 if time == "Lunch" else 1
]])

input_scaled = scaler.transform(input_data)
predicted_tip = model.predict(input_scaled)[0]

st.markdown(
    f'<div class="prediction-box"> Predicted Tip: $ {predicted_tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

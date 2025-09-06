# 🧠 Disease Prediction Dashboard

An interactive Streamlit app that predicts **COVID**, **Flu**, **Cold**, or **Allergy** based on user-selected symptoms. Built with `scikit-learn`, `pandas`, and `matplotlib`, this dashboard includes:

- ✅ Real-time prediction using a trained ML model  
- 📊 Confidence score visualization  
- 🗂️ Logging of user inputs and predictions  
- 🎨 Clean, responsive UI with sidebar branding

---

## 🚀 Demo

![Dashboard Screenshot](link-to-screenshot-if-you-have-one)

Try it live (if deployed): [Streamlit App](link-to-deployment)

---

## 📦 Features

- Select symptoms via checkboxes  
- Predict disease instantly  
- View model confidence scores  
- Log predictions to CSV for future tuning  
- Sidebar with GitHub link and branding

---

## 🧠 Model Details

- **Algorithm**: Logistic Regression  
- **Training Data**: Custom symptom dataset  
- **Classes**: COVID (0), FLU (1), COLD (2), ALLERGY (3)  
- **Balanced** using `class_weight='balanced'`  
- Saved as `model.pkl` using `joblib`

---

## 📁 Project Structure
## 🚀 Live Demo

Try the app here: [Disease Predictor Dashboard](https://predict-disease-ai.streamlit.app)

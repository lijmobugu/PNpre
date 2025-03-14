# 基础库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
import shap
# 机器学习库
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE, mutual_info_classif
from sklearn.metrics import roc_auc_score, make_scorer

# 分类算法
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from lime.lime_tabular import LimeTabularExplainer  
import joblib
import logging
import warnings
import streamlit as st  
# 加载模型
try:
    model = joblib.load('stacking_final.pkl')
except FileNotFoundError:
    st.error("Model file 'stacking_final.pkl' not found. Please upload the model file.")
    st.stop()

# 特征范围定义
feature_names = [
    "age", "cm", "ASA score", "Smoke", "Drink","Fever", "linbaxibaojishu", "HB", "PLT","ningxuemeiyuanshijian"
]
feature_ranges = {
    "age": {"type": "numerical", "min": 18, "max": 80, "default": 40},
    "cm": {"type": "numerical", "min": 140, "max": 170, "default": 160},
    "ASA score": {"type": "categorical", "options": ["1", "2", "3"]},
    "Smoke": {"type": "categorical", "options": ["YES", "NO"]},
    "Drink": {"type": "categorical", "options": ["YES", "NO"]},
    "Fever": {"type": "categorical", "options": ["YES", "NO"]},
    "linbaxibaojishu": {"type": "numerical", "min": 0, "max": 170, "default": 0},
    "HB": {"type": "numerical", "min": 0, "max": 200, "default": 0},
    "PLT": {"type": "numerical", "min": 0, "max": 170, "default": 0},
    "ningxuemeiyuanshijian": {"type": "numerical", "min": 0, "max": 170, "default": 0}
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

feature_values = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        feature_values[feature] = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        feature_values[feature] = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )

# 处理分类特征
label_encoders = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(properties["options"])
        feature_values[feature] = label_encoders[feature].transform([feature_values[feature]])[0]

# 转换为模型输入格式
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测与 SHAP 可视化
if st.button("Predict"):
    try:
        # 模型预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # 提取预测的类别概率
        probability = predicted_proba[predicted_class] * 100

        # 显示预测结果
        st.subheader("Prediction Result:")
        st.write(f"Predicted possibility of AKI is **{probability:.2f}%**")

    except Exception as e:
        st.error(f"An error occurred: {e}")

  # 预测与 LIME 可视化  
if st.button("Predict"):  
    try:  
        # 模型预测  
        predicted_class = model.predict(features)[0]  
        predicted_proba = model.predict_proba(features)[0]  

        # 提取预测的类别概率  
        probability = predicted_proba[predicted_class] * 100  

        # 显示预测结果  
        st.subheader("Prediction Result:")  
        st.write(f"Predicted possibility of AKI is **{probability:.2f}%**")  

        # 创建LIME解释器  
        explainer = LimeTabularExplainer(  
            training_data=features.values,  # 训练数据  
            feature_names=feature_names,     # 特征名称  
            class_names=["No AKI", "AKI"],   # 类别名称  
            mode='classification'             # 模式  
        )  

        # 生成解释  
        exp = explainer.explain_instance(  
            data_row=features.values[0],    # 当前预测的特征  
            predict_fn=model.predict_proba    # 预测函数  
        )  

        # 显示LIME解释图  
        fig = exp.as_pyplot_figure()  
        st.pyplot(fig)  

    except Exception as e:  
        st.error(f"An error occurred: {e}")  

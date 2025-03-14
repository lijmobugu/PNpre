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

# 预测与 SHAP 可视化
if st.button("Predict"):
    try:
        # ... [保持原有的预测代码不变] ...

        # 模型预测部分保持不变
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]
        probability = predicted_proba[predicted_class] * 100
        st.subheader("Prediction Result:")
        st.write(f"Predicted possibility of AKI is **{probability:.2f}%**")

        # ============== 修改的SHAP部分开始 ==============
        def model_predict(x):
            return model.predict_proba(x)

        # 生成更稳定的背景数据
        background = shap.utils.sample(features.values, 10)
        
        # 创建解释器时添加种子保证可重复性
        explainer = shap.KernelExplainer(model_predict, background, seed=42)
        
        # 计算SHAP值时限制样本数量
        shap_values = explainer.shap_values(features.values, nsamples=50)
        
        # 使用matplotlib绘制静态force plot
        st.subheader("SHAP Force Plot")
        try:
            plt.figure(figsize=(10, 3))
            shap.force_plot(
                base_value=explainer.expected_value[1],
                shap_values=shap_values[1][0],
                features=features.values[0],
                feature_names=feature_names,
                matplotlib=True,
                show=False  # 禁用自动显示
            )
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
        except Exception as plot_error:
            st.error(f"可视化生成失败: {str(plot_error)}")
            st.markdown("""**备用可视化方案**:  
            使用特征影响表格代替SHAP图""")

        # 显示特征影响表格
        st.subheader("Feature Impact Analysis")
        impact_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': shap_values[1][0],
            'Feature Value': features.values[0]
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        # 添加颜色标记
        def color_shap(val):
            color = 'red' if val > 0 else 'blue'
            return f'color: {color}; font-weight: bold'
        
        st.dataframe(
            impact_df.style.applymap(color_shap, subset=['SHAP Value']),
            height=400
        )
        # ============== 修改的SHAP部分结束 ==============

    except Exception as e:
        st.error(f"系统错误: {str(e)}")
        st.markdown("""**故障排除建议**:  
        1. 确认模型文件格式正确  
        2. 检查输入值是否在有效范围内  
        3. 尝试重新加载页面""")

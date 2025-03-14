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

    except Exception as e:
        st.error(f"An error occurred: {e}")

# 在预测结果显示之后替换为以下代码

# SHAP可视化部分
st.subheader("SHAP特征影响解析")

try:
    # 数据完整性检查
    if features.isnull().any().any() or np.isinf(features.values).any():
        raise ValueError("输入数据包含缺失值或无限值，请检查所有数值特征的输入范围")

    # 使用训练数据作为背景（示例需替换实际训练数据路径）
    @st.cache_resource
    def load_background_data():
        try:
            return pd.read_csv('training_data.csv').sample(50, random_state=42)
        except:
            # 生成合法背景数据的保底方案
            synthetic_data = pd.DataFrame({feat: np.linspace(props['min'], props['max'], 50) 
                                         if props['type'] == 'numerical' 
                                         else [props['options'][0]]*50 
                                         for feat, props in feature_ranges.items()})
            return synthetic_data

    background_data = load_background_data()

    # 创建稳健的解释器
    with st.spinner("正在生成解释（预计耗时8-15秒）..."):
        explainer = shap.KernelExplainer(
            model.predict_proba, 
            background_data,
            feature_names=feature_names
        )
        
        # 分步计算SHAP值
        shap_values = explainer.shap_values(
            features,
            nsamples=50,  # 平衡精度与速度
            l1_reg="num_features(10)"  # 特征归并
        )

    # 可视化组件
    with st.expander("特征全局影响力排名"):
        plt.figure(figsize=(8, 4))
        shap.summary_plot(shap_values[1], features, 
                         plot_type="bar", 
                         color_bar=False,
                         max_display=10)
        plt.title("TOP 10关键临床指标", fontsize=12)
        st.pyplot(plt.gcf())
        plt.clf()

    with st.expander("个体化影响分解图"):
        plt.figure(figsize=(10, 4))
        shap.decision_plot(explainer.expected_value[1], 
                          shap_values[1], 
                          features.values[0],
                          feature_names=feature_names,
                          highlight=3)  # 高亮前三重要特征
        plt.xticks(fontsize=8)
        st.pyplot(plt.gcf())

except Exception as e:
    error_info = f"""
    ## 解析失败: {str(e)}
    
    **可能原因及应对措施**：
    
    1. 数据完整性问题 → 检查所有数值输入是否在指定范围内
    2. 内存限制 → 尝试刷新页面后重新提交
    3. 复杂特征交互 → 联系技术人员调整模型解释参数
    
    **技术细节供开发参考**:
    - 输入数据摘要: {features.describe().to_dict()}
    - 异常特征检测: {features.isnull().sum().to_dict()}
    """
    st.error(error_info)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="Mô tả dự án",
                   layout="wide",
                   initial_sidebar_state="auto")

st.markdown("# <center>Project 1:<span style='color:#4472C4; font-family:Calibri (Body);font-style: italic;'> Sentiment Analysis</span></center>", unsafe_allow_html=True)

st.subheader("Machine Learning", divider='rainbow')
st.markdown("#### Model Selection")
models = '''models = [
    ('KNN_3', KNeighborsClassifier(n_neighbors=3)),
    ('KNN_5', KNeighborsClassifier(n_neighbors=5)),
    ('KNN_7', KNeighborsClassifier(n_neighbors=7)),
    ('LogisticRegression', LogisticRegression(max_iter=10000)),
    ('DecisionTree', DecisionTreeClassifier()),
    ('RandomForest_100', RandomForestClassifier(n_estimators=100)),
    ('RandomForest_200', RandomForestClassifier(n_estimators=200)),
    ('RandomForest_300', RandomForestClassifier(n_estimators=300))
]
'''
st.code(models, language='python')
col1, col2, col3 = st.columns((2,60,2))
with col1:
    col1.write(' '*5)
with col2:
    col2.image('images/model_select.png')
with col3:
    col3.write(' '*5)

st.markdown("#### Logistic Regression")
col1, col2 = st.columns((60,60))
with col1:
    col1.image('images/lr_cm.png')
with col2:
    col2.image('images/roccurve.png')

col1, col2, col3 = st.columns((50, 10, 50))
with col1:
    col1.image('images/classireport_lr.JPG')
with col2:
    st.write("")
with col3:
    st.write("### Nhận xét")
    st.write(":blue[Accuracy] = 83.1%", ":blue[AUC] = 91%", ":blue[f1] = 83.1%")
    st.write(":blue[Accuracy dự đoán nhóm tiêu cực] = 81.8%")
    st.write("==> :blue[Sai số dự đoán nhóm tiêu cực thành tích cực] = 18.2%")
    
st.subheader("PySpark", divider='rainbow')
col1, col2 = st.columns((60, 60))
with col1:
    st.markdown("#### Logistic Regression")
    lr_spark ='''
    from pyspark.ml.classification import LogisticRegression
    logistic = LogisticRegression(featuresCol='P_features',
                      labelCol='Sentiment_2_idx',
                      predictionCol='Logistic_prediction')'''
    st.code(lr_spark, language='python')
    st.write(":blue[Accuracy] = 85.6%", ":blue[AUC] = 85.2%", ":blue[f1] = 85.6%")
    st.image('images/lr_pyspark_cm.JPG')
    st.write(":blue[Accuracy dự đoán nhóm tiêu cực] = 65.6% ==> :blue[Sai số] = 34.4%")
with col2:
    st.markdown("#### Random Forest")
    rf_spark ='''
    from pyspark.ml.classification import RandomForestClassifier
    rfc = RandomForestClassifier(featuresCol='P_features',
                      labelCol='Sentiment_2_idx',
                      predictionCol='RFC_prediction')'''
    st.code(rf_spark, language='python')
    st.write(":blue[Accuracy] = 81.6%", ":blue[AUC] = 84.1%", ":blue[f1] = 75.9%")
    st.image('images/rf_pyspark_cm.JPG')
    st.write(":blue[Accuracy dự đoán nhóm tiêu cực] = 12.9% ==> :blue[Sai số] = 87.1%")

st.subheader("Kết luận", divider='rainbow')
st.markdown("##### Chọn Logistic Regression trên Machine Leaning")
st.write("* :blue[Accuracy] trên test data cao: :blue[83.1%]")
st.write("* :blue[Confusion matrix] có sai số dự đoán 0 (tiêu cực) --> 1 (tích cực) thấp nhất: :blue[18.2%]")
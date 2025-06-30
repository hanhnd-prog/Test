import streamlit as st
import numpy as np
import pandas as pd
import json
from datetime import datetime,timedelta

import csv

import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler # Data normalization
from scipy.special import inv_boxcox

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from model_utils import *

st.set_page_config(layout="wide")
st.title("📈 Dự báo dòng chảy trên sông Đà")
st.subheader("Phần mềm này là sản phẩm của đề tài ĐTĐL.CN.06.23")

st.sidebar.header("⚙️ Tùy chọn điểm cần dự báo") 
prediction_site = st.sidebar.selectbox("Dự báo dòng chảy:", ["Đến hồ Lai Châu", "Đến hồ Bản Chát", "Trạm thủy văn Nậm Giàng"])

uploaded_file = st.file_uploader("📤 Upload file CSV (gồm 'date' và 'flow')", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
    #df = df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'value'})
    df = df.set_index('date')
    
    df = df.rename(columns={'value':'Flow'})
    last_row_df = [df.index[-1].date()] + df.iloc[-1].tolist()
    
    # Ve du lieu goc
    st.subheader("🔍 Dữ liệu Gốc")
    #st.line_chart(df['Flow'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index[:],
        y=df['Flow'][:],
        mode='lines', #name='Dòng chảy quan trắc',
        line=dict(color = 'skyblue')
    ))
    st.plotly_chart(fig)
    
    data = np.array(df['Flow']) # data la du lieu goc
        
    st.sidebar.header("⚙️ Thiết đặt thông số dự báo")    
    # 3.5. Creat lag features
    #n_lag = st.sidebar.slider("Số ngày trễ (lag)", 1, 10, 1)
    lag_input = st.sidebar.text_input("Nhập các giá trị lag (cách nhau bằng dấu phẩy):", "1,2,7,14")
    lags_lst = [int(i.strip()) for i in lag_input.split(",") if i.strip().isdigit()]
    #lags_lst = [1,2,3] # Các độ trễ
    
    # 3.7. Tạo nhãn dự báo multi-output (n_steps_ahead bước tiếp theo)
    #n_steps_ahead = 10 # Số bước cần dự báo trước
    n_steps_ahead = st.sidebar.slider("Số ngày dự báo (ahead):", 1, 10, 1)
    
    
    st.sidebar.header("⚙️ Tùy chọn mô hình")
    model_type = st.sidebar.selectbox("Chọn mô hình:", ["LGBM", "SVR", "Linear"])
      
    st.sidebar.header("⚙️ Tùy chọn siêu tham số")
    if model_type == "LGBM":        
        # Upload  file chua hyperparameters
        st.sidebar.subheader("📤 Tải file hyperparameters (JSON hoặc Excel)")
        hyperparameters_file = st.sidebar.file_uploader("Chọn file:", type=["json", "xlsx"])
        
        if hyperparameters_file is not None:
            try:
                if hyperparameters_file.name.endswith(".json"):
                    params_from_file = json.load(hyperparameters_file)
                elif hyperparameters_file.name.endswith(".xlsx"):
                    df_params = pd.read_excel(hyperparameters_file,header=None, skiprows=1, parse_dates=[0], names=['Parameter', 'Value'])
                    params_from_file = {}
                    for i in range(len(df_params)):
                        params_from_file[df_params['Parameter'][i]]=df_params['Value'][i]
                else:
                    st.sidebar.error("Định dạng file không hợp lệ!")
            except Exception as e:
                st.sidebar.error(f"Lỗi khi đọc file: {e}")
            #st.sidebar.subheader("🎯 Các hyperparameters đang sử dụng:")
            st.sidebar.write("Number of leaves:",int(params_from_file['num_leaves']))
            st.sidebar.write("Max depth:",int(params_from_file['max_depth']))
            st.sidebar.write("Learning rate:",params_from_file['learning_rate'])
            st.sidebar.write("Number of estimators:",int(params_from_file['n_estimators']))    
        else:
            pass        
        
        
    elif model_type == "SVR":
        pass
    elif model_type == "Linear":
        pass
            
    if st.sidebar.button("🧠 Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            model, train_pred, test_pred = train_and_predict(df,model_type,lags_lst = lags_lst, n_steps_ahead = n_steps_ahead,params=params_from_file,random_state = 42)
            # Xuất kết quả dự báo ra file Forecast_for_LaiChau.csv
            lst = last_row_df + test_pred.flatten().tolist()
            output_path = "data/Outputs/DubaoLaiChau.csv"
            with open(output_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(lst)
            
            st.success(f"Mô hình {model_type} huấn luyện xong!")
            #save_model(model)
            
            # Hien thi bang ket qua
            st.subheader("🔍 Kết quả dự báo")
            ten_cot = [str(df.index[-1].date()+timedelta(days=i)) for i in range(1,n_steps_ahead+1)]       
            test_pred = pd.DataFrame(data = test_pred,columns=ten_cot)
            st.dataframe(test_pred)
            
            # Ve hinh ket qua du bao           
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index[-10:],
                y=df['Flow'][-10:],
                mode='lines',
                name='Quan trắc gần đây',
                line=dict(color = 'skyblue')
            ))
            
            test_pred_transposed = test_pred.iloc[0].to_frame(name='value')
            fig.add_trace(go.Scatter(
                x=test_pred_transposed.index,
                y=test_pred_transposed['value'],
                mode='lines',
                name='Dự báo bằng mô hình '+model_type,
                line = dict(color='yellow')
            ))
            st.plotly_chart(fig)

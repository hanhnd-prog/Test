import pandas as pd
import numpy as np

import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler # Data normalization
from scipy.special import inv_boxcox

import pywt

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

def create_features(df):
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['year'] = df.index.year
    return df

def get_model(name):
    if name == "LGBM":
        return MultiOutputRegressor(lgb.LGBMRegressor())
    elif name =="SVR":
        return MultiOutputRegressor(SVR())
    elif name == "Linear":
        return MultiOutputRegressor(LinearRegression())

def train_and_predict(df,model_name,n_lag,n_ahead):
    pass



def train_lgbm_forecast_model(df, n_lag=7, n_ahead=3):
    # Tạo các đặc trưng trễ
    for i in range(1, n_lag+1):
        df[f"lag_{i}"] = df['value'].shift(i)
    
    df = df.dropna()
    X = df[[f"lag_{i}" for i in range(1, n_lag+1)]]
    y = pd.concat([df['value'].shift(-i) for i in range(n_ahead)], axis=1)
    y.columns = [f"t+{i+1}" for i in range(n_ahead)]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = MultiOutputRegressor(lgb.LGBMRegressor())
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    return y_test, preds

def detect_outliers(series):
    # Xác định outliers

    # series: 1-D numpy array input
    Q1 = np.quantile(series, 0.25)
    Q3 = np.quantile(series, 0.75)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    lower_compare = series <= lower_bound
    upper_compare = series >= upper_bound
    outlier_idxs = np.where(lower_compare | upper_compare)[0]
    return outlier_idxs

def wavelet_denoise(data, wavelet='db4', level=3):
    """Làm mịn dữ liệu sử dụng phân tích wavelet"""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Giữ lại hệ số xấp xỉ, đặt 0 cho các hệ số chi tiết
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    data_denoised = pywt.waverec(coeffs, wavelet)
    return data_denoised[:len(data)]

def train_and_predict(df,model_type,lags_lst = [1,2,3], n_steps_ahead = 3,params=None,random_state = 42):
    
    # ---------------------------
    # 3. Data preprocessing
    # ---------------------------
    ## 3.1. Data transformations: 
    # Box-Cox transformation: --> df_BoxCox
    df_copy = df.copy()
    df_BoxCox = df_copy.copy()
    df_BoxCox['Flow'], param_1 = stats.boxcox(df_copy['Flow'])
    # print('Optimal lamda1:', param_1)
    np.random.seed(seed=1500)

    ## 3.2. Handling outliers
    outlier_idxs_Q = detect_outliers(df_BoxCox["Flow"])
    #print("Outlier values: ", df_BoxCox["value"][outlier_idxs_Q])
    #data_box_cox_outliers = df_BoxCox.loc[outlier_idxs_Q, ['date', 'Flow']]

    # Replace Outliers by margin values
    Q1 = df_BoxCox['Flow'].quantile(0.25)
    Q3 = df_BoxCox['Flow'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    # Remove outliers from the series
    df_BoxCox_removeOutliers = df_BoxCox.copy()

    # Thay cac outliers bang cac bien tren va bien duoi tuong ung
    df_BoxCox_removeOutliers['Flow'] = df_BoxCox['Flow'].clip(lower=lower_bound,upper=upper_bound) 
    # Interpolate outlier values using linear interpolation
    # df_BoxCox_removeOutliers['Flow'].loc[outlier_idxs_Q] = df_BoxCox_removeOutliers['Flow'].loc[outlier_idxs_Q].interpolate(method='linear')

    # # Display the new dataframe in which outliers were replaced by margin values
    # df_copy_remove_outliers.to_csv('LGBM_LC_1day_data_frame_copy_remove_outliers.csv')

    ## 3.3. Data normalization: Min-max scale
    df_BoxCox_removeOutliers_scaled=df_BoxCox_removeOutliers.copy()
    scaler = MinMaxScaler()
    df_BoxCox_removeOutliers_scaled['Flow']=scaler.fit_transform(df_BoxCox_removeOutliers[['Flow']])

    # 3.4. Denoising data: Methods: WT, FFT, TSR_WT, TSR_FFT, DAE
    df_BoxCox_removeOutliers_scaled_denoised=df_BoxCox_removeOutliers_scaled.copy()
    # Wavelt Transform Denoise: denoised_method = WT
    denoised_method = 'WT'
    if denoised_method=='WT':
        df_BoxCox_removeOutliers_scaled_denoised['Flow'] = wavelet_denoise(df_BoxCox_removeOutliers_scaled['Flow'],wavelet='db4', level=3)
        # print(df_BoxCox_removeOutliers_scaled_denoised)
    if denoised_method=='FFT':
        pass
    if denoised_method=='TSR_WT':
        pass
    if denoised_method=='TSR_FFT':
        pass
    if denoised_method=='DAE':
        pass
    
    for lag in lags_lst:
        df_BoxCox_removeOutliers_scaled_denoised[f'lag_{lag}']=df_BoxCox_removeOutliers_scaled_denoised['Flow'].shift(lag)
    #print(df_BoxCox_removeOutliers_scaled_denoised.columns)

    # 3.6. Thêm features thời gian (phân loại) và encode dữ liệu này
    df_BoxCox_removeOutliers_scaled_denoised['month']=df_BoxCox_removeOutliers_scaled_denoised.index.month
    #print(df_BoxCox_removeOutliers_scaled_denoised['month'])
    # One-hot encoding cho tháng
    df_BoxCox_removeOutliers_scaled_denoised=pd.get_dummies(df_BoxCox_removeOutliers_scaled_denoised,columns=['month'],prefix='month')
    df_BoxCox_removeOutliers_scaled_denoised_copy = df_BoxCox_removeOutliers_scaled_denoised.copy()    
    # print(df_BoxCox_removeOutliers_scaled_denoised.columns)

    for i in range(1,n_steps_ahead+1):
        df_BoxCox_removeOutliers_scaled_denoised_copy[f'Flow_t+{i}']=df_BoxCox_removeOutliers_scaled_denoised_copy['Flow'].shift(-i)

    # Xóa NaN do shift()
    df_BoxCox_removeOutliers_scaled_denoised_copy.dropna(inplace=True)
    
    # 4. Xây dựng mô hình
    # 4.1. Chia tập huấn luyện và kiểm tra
    #split_index = int(0.8 * len(df_BoxCox_removeOutliers_scaled_denoised))
    split_index = len(df_BoxCox_removeOutliers_scaled_denoised_copy) - max(lags_lst)

    train = df_BoxCox_removeOutliers_scaled_denoised_copy[:split_index]
    #print("train length: ",len(train))
    test = df_BoxCox_removeOutliers_scaled_denoised[-1:]
    #print("test length: ",len(test))

    # Dùng 1 trong 2 cách 1 hoặc 2 sau:
    # 1.
    X_train = train.drop(columns= [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    y_train = train[[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    # y_train.to_csv('LC_y_train.csv')

    # X_test = test.drop(columns= [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    X_test = test
    # y_test = test[[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    # y_test.to_csv('LC_y_test.csv')
    
    print("X_test:",X_test.shape)
    
    # 2.
    # X_train = train.drop(columns=[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    # y_train = train[['Flow'] + [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    # y_train.to_csv('LC_y_train.csv')

    # X_test = test.drop(columns=[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    # y_test = test[['Flow'] + [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    # y_test.to_csv('LC_y_test.csv')
    
    if params is None:
        params = {}
        
    # Chon mo hinh phu hop
    if model_type =="LGBM":
        model = MultiOutputRegressor(lgb.LGBMRegressor(num_leaves=int(params['num_leaves']),
                                                        max_depth=int(params['max_depth']),
                                                        learning_rate=params['learning_rate'],
                                                        n_estimators=int(params['n_estimators']),
                                                        random_state=random_state))
    elif model_type =="SVR":
        pass
    elif model_type == "Linear":
        pass
    
    model.fit(X_train,y_train)

    ## 4.4. Dự báo trên tập train và test
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    #print('X_test',X_test)

    ## 4.5. inverse min-max scaler
    train_pred_unscaled = scaler.inverse_transform(train_pred)
    test_pred_unscaled = scaler.inverse_transform(test_pred)

    # Y_train_obs_unscaled = scaler.inverse_transform(y_train)
    # Y_test_obs_unscaled = scaler.inverse_transform(Y_test_obs)

    ## 4.6. inverse box-cox
    train_pred = inv_boxcox(train_pred_unscaled, param_1)
    test_pred = inv_boxcox(test_pred_unscaled, param_1)
    return model, train,train_pred,test_pred

def append_results_to_csv():
    pass

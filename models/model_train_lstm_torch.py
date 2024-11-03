import mlflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# LSTM 모델 정의 함수
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# LSTM 모델 학습 함수
def model_train_lstm(url, experiment_name, X_train, X_test, y_train, y_test):
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(url)

    # 데이터 형태 조정
    X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    

    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)

    # MLFlow run 시작
    with mlflow.start_run():
        # LSTM 모델 파라미터 로깅
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("units", 50)
        mlflow.log_param("activation", "relu")
        mlflow.log_param("optimizer", "adam")

        # 모델 학습
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        # 평가 지표 계산
        predictions = model.predict(X_test).flatten()
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # 평가 결과 로깅
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # 학습 곡선 (loss) 시각화 및 로깅
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training and Validation Loss")
        plt.savefig("training_validation_loss.png")
        mlflow.log_artifact("training_validation_loss.png")

        # 모델 기록
        mlflow.keras.log_model(model, "model")

    print(f"Model training complete. MAE: {mae}, RMSE: {rmse}")




import preprocessing

X_train, X_test, y_train, y_test = preprocessing.preprocessing(file_path="../data/train.csv",test_size=0.2)

# 사용 예시
model_train_lstm("http://127.0.0.1:5000", "bike_prediction_experiment_torch", X_train, X_test, y_train, y_test)

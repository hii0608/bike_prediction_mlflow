import mlflow.sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np


# 모델 정의
model = DecisionTreeRegressor(max_depth=10, random_state=42)

def model_train_decision(model, url, experiment_name, X_train, X_test, y_train, y_test):
  import mlflow
  mlflow.set_experiment("bike_prediction_experiment")
  mlflow.set_tracking_uri("http://127.0.0.1:5000")


  # Start an MLFlow run
  with mlflow.start_run():
      #model parameter를 log
      mlflow.log_param("model_type", "DecisionTreeRegressor")
      mlflow.log_param("max_depth", 10)

      # 모델 학습
      model.fit(X_train, y_train)

      # test set에 대한 예측
      predictions = model.predict(X_test)

      # evaluation metrics 계산
      mae = mean_absolute_error(y_test, predictions)
      rmse = np.sqrt(mean_squared_error(y_test, predictions))

      # 평가 결과 로깅
      mlflow.log_metric("MAE", mae)
      mlflow.log_metric("RMSE", rmse)

      # feature importance를 artifact로 plotting 하고 기록
      feature_importances = model.feature_importances_
      plt.figure(figsize=(10,6))
      plt.barh(X_train.columns, feature_importances)
      plt.title("Feature Importance")
      plt.savefig("feature_importance.png")

      # 아티팩트(특징 중요도) 기록
      mlflow.log_artifact("feature_importance.png")

      #model 자체 기록
      mlflow.sklearn.log_model(model, "model")

  print(f"Model training complete. MAE: {mae}, RMSE: {rmse}")
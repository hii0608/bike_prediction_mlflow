'''
  모델 저장 및 배포 코드
  run_name 없음
'''

from sys import version_info
import xgboost as xgb
from sklearn import datasets
from sklearn.model_selection import train_test_split

PYTHON_VERSION = "{major}.{minor}.{micro}".format(major=version_info.major, minor=version_info.minor, micro=version_info.micro)
iris = datasets.load_iris()
x = iris.data[:, 2:]
y = iris.target
x_train, x_test, y_train, _ = train_test_split(x, y, test_size = 0.2, random_state= 42)

dtrain = xgb.DMatrix(x_train, label=y_train)

#모델 학습 및 저장
xgb_model = xgb.train(params={"max_depth": 10}, dtrain=dtrain, num_boost_round=10)
xgb_model_path = "./ckpt/xgb_model.pth"
xgb_model.save_model(xgb_model_path)

artifacts = {
  "xgb_model": xgb_model_path
}

#모델 클래스 정의
import mlflow.pyfunc


# 실험 이름 및 Tracking URI 설정
mlflow.set_tracking_uri("http://localhost:5000")  # MLflow Tracking Server URL 설정
mlflow.set_experiment("XGBoost_Iris_Experiment")  # 실험 이름 설정


class XGBWrapper(mlflow.pyfunc.PythonModel):
  def load_context(self, context):
    import xgboost as xgb
    self.xgb_model = xgb.Booster()
    self.xgb_model.load_model(context.artifacts["xgb_model"])
    
  def predict(self, context, model_input):
    input_matrix = xgb.DMatrix(model_input.values)
    return self.xgb_model.predict(input_matrix)


#conda 환경 설정
import cloudpickle
conda_env = {
  'channels': ['defaults'],
  'dependencies': [
    'python={}'.format(PYTHON_VERSION),
    'pip',
    {
      'pip': [
        'mlflow',
        'xgboost=={}'.format(xgb.__version__),
        'cloudpickle=={}'.format(cloudpickle.__version__),
      ],
    },
  ],
  'name':'xgb_env'
}


#MLFlow에서 Model 저장
mlflow_pyfunc_model_path = "xgb_mlflow_pyfunc"
mlflow.pyfunc.save_model(
  path = mlflow_pyfunc_model_path,
  python_model = XGBWrapper(), artifacts=artifacts,
  conda_env = conda_env
)

#파이썬 함수 형태로 모델 로드 
loaded_model = mlflow.pyfunc.load_model(mlflow_pyfunc_model_path)

#모델 평가
import pandas as pd
test_predictions = loaded_model.predict(pd.DataFrame(x_test))
print(test_predictions)




'''
  아래 명령어로 서빙
  mlflow models serve -m xgb_mlflow_pyfunc -p 1234 --no-conda

  모델 서빙 확인을 위한 예시 명령어  
  curl --location --request POST 'localhost:1234/invocations' \
  --header 'Content-Type:application/json' \
  --data-raw '{
      "dataframe_split": {
          "columns" : ["petal length (cm)", "petal width (cm)"],
          "data" : [[3, 4]]
      }
  }'
'''
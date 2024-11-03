import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def preprocessing(file_path="../data/train.csv",test_size=0.2 ): 
  #data load
  data = pd.read_csv(file_path)

  # 누락된 값을 notebook에서 확인했을 때, 따로 없었음
  missing_values = data.isnull().sum() 

  # Cap windspeed at a maximum threshold if necessary
  data['windspeed'] = data['windspeed'].clip(upper=40)

  # 음수 또는 잘못된 값이 있는 행 제거
  data = data[data[ 'temp' ] >= 0 ]

  # 'datetime' 열을 datetime 객체로 변환
  data[ 'datetime' ] = pd.to_datetime(data[ 'datetime' ])

  # 유의미한 특성 생성
  data['hour'] = data['datetime'].dt.hour 
  data['day_of_week'] = data['datetime'].dt.dayofweek 
  data['month'] = data['datetime'].dt.month

  #  맑은날 또는 비오는 날로 이진 특성 생성
  data[ 'is_clear_weather' ] = (data[ 'weather' ] == 1 ).astype( int ) 
  data[ 'is_rainy_weather' ] = (data[ 'weather' ] >= 3 ).astype( int )

  # 휴일과 근무일에 대한 결합된 특성 생성
  data[ 'is_holiday_workingday' ] = ((data[ 'holiday' ] == 1 ) & (data[ 'workingday' ] == 1 )).astype( int )

  # 관련 없는 열 삭제
  data.drop(columns=[ "datetime" ], inplace= True )

  X = data.drop(columns=["count"]) # 독립변수(Feature)
  y = data["count"] # 종속변수(Target)

  # 80-20 train-test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

  # 분할 확인
  print(f"Training data size: {X_train.shape}")
  print(f"Testing data size: {X_test.shape}")
  
  return X_train, X_test, y_train, y_test
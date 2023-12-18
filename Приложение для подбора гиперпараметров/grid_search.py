import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import streamlit as st
from PIL import Image
import time

# функция для вычисления MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# изображение для фона
script_directory = os.path.dirname(os.path.abspath(__file__))
background_image_path = os.path.join(script_directory, "Фон приложение_1.png")
background_image = Image.open(background_image_path)
st.image(background_image, caption='', use_column_width=True)

# виджет для загрузки файла с данными
uploaded_file_data = st.file_uploader("Выберите файл с данными (CSV)", type=["csv"])

# проверка наличия загруженного файла с данными
if uploaded_file_data is not None:
    # Чтение файла с данными в DataFrame
    aplan = pd.read_csv(uploaded_file_data)
    aplan.columns = aplan.columns.str.strip('aggr_id_')
    aplan.rename(columns={aplan.columns[0]: 'target'}, inplace=True)

time.sleep(10)

# виджет для загрузки файла с маппингом названий фичей
id_features = st.file_uploader("Выберите файл с маппингом названий фичей (CSV)", type=["csv"])
    
# проверка наличия файла
if id_features is not None:
    id_features_df = pd.read_csv(id_features)
    
    # маппинг названий фичей
    for i in range(len(id_features_df)):
        col_id = id_features_df.iloc[i]['id']
        new_name = id_features_df.iloc[i]['name']
        aplan.rename(columns={str(col_id): new_name}, inplace=True)
    st.write("Маппинг фичей обновлен.")

time.sleep(10)

# формируем датафрейм
aplan = (aplan.rename(columns={aplan.columns[0]:'target', 
                               'Календарь: Номер часа':'час',
                               'Календарь: День месяца':'день',
                               'Календарь: Номер месяца':'месяц',
                               'Календарь: Агрегация номер года':'год', 
                               }))

# меняем тип данных
aplan['час'] = aplan['час'].astype(int)
aplan['день'] = aplan['день'].astype(int)
aplan['месяц'] = aplan['месяц'].astype(int)
aplan['год'] = aplan['год'].astype(int)
aplan['год'] = aplan['год'].astype(str)
aplan['месяц'] = aplan['месяц'].astype(str)
aplan['день'] = aplan['день'].astype(str)
aplan['час'] = aplan['час'].astype(str)

# добавляем 0 к значениям
aplan['месяц'] = aplan['месяц'].str.zfill(2)
aplan['день'] = aplan['день'].str.zfill(2)
aplan['час'] = aplan['час'].str.zfill(2)

# создание столбца с датой
aplan['date'] = pd.to_datetime(aplan['год'] + '-' + aplan['месяц'] + '-' + aplan['день'] + ' ' + aplan['час'])

# удаление ненужных символов из столбца 'date'
aplan['date'] = aplan['date'].dt.strftime('%Y-%m-%d %H:%m:%s')
aplan['date'] = pd.to_datetime(aplan['date'])

# возвращаем типы в числовые
aplan['час'] = aplan['час'].astype(int)
aplan['день'] = aplan['день'].astype(int)
aplan['месяц'] = aplan['месяц'].astype(int)
aplan['год'] = aplan['год'].astype(int)
aplan = aplan.query('date >= "2021-01-01"')

# дашборд 
fig = px.line(aplan, x='date', y='target', title='Распределение целевой переменной')
fig.update_traces(line_color='blue', line_dash='dot')
fig.update_xaxes(title_text='Дата', tickformat='%Y-%m-%d')
fig.update_yaxes(title_text='Целевая переменная')
st.plotly_chart(fig)

# добавление виджетов для выбора дат
st.markdown('<span style="font-size:20px; color: green;">Обучающая выборка</span>', unsafe_allow_html=True)
start_date_train = st.date_input('Выберите начальную дату (обучающая выборка)', pd.to_datetime('2021-01-01'), key='start_train')
end_date_train = st.date_input('Выберите конечную дату (обучающая выборка)', pd.to_datetime('2023-01-01'), key='end_train')
st.markdown('<span style="font-size:20px; color: green;">Тестовая выборка</span>', unsafe_allow_html=True)
start_date_test = st.date_input('Выберите начальную дату (тестовая выборка)', pd.to_datetime('2023-01-01'), key='start_test')
end_date_test = st.date_input('Выберите конечную дату (тестовая выборка)', pd.to_datetime('2023-02-01'), key='end_test')
start_date_train = pd.to_datetime(start_date_train)
end_date_train = pd.to_datetime(end_date_train)
start_date_test = pd.to_datetime(start_date_test)
end_date_test = pd.to_datetime(end_date_test)

train_df = aplan[(aplan['date'] >= start_date_train) & (aplan['date'] < end_date_train)]
test_df = aplan[(aplan['date'] >= start_date_test) & (aplan['date'] < end_date_test)]
train_df.drop('date', axis=1, inplace=True)
test_df.drop('date', axis=1, inplace=True)

# признаки и целевая переменная для обучающей и тестовой выборки
X_train = train_df.iloc[:, 1:]  
y_train = train_df['target']
X_test = test_df.iloc[:, 1:]  
y_test = test_df['target']

# класс модели с поиском гиперпараметров по сетке
class GridSearchModelTrainer:
    def __init__(self, model, param_grid):
        self.model = model
        self.param_grid = param_grid
        self.grid_search = GridSearchCV(self.model, self.param_grid, cv=5,
                                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=100)
    
    # метод обучения, прогноза и оценки
    def train_evaluate(self, X_train, X_test, y_train, y_test):
        self.grid_search.fit(X_train, y_train)
        best_params = self.grid_search.best_params_
        best_model = self.grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print("Лучшие параметры:", best_params)
        print('МЕТРИКИ МОДЕЛИ:')
        print('MAE:', round(mean_absolute_error(y_pred, y_test), 5))
        print('MAPE:', round(mean_absolute_percentage_error(y_pred, y_test), 5))
        print('MSE:', round(mean_squared_error(y_pred, y_test), 5))
        print('RMSE:', round(np.sqrt(mean_squared_error(y_pred, y_test))))
        print('R2:', round(r2_score(y_pred, y_test), 5))
        print('---------------------------------------------------------------------------')

# класс CatBoost с сеткой гиперпараметров
class CatBoostModelTrainerGridSearch(GridSearchModelTrainer):
    def __init__(self):
        model = Pipeline(steps=[('forecast', CatBoostRegressor(verbose=0))])
        param_grid = {'forecast__n_estimators': [100, 300, 500], 'forecast__learning_rate': [0.1, 0.01], 'forecast__depth': range(3, 10)}
        super().__init__(model, param_grid)

# класс GradientBoostingRegressor с сеткой гиперпараметров
class GradientBoostingModelTrainerGridSearch(GridSearchModelTrainer):
    def __init__(self):
        model = Pipeline(steps=[('forecast', GradientBoostingRegressor())])
        param_grid = {'forecast__n_estimators': [100, 300, 500], 'forecast__max_depth': range(3, 10)}
        super().__init__(model, param_grid)

# класс RandomForest с сеткой гиперпараметров
class RandomForestModelTrainerGridSearch(GridSearchModelTrainer):
    def __init__(self):
        model = Pipeline(steps=[('forecast', RandomForestRegressor())])
        param_grid = {'forecast__n_estimators': [100, 300, 500], 'forecast__max_depth': range(1, 10)}
        super().__init__(model, param_grid)        


# Модель CatBoost
if st.button("Подбор гиперпараметров к модели CatBoost"):
    st.write("\nМодель CatBoost:")
    catboost_trainer = CatBoostModelTrainerGridSearch()
    catboost_trainer.train_evaluate(X_train, X_test, y_train, y_test)

    st.write("Лучшие параметры:", catboost_trainer.grid_search.best_params_)
    st.write('МЕТРИКИ МОДЕЛИ:')
    st.write('MAE:', round(mean_absolute_error(catboost_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('MAPE:', round(mean_absolute_percentage_error(catboost_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('MSE:', round(mean_squared_error(catboost_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('RMSE:', round(np.sqrt(mean_squared_error(catboost_trainer.grid_search.best_estimator_.predict(X_test), y_test))))
    st.write('R2:', round(r2_score(catboost_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('---------------------------------------------------------------------------')

    # визуализация факт/прогноз лучшей модели - CatBoost
    y_pred_best_catboost = catboost_trainer.grid_search.best_estimator_.predict(X_test)  
    test_df = aplan[(aplan['date'] >= start_date_test) & (aplan['date'] < end_date_test)]
    test_df['y_pred_best_catboost'] = y_pred_best_catboost

    # дашборд 
    fig = px.line(test_df, x=test_df['date'], y=['target', 'y_pred_best_catboost'], 
              labels={'index': 'Дата', 'value': 'Целевая переменная'},
              title='Факт/прогноз лучшей модели',
              color_discrete_map={'target': 'green', 'y_pred_best_catboost': 'blue'})
    st.plotly_chart(fig)

    # R2 для каждого дня
    r2 = []
    for i in test_df['день'].unique():
        r2_value = r2_score(test_df[test_df['день'] == i]['target'],
                            test_df[test_df['день'] == i]['y_pred_best_catboost'])
        r2.append(round(r2_value, 4))

    # DataFrame для R2
    r2_df = pd.DataFrame({'R2': r2})
    r2_df.index += 1

    # функция для создания CSS классов в зависимости от значения R2
    def color_r2(val):
        if val >= 0.90:
            color = 'background-color: green; color: white'
        elif 0.9 > val >= 0.8:
            color = 'background-color: yellow; color: black'
        else:
            color = 'background-color: red; color: white'
        return color

    r2_styled = r2_df.style.applymap(lambda x: color_r2(x), subset=['R2'])
    st.write("### R2 на каждый день")
    st.write(r2_styled, unsafe_allow_html=True)
    

# Модель GradientBoostingRegressor
if st.button("Подбор гиперпараметров к модели GradientBoosting"):
    st.write("\nМодель GradientBoosting:")
    gradient_boosting_trainer = GradientBoostingModelTrainerGridSearch()
    gradient_boosting_trainer.train_evaluate(X_train, X_test, y_train, y_test)

    st.write("Лучшие параметры:", gradient_boosting_trainer.grid_search.best_params_)
    st.write('МЕТРИКИ МОДЕЛИ:')
    st.write('MAE:', round(mean_absolute_error(gradient_boosting_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('MAPE:', round(mean_absolute_percentage_error(gradient_boosting_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('MSE:', round(mean_squared_error(gradient_boosting_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('RMSE:', round(np.sqrt(mean_squared_error(gradient_boosting_trainer.grid_search.best_estimator_.predict(X_test), y_test))))
    st.write('R2:', round(r2_score(gradient_boosting_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('---------------------------------------------------------------------------')

    # визуализация факт/прогноз лучшей модели - CatBoost
    y_pred_best_gradientboosting = gradient_boosting_trainer.grid_search.best_estimator_.predict(X_test)  
    test_df = aplan[(aplan['date'] >= start_date_test) & (aplan['date'] < end_date_test)]
    test_df['y_pred_best_gradientboosting'] = y_pred_best_gradientboosting

    # дашборд 
    fig = px.line(test_df, x=test_df['date'], y=['target', 'y_pred_best_gradientboosting'], 
              labels={'index': 'Дата', 'value': 'Целевая переменная'},
              title='Факт/прогноз лучшей модели',
              color_discrete_map={'target': 'green', 'y_pred_best_gradientboosting': 'blue'})
    st.plotly_chart(fig)

    # R2 для каждого дня
    r2 = []
    for i in test_df['день'].unique():
        r2_value = r2_score(test_df[test_df['день'] == i]['target'],
                            test_df[test_df['день'] == i]['y_pred_best_gradientboosting'])
        r2.append(round(r2_value, 4))

    # DataFrame для R2
    r2_df = pd.DataFrame({'R2': r2})
    r2_df.index += 1

    # функция для создания CSS классов в зависимости от значения R2
    def color_r2(val):
        if val >= 0.90:
            color = 'background-color: green; color: white'
        elif 0.9 > val >= 0.8:
            color = 'background-color: yellow; color: black'
        else:
            color = 'background-color: red; color: white'
        return color

    r2_styled = r2_df.style.applymap(lambda x: color_r2(x), subset=['R2'])
    st.write("### R2 на каждый день")
    st.write(r2_styled, unsafe_allow_html=True)



# Модель RandomForest
if st.button("Подбор гиперпараметров к модели RandomForest"):
    st.write("\nМодель RandomForest:")
    random_forest_trainer = RandomForestModelTrainerGridSearch()
    random_forest_trainer.train_evaluate(X_train, X_test, y_train, y_test)

    st.write("Лучшие параметры:", random_forest_trainer.grid_search.best_params_)
    st.write('МЕТРИКИ МОДЕЛИ:')
    st.write('MAE:', round(mean_absolute_error(random_forest_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('MAPE:', round(mean_absolute_percentage_error(random_forest_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('MSE:', round(mean_squared_error(random_forest_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('RMSE:', round(np.sqrt(mean_squared_error(random_forest_trainer.grid_search.best_estimator_.predict(X_test), y_test))))
    st.write('R2:', round(r2_score(random_forest_trainer.grid_search.best_estimator_.predict(X_test), y_test), 5))
    st.write('---------------------------------------------------------------------------')

    # визуализация факт/прогноз лучшей модели - CatBoost
    y_pred_best_randomforest = random_forest_trainer.grid_search.best_estimator_.predict(X_test)  
    test_df = aplan[(aplan['date'] >= start_date_test) & (aplan['date'] < end_date_test)]
    test_df['y_pred_best_randomforest'] = y_pred_best_randomforest

    # дашборд 
    fig = px.line(test_df, x=test_df['date'], y=['target', 'y_pred_best_randomforest'], 
              labels={'index': 'Дата', 'value': 'Целевая переменная'},
              title='Факт/прогноз лучшей модели',
              color_discrete_map={'target': 'green', 'y_pred_best_randomforest': 'blue'})
    st.plotly_chart(fig)

    # R2 для каждого дня
    r2 = []
    for i in test_df['день'].unique():
        r2_value = r2_score(test_df[test_df['день'] == i]['target'],
                            test_df[test_df['день'] == i]['y_pred_best_randomforest'])
        r2.append(round(r2_value, 4))

    # DataFrame для R2
    r2_df = pd.DataFrame({'R2': r2})
    r2_df.index += 1

    # функция для создания CSS классов в зависимости от значения R2
    def color_r2(val):
        if val >= 0.90:
            color = 'background-color: green; color: white'
        elif 0.9 > val >= 0.8:
            color = 'background-color: yellow; color: black'
        else:
            color = 'background-color: red; color: white'
        return color

    r2_styled = r2_df.style.applymap(lambda x: color_r2(x), subset=['R2'])
    st.write("### R2 на каждый день")
    st.write(r2_styled, unsafe_allow_html=True)

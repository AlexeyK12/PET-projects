import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
import streamlit as st
from PIL import Image

# функция для вычисления MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# настройки заголовка и фона
dark_theme = """
    <style>
        body {
            background-color: #1e1e1e; /* Цвет тёмного фона */
            color: #ffffff; /* Цвет текста */
        }
    </style>
"""
st.markdown(dark_theme, unsafe_allow_html=True)

# загрузка данных
script_directory = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_directory, "train.csv")
train = pd.read_csv(data_path)

# извлечение признаков и целевой переменной
categorical_columns = train.select_dtypes(include='object').columns
X = train.drop(columns='SalePrice')
y = train['SalePrice']
X[categorical_columns] = X[categorical_columns].fillna('NA')

# разделение данных на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# определение предобработчиков
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, X_train.select_dtypes(include=[np.number]).columns),
    ('cat', categorical_transformer, categorical_columns)
])

# определение класса CatBoostModelTrainer
class CatBoostModelTrainer:
    # инициализация модели с указанными параметрами
    def __init__(self, n_estimators, learning_rate, depth, rsm, min_child_samples):
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('forecast', CatBoostRegressor(verbose=0, n_estimators=n_estimators, rsm=rsm,
                                                                     min_child_samples=min_child_samples, loss_function=loss_function,
                                                                     learning_rate=learning_rate, depth=depth, random_seed=42))])
    # метод обучения и расчёта метрик
    def train_evaluate(self, X_train, X_test, y_train, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        st.markdown("<h4 style='color: #2ecc71; text-align: center;'>МЕТРИКИ КАЧЕСТВА МОДЕЛИ:</h1>", unsafe_allow_html=True)
        st.markdown('**MAE:** {:.5f}'.format(mean_absolute_error(y_pred, y_test)))
        st.markdown('**MAPE:** {:.5f}'.format(mean_absolute_percentage_error(y_pred, y_test)))
        st.markdown('**MSE:** {:.5f}'.format(mean_squared_error(y_pred, y_test)))
        st.markdown('**RMSE:** {:.5f}'.format(np.sqrt(mean_squared_error(y_pred, y_test))))
        st.markdown('**R2:** {:.5f}'.format(r2_score(y_pred, y_test)))
        st.markdown('----------------------------------------------------------------')

# настройки заголовка и фона
st.markdown("""
    <h1 style='color: #2ecc71; text-align: center; padding: 20px; font-size: 36px; background-color: #0000FF;'>Модель прогнозирования цен на дома</h1>
""", unsafe_allow_html=True)

# изображение для фона
background_image_path = os.path.join(script_directory, "Prices-Soaring-Supplies-Vanishing-in-Popular-Migration-Destinations.jpg")
background_image = Image.open(background_image_path)
st.image(background_image, caption='', use_column_width=True)

# ввод гиперпараметров в сайдбаре
st.sidebar.markdown("""
    <span style="color: red; font-size: 24px; font-weight: bold;">Настройки гиперпараметров</span>
                    """, unsafe_allow_html=True)
loss_function  = st.sidebar.radio("Функция потерь", ('RMSE', 'MAE', 'Quantile'))
n_estimators = st.sidebar.number_input("Количество деревьев", min_value=1, max_value=1000, value=500, step=50)
learning_rate = st.sidebar.slider("Скорость обучения", min_value=0.01, max_value=0.1, value=0.1, step=0.01)
depth = st.sidebar.slider("Глубина дерева", min_value=1, max_value=10, value=5, step=1)
rsm  = st.sidebar.slider("Доля признаков в дереве", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
min_child_samples = st.sidebar.slider('Минимальное количество объектов узла', min_value=1, max_value=30, value=12, step=1)
max_leaves = st.sidebar.selectbox('Максимальное количество листьев в дереве', [1,2,3,4,5,6,7,8,9,10])

# экземпляр CatBoostModelTrainer с выбранными гиперпараметрами
catboost_trainer = CatBoostModelTrainer(n_estimators, learning_rate, depth, rsm, min_child_samples)

# обучение и оценка модели
catboost_trainer.train_evaluate(X_train, X_test, y_train, y_test)
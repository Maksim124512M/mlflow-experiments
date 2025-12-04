import mlflow

from data_processing import df_encoded

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from catboost import CatBoostRegressor

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment('First experiment')

X = df_encoded.drop(columns=['deposit_yes', 'deposit_no'])
y = df_encoded['deposit_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)

scaler = StandardScaler()
print(mlflow.get_experiment_by_name("First experiment"))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

grid_params = {
    'depth': [4, 6, 8, 10],
    'iterations': [500, 1000, 1500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'bagging_temperature': [0.0, 0.5, 1.0, 2.0],
}

model = CatBoostRegressor(verbose=0)

grid = GridSearchCV(estimator=model, param_grid=grid_params, scoring='neg_mean_absolute_error', n_jobs=-1)

with mlflow.start_run():
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)

    MAE = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    RMSE = root_mean_squared_error(y_true=y_test, y_pred=y_pred)
    R2_score = r2_score(y_true=y_test, y_pred=y_pred)

    print(f'MAE: {MAE}')
    print(f'RMSE: {RMSE}')
    print(f'R2: {R2_score}')

    mlflow_metrics = {
        'mae': MAE,
        'rmse': RMSE,
        'r2': R2_score,
    }

    mlflow.log_metrics(mlflow_metrics)
    mlflow.sklearn.log_model(grid.best_estimator_, 'model')
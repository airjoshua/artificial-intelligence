from fileinput import filename

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import randint
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.cluster import KMeans
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
)
from sklearn.ensemble import (
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    root_mean_squared_error,
)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import (
    Pipeline,
    make_pipeline,
)
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.tree import DecisionTreeRegressor

load_dotenv()


class Data:
    def __init__(
        self,
        filepath,
    ):
        self.data = self._split_data(self, filepath=filepath)
        self.preprocessing: ColumnTransformer = self._design_data_pipelines()

    @staticmethod
    def _load_data(filepath):
        return pd.read_csv(filepath_or_buffer=filepath)

    @staticmethod
    def _split_data(self, filepath):
        data = self._load_data(filepath)
        target_variable = "median_house_value"
        X = data.drop(columns=[target_variable]).copy()
        y = data[target_variable].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _build_num_pipeline():
        num_pipeline_steps = [
            ("impute_median", SimpleImputer(strategy="median")),
            ("standardize", StandardScaler()),
        ]
        return Pipeline(num_pipeline_steps)

    @staticmethod
    def _build_cat_pipeline() -> Pipeline:
        cat_pipeline_steps = [
            ("impute_most_frequent", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
        return Pipeline(cat_pipeline_steps)

    def _design_data_pipelines(self) -> ColumnTransformer:
        num_pipeline = self._build_num_pipeline()
        cat_pipeline = self._build_cat_pipeline()
        column_transformers_option_1 = [
            ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ]

        # column_transformers_option_2
        ratio_pipeline = make_pipeline(
            SimpleImputer(strategy="median", missing_values=np.nan),
            FunctionTransformer(
                func=lambda x: x[:, [0]] / x[:, [1]],
                feature_names_out=lambda function_transformer, feature_names_in: [
                    "ratio"
                ],
            ),
            StandardScaler(),
        )

        log_pipeline = make_pipeline(
            SimpleImputer(strategy="median", missing_values=np.nan),
            FunctionTransformer(
                func=np.log,
                feature_names_out="one-to-one",
            ),
            StandardScaler(),
        )
        default_num_pipeline = make_pipeline(
            SimpleImputer(strategy="median", missing_values=np.nan),
            StandardScaler(),
        )
        cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0)
        transformers = [
            (
                "bedrooms",
                ratio_pipeline,
                ["total_bedrooms", "total_rooms"],
            ),
            (
                "rooms_per_house",
                ratio_pipeline,
                ["total_rooms", "households"],
            ),
            (
                "people_per_house",
                ratio_pipeline,
                ["population", "households"],
            ),
            (
                "log",
                log_pipeline,
                [
                    "total_bedrooms",
                    "total_rooms",
                    "population",
                    "households",
                    "median_income",
                ],
            ),
            ("geo", cluster_simil, ["latitude", "longitude    "]),
            ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
        ]
        return ColumnTransformer(transformers=transformers)

    def feature_engineering(self): ...


class ModelTraining(Data):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.param_grid = self._get_param_grid()
        self.sklearn_pipeline: Pipeline = self._build_end_to_end_ml_pipeline()
        self.grid_search = self._grid_search()
        self.best_estimator = self.grid_search.best_estimator_
        self.results = self.grid_search.cv_results_

    def _grid_search(self):
        X_train, X_test, y_train, y_test = self.data
        params = list(ParameterGrid(self.param_grid))
        self.preprocessing
        self.sklearn_pipeline
        grid_search = GridSearchCV(
            estimator=self.sklearn_pipeline,
            param_grid=self.param_grid[1],
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        from sklearn import set_config

        params[0]
        self.param_grid
        self.sklearn_pipeline.named_steps
        for params in ParameterGrid(self.param_grid):
            estimator: Pipeline = self.sklearn_pipeline.set_params(**params)
            estimator.fit(X=X_train, y=y_train)
        grid_search.fit(X=X_train, y=y_train)

        return grid_search

        self.best_estimator = grid_search.best_estimator_
        results = pd.DataFrame(grid_search.cv_results_)
        filename = "my_california_housing_grid_search.pkl"
        joblib.dump(value=..., filename=filename)

    def _grid_search_2(self):
        X_train, X_test, y_train, y_test = self.data
        for params in ParameterGrid(self.param_grid):
            estimator_ = self.sklearn_pipeline.set_params(**params)
            estimator_.fit(X=X_train, y=y_train)

        return self

    def train_model(self):
        X_train, X_test, y_train, y_test = self.data
        lin_reg = make_pipeline(self.preprocessing, LinearRegression())
        lin_reg.fit(X=X_train, y=y_train)

        tree_reg = make_pipeline(
            self.preprocessing, DecisionTreeRegressor(random_state=42)
        )
        tree_reg.fit(X=X_train, y=y_train)

        forest_reg = make_pipeline(
            self.preprocessing, RandomForestRegressor(random_state=42)
        )
        forest_rmses = -cross_val_score(
            forest_reg,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=10,
        )
        forest_reg.fit(X=X_train, y=y_train)
        housing_predictions = forest_reg.predict(X_test)
        forest_rmse = root_mean_squared_error(y_true=y_test, y_pred=housing_predictions)

    def _build_end_to_end_ml_pipeline(self) -> Pipeline:
        preprocessing = self._design_data_pipelines()
        steps = [
            ("preprocessing", preprocessing),
            ("random_forest", RandomForestRegressor(random_state=42)),
        ]
        return Pipeline(steps=steps)

    @staticmethod
    def _get_param_grid() -> list[dict[str, list[int]]]:
        return [
            {
                "preprocessing__geo__n_clusters": [5, 8, 10],
                "random_forest__max_features": [4, 6, 8],
            },
            {
                "preprocessing__geo__n_clusters": [10, 15],
                "random_forest__max_features": [6, 8, 10],
            },
        ]

    def _randomized_grid_search(self):
        housing = ...
        housing_labels = ...
        steps = [
            ("preprocessing", self.preprocessing),
            ("random_forest", RandomForestRegressor(random_state=42)),
        ]
        full_pipeline = Pipeline(steps=steps)
        param_distribs = {
            "preprocessing__geo__n_clusters": randint(low=3, high=50),
            "random_forest__max_features": randint(low=2, high=20),
        }

        rnd_search = RandomizedSearchCV(
            full_pipeline,
            param_distributions=param_distribs,
            n_iter=10,
            cv=3,
            scoring="neg_root_mean_squared_error",
        )

        rnd_search.fit(housing, housing_labels)
        ...

    def cross_validation(self, data):

        tree_rmses = -cross_val_score(
            estimator=...,
            X=...,
            y=...,
            scoring="neg_root_mean_squared_error",
            cv=10,
        )
        ...


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state
        self.kmeans_ = KMeans(
            self.n_clusters, n_init=10, random_state=self.random_state
        )

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


if __name__ == "__main__":
    ModelTraining(filepath="./data/housing.csv")

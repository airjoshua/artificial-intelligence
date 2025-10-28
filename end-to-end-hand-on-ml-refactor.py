from dotenv import load_dotenv
from sklearn import datasets

import joblib
import numpy as np
import pandas as pd
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
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    # accuracy_score,
    root_mean_squared_error,
)
from sklearn.metrics.pairwise import rbf_kernel

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    ParameterGrid,
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
from sklearn.utils.validation import (
    check_array,
    check_is_fitted,
)

load_dotenv()


def split_data(data):
    X = data.drop(columns="median_house_value").copy()
    y = data["median_house_value"].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
    )
    return X_train, X_test, y_train, y_test


def design_data_pipelines():
    # column_transformers_option_1
    num_pipeline = build_num_pipeline()
    cat_pipeline = build_cat_pipeline()
    column_transformers_option_1 = [
        ("num", num_pipeline, make_column_selector(dtype_include=np.number)),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ]

    # column_transformers_option_2
    ratio_pipeline = make_pipeline(
        SimpleImputer(strategy="median", missing_values=np.nan),
        FunctionTransformer(
            func=lambda x: x[:, [0]] / x[:, [1]],
            feature_names_out=lambda function_transformer, feature_names_in: ["ratio"],
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
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ]
    return ColumnTransformer(
        transformers=transformers,
        remainder=default_num_pipeline,  # one column remaining: housing_median_age
    )


def build_cat_pipeline() -> Pipeline:
    cat_pipeline_steps = [
        ("impute_most_frequent", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
    ]
    return Pipeline(cat_pipeline_steps)


def build_num_pipeline() -> Pipeline:
    num_pipeline_steps = [
        ("impute_median", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ]
    return Pipeline(num_pipeline_steps)


class StandardScalerClone(BaseEstimator, TransformerMixin):
    def __init__(self, with_mean=True):  # no *args or **kwargs!
        self.with_mean = with_mean

    def fit(self, X, y=None):  # y is required even though we don't use it
        X = check_array(X)  # checks that X is an array with finite float values
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
        return self  # always return self!

    def transform(self, X):
        check_is_fitted(self)  # looks for learned attributes (with trailing _)
        X = check_array(X)
        if self.with_mean:
            X = X - self.mean_
        return X / self.scale_


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(
            self.n_clusters, n_init=10, random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)

    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]


def feature_engineering(data): ...


def train_model(data):
    X_train, X_test, y_train, y_test = data
    preprocessing = design_data_pipelines()
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(X=X_train, y=y_train)

    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(X=X_train, y=y_train)

    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
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


def grid_search(data):
    X_train, X_test, y_train, y_test = data
    full_pipeline, param_grid = build_full_pipeline()

    params = list(ParameterGrid(param_grid))
    grid_search = GridSearchCV(
        estimator=full_pipeline,
        param_grid=param_grid[1],
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X=X_train, y=y_train)
    results = pd.DataFrame(grid_search.cv_results_)
    filename = "my_california_housing_grid_search.pkl"
    joblib.dump(value=..., filename=filename)


def build_full_pipeline():
    preprocessing = design_data_pipelines()
    steps = [
        ("preprocessing", preprocessing),
        ("random_forest", RandomForestRegressor(random_state=42)),
    ]
    full_pipeline = Pipeline(steps=steps)
    param_grid = [
        {
            "preprocessing__geo__n_clusters": [5, 8, 10],
            "random_forest__max_features": [4, 6, 8],
        },
        {
            "preprocessing__geo__n_clusters": [10, 15],
            "random_forest__max_features": [6, 8, 10],
        },
    ]
    return full_pipeline, param_grid


def randomized_grid_search():
    preprocessing = ...
    housing = ...
    housing_labels = ...
    steps = [
        ("preprocessing", preprocessing),
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


def cross_validation(data):

    tree_rmses = -cross_val_score(
        estimator=...,
        X=...,
        y=...,
        scoring="neg_root_mean_squared_error",
        cv=10,
    )
    ...


if __name__ == "__main__":
    df = pd.read_csv("./data/housing.csv")
    data_ = split_data(df)
    grid_search(data_)
    _ = datasets("load_diabetes")

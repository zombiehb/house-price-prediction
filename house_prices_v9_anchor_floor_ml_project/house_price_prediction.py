#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import boxcox1p
from scipy.stats import skew

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso, ElasticNet, Ridge, BayesianRidge, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False


RANDOM_STATE = 2026


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def safe_mode(series: pd.Series, default="None") -> str:
    mode = series.mode(dropna=True)
    return str(mode.iloc[0]) if len(mode) else default


def fill_missing(all_data: pd.DataFrame) -> pd.DataFrame:
    all_data = all_data.copy()

    none_cols = [
        "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu",
        "GarageType", "GarageFinish", "GarageQual", "GarageCond",
        "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
        "BsmtFinType2", "MasVnrType"
    ]
    for col in none_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna("None")

    zero_cols = [
        "GarageYrBlt", "GarageArea", "GarageCars",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "BsmtFullBath", "BsmtHalfBath", "MasVnrArea"
    ]
    for col in zero_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(0)

    if "LotFrontage" in all_data.columns and "Neighborhood" in all_data.columns:
        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
            lambda s: s.fillna(s.median())
        )
        all_data["LotFrontage"] = all_data["LotFrontage"].fillna(all_data["LotFrontage"].median())

    mode_cols = [
        "MSZoning", "Electrical", "KitchenQual", "Exterior1st",
        "Exterior2nd", "SaleType", "Functional"
    ]
    for col in mode_cols:
        if col in all_data.columns:
            all_data[col] = all_data[col].fillna(safe_mode(all_data[col], "None"))

    for col in all_data.select_dtypes(include=["object"]).columns:
        all_data[col] = all_data[col].fillna(safe_mode(all_data[col], "None"))

    for col in all_data.select_dtypes(exclude=["object"]).columns:
        all_data[col] = all_data[col].fillna(all_data[col].median())

    return all_data


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    qual_map = {"None": 0, "NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    exposure_map = {"None": 0, "NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
    finish_map = {"None": 0, "NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}
    functional_map = {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7}

    quality_cols = [
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"
    ]
    for col in quality_cols:
        if col in df.columns:
            df[col + "_Score"] = df[col].map(qual_map).fillna(0).astype(float)

    if "BsmtExposure" in df.columns:
        df["BsmtExposure_Score"] = df["BsmtExposure"].map(exposure_map).fillna(0).astype(float)

    for col in ["BsmtFinType1", "BsmtFinType2"]:
        if col in df.columns:
            df[col + "_Score"] = df[col].map(finish_map).fillna(0).astype(float)

    if "Functional" in df.columns:
        df["Functional_Score"] = df["Functional"].map(functional_map).fillna(7).astype(float)

    total_bsmt = safe_numeric(df, "TotalBsmtSF")
    first = safe_numeric(df, "1stFlrSF")
    second = safe_numeric(df, "2ndFlrSF")
    grliv = safe_numeric(df, "GrLivArea")
    lotarea = safe_numeric(df, "LotArea")
    overall_qual = safe_numeric(df, "OverallQual")
    overall_cond = safe_numeric(df, "OverallCond")
    yr_sold = safe_numeric(df, "YrSold")
    year_built = safe_numeric(df, "YearBuilt")
    year_remod = safe_numeric(df, "YearRemodAdd")
    garage_yr = safe_numeric(df, "GarageYrBlt")
    garage_area = safe_numeric(df, "GarageArea")
    garage_cars = safe_numeric(df, "GarageCars")
    total_rooms = safe_numeric(df, "TotRmsAbvGrd")
    bedrooms = safe_numeric(df, "BedroomAbvGr")
    fireplaces = safe_numeric(df, "Fireplaces")

    df["TotalSF"] = total_bsmt + first + second
    df["TotalFlrSF"] = first + second
    df["TotalFinishedSF"] = first + second + safe_numeric(df, "BsmtFinSF1") + safe_numeric(df, "BsmtFinSF2")
    df["TotalBsmtFinSF"] = safe_numeric(df, "BsmtFinSF1") + safe_numeric(df, "BsmtFinSF2")
    df["TotalPorchSF"] = (
        safe_numeric(df, "OpenPorchSF")
        + safe_numeric(df, "EnclosedPorch")
        + safe_numeric(df, "3SsnPorch")
        + safe_numeric(df, "ScreenPorch")
        + safe_numeric(df, "WoodDeckSF")
    )

    df["TotalBath"] = (
        safe_numeric(df, "FullBath")
        + 0.5 * safe_numeric(df, "HalfBath")
        + safe_numeric(df, "BsmtFullBath")
        + 0.5 * safe_numeric(df, "BsmtHalfBath")
    )

    df["HouseAge"] = (yr_sold - year_built).clip(lower=0)
    df["RemodAge"] = (yr_sold - year_remod).clip(lower=0)
    df["GarageAge"] = (yr_sold - garage_yr).clip(lower=0)
    df["IsRemodeled"] = (year_built != year_remod).astype(int)
    df["IsNewHouse"] = (year_built == yr_sold).astype(int)

    df["OverallGrade"] = overall_qual * overall_cond
    df["Qual_x_TotalSF"] = overall_qual * df["TotalSF"]
    df["Qual_x_GrLivArea"] = overall_qual * grliv
    df["Qual_x_TotalFinishedSF"] = overall_qual * df["TotalFinishedSF"]

    df["GarageScore"] = garage_area * garage_cars
    df["BsmtScore"] = total_bsmt * (safe_numeric(df, "BsmtQual_Score") + 1)
    df["KitchenScore"] = safe_numeric(df, "KitchenAbvGr") * (safe_numeric(df, "KitchenQual_Score") + 1)
    df["FireplaceScore"] = fireplaces * (safe_numeric(df, "FireplaceQu_Score") + 1)

    df["RoomSizeAvg"] = grliv / (total_rooms + 1)
    df["BathPerBedroom"] = df["TotalBath"] / (bedrooms + 1)
    df["GarageAreaRatio"] = garage_area / (df["TotalSF"] + 1)
    df["BsmtRatio"] = total_bsmt / (df["TotalSF"] + 1)
    df["PorchRatio"] = df["TotalPorchSF"] / (df["TotalSF"] + 1)
    df["LotAreaPerRoom"] = lotarea / (total_rooms + 1)

    df["HasGarage"] = (garage_area > 0).astype(int)
    df["HasBsmt"] = (total_bsmt > 0).astype(int)
    df["HasFireplace"] = (fireplaces > 0).astype(int)
    df["Has2ndFloor"] = (second > 0).astype(int)
    df["HasPool"] = (safe_numeric(df, "PoolArea") > 0).astype(int)
    df["HasMasVnr"] = (safe_numeric(df, "MasVnrArea") > 0).astype(int)

    if "Neighborhood" in df.columns:
        for col in ["TotalSF", "GrLivArea", "OverallQual", "HouseAge"]:
            grouped = df.groupby("Neighborhood")[col]
            df[f"Neighborhood_{col}_Median"] = df["Neighborhood"].map(grouped.median())
            df[f"{col}_Minus_NeighMedian"] = df[col] - df[f"Neighborhood_{col}_Median"]

    ratio_cols = [
        "RoomSizeAvg", "BathPerBedroom", "GarageAreaRatio",
        "BsmtRatio", "PorchRatio", "LotAreaPerRoom"
    ]
    for col in ratio_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return df


def preprocess(
    train: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train = train.copy()
    test = test.copy()
    test_id = test["Id"].copy()

    train = train.drop(train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)].index)

    y = np.log1p(train["SalePrice"])
    train_x = train.drop(columns=["SalePrice"])

    all_data = pd.concat([train_x, test], axis=0, ignore_index=True)
    all_data = all_data.drop(columns=["Id"])

    all_data = fill_missing(all_data)
    all_data = add_features(all_data)

    for col in ["MSSubClass", "OverallCond", "YrSold", "MoSold"]:
        if col in all_data.columns:
            all_data[col] = all_data[col].astype(str)

    if "Utilities" in all_data.columns:
        all_data = all_data.drop(columns=["Utilities"])

    cat_cols = all_data.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        values = all_data[col].astype(str)
        freq = values.value_counts(normalize=True)
        all_data[f"{col}_Freq"] = values.map(freq).astype(float)

    numeric_cols = all_data.select_dtypes(exclude=["object", "category"]).columns
    skewness = all_data[numeric_cols].apply(lambda s: skew(s.dropna()))
    skewed_cols = skewness[abs(skewness) > 0.75].index

    lam = 0.15
    for col in skewed_cols:
        min_val = all_data[col].min()
        if min_val <= -1:
            all_data[col] = all_data[col] - min_val
        all_data[col] = boxcox1p(all_data[col], lam)

    all_data = pd.get_dummies(all_data, drop_first=False)
    all_data = all_data.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = all_data.iloc[:len(y)].copy()
    X_test = all_data.iloc[len(y):].copy()

    return X, y, X_test, test_id


def make_models(fast: bool = False) -> Dict[str, BaseEstimator]:
    n_boost = 600 if fast else 2600
    n_tree = 400 if fast else 1000

    models: Dict[str, BaseEstimator] = {}

    models["lasso"] = make_pipeline(
        RobustScaler(),
        Lasso(alpha=0.00045, max_iter=80000, random_state=RANDOM_STATE)
    )
    models["elasticnet"] = make_pipeline(
        RobustScaler(),
        ElasticNet(alpha=0.00045, l1_ratio=0.88, max_iter=80000, random_state=RANDOM_STATE)
    )
    models["ridge"] = make_pipeline(
        RobustScaler(),
        Ridge(alpha=10.0, random_state=RANDOM_STATE)
    )
    models["bayesian_ridge"] = make_pipeline(
        RobustScaler(),
        BayesianRidge()
    )
    models["kernel_ridge"] = make_pipeline(
        RobustScaler(),
        KernelRidge(alpha=0.55, kernel="polynomial", degree=2, coef0=2.5)
    )
    models["svr"] = make_pipeline(
        RobustScaler(),
        SVR(C=18.0, epsilon=0.008, gamma=0.00035)
    )

    models["gbr"] = GradientBoostingRegressor(
        n_estimators=n_boost,
        learning_rate=0.018,
        max_depth=4,
        max_features="sqrt",
        min_samples_leaf=15,
        min_samples_split=10,
        loss="huber",
        random_state=RANDOM_STATE,
    )

    models["extra_trees"] = ExtraTreesRegressor(
        n_estimators=n_tree,
        max_features=0.65,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    models["random_forest"] = RandomForestRegressor(
        n_estimators=n_tree,
        max_features=0.55,
        min_samples_split=4,
        min_samples_leaf=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=n_boost,
            learning_rate=0.018,
            max_depth=3,
            min_child_weight=2,
            gamma=0.0,
            subsample=0.76,
            colsample_bytree=0.72,
            reg_alpha=0.0005,
            reg_lambda=1.15,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if HAS_LGBM:
        models["lgbm"] = LGBMRegressor(
            objective="regression",
            n_estimators=n_boost,
            learning_rate=0.018,
            num_leaves=5,
            max_depth=-1,
            min_child_samples=8,
            subsample=0.72,
            subsample_freq=1,
            colsample_bytree=0.56,
            reg_alpha=0.0005,
            reg_lambda=0.03,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    if HAS_CATBOOST:
        models["catboost"] = CatBoostRegressor(
            iterations=n_boost,
            learning_rate=0.018,
            depth=4,
            l2_leaf_reg=4.0,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )

    return models


def get_oof(
    models: Dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    folds: int
):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    names = list(models.keys())

    oof = np.zeros((len(X), len(names)))
    test_pred = np.zeros((len(X_test), len(names)))
    cv_rows = []

    for j, name in enumerate(names):
        print(f"\n{name}")
        fold_test = np.zeros((len(X_test), folds))
        scores = []

        for fold, (tr, va) in enumerate(kf.split(X, y), 1):
            est = clone(models[name])

            try:
                est.fit(X.iloc[tr], y.iloc[tr])

                pred_valid = np.asarray(est.predict(X.iloc[va]), dtype=float)
                pred_test = np.asarray(est.predict(X_test), dtype=float)

                if not np.isfinite(pred_valid).all() or not np.isfinite(pred_test).all():
                    raise ValueError("prediction contains non-finite values")

                oof[va, j] = pred_valid
                fold_test[:, fold - 1] = pred_test

                score = rmse(y.iloc[va], pred_valid)
                scores.append(score)
                print(f"fold {fold}: {score:.5f}")

            except Exception as err:
                print(f"fold {fold} skipped: {err}")
                fallback = float(y.iloc[tr].mean())
                oof[va, j] = fallback
                fold_test[:, fold - 1] = fallback
                scores.append(rmse(y.iloc[va], np.full(len(va), fallback)))

        test_pred[:, j] = fold_test.mean(axis=1)
        cv_rows.append({
            "model": name,
            "cv_rmse_log": float(np.mean(scores)),
            "cv_std": float(np.std(scores))
        })

        print(f"{name}: {np.mean(scores):.5f} ± {np.std(scores):.5f}")

    cv_df = pd.DataFrame(cv_rows).sort_values("cv_rmse_log")
    return oof, test_pred, cv_df, names


def optimize_weights(oof: np.ndarray, y: pd.Series) -> np.ndarray:
    n_models = oof.shape[1]
    x0 = np.ones(n_models) / n_models

    def objective(weights):
        return rmse(y, oof @ weights)

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0, 1)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1},
        options={"maxiter": 1000},
    )

    if result.success:
        return result.x

    return x0


def final_predictions(oof, test_pred, y, names, folds):
    kf = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)

    meta_models = [
        ("meta_ridge", RidgeCV(alphas=np.logspace(-3, 3, 80))),
        (
            "meta_elasticnet",
            ElasticNetCV(
                l1_ratio=[0.2, 0.5, 0.8],
                alphas=np.logspace(-4, -1, 40),
                cv=5,
                max_iter=50000,
                random_state=RANDOM_STATE,
            )
        ),
    ]

    meta_test_all = []

    for meta_name, meta in meta_models:
        print(f"\n{meta_name}")

        meta_oof = np.zeros(len(y))
        meta_test = np.zeros((test_pred.shape[0], folds))

        for fold, (tr, va) in enumerate(kf.split(oof, y), 1):
            est = clone(meta)
            est.fit(oof[tr], y.iloc[tr])

            meta_oof[va] = est.predict(oof[va])
            meta_test[:, fold - 1] = est.predict(test_pred)

            print(f"fold {fold}: {rmse(y.iloc[va], meta_oof[va]):.5f}")

        print(f"{meta_name} OOF: {rmse(y, meta_oof):.5f}")
        meta_test_all.append(meta_test.mean(axis=1))

    weights = optimize_weights(oof, y)
    blend_test = test_pred @ weights
    blend_oof = oof @ weights

    print(f"\nBase blend OOF: {rmse(y, blend_oof):.5f}")
    print("\nModel weights:")
    for model_name, weight in sorted(zip(names, weights), key=lambda item: -item[1]):
        print(f"{model_name:15s}: {weight:.5f}")

    meta_avg = np.mean(np.column_stack(meta_test_all), axis=1)
    model_only_log = 0.35 * meta_avg + 0.65 * blend_test

    return model_only_log, weights


def read_anchor(anchor_path: str | None, test_id: pd.Series) -> pd.DataFrame | None:
    if not anchor_path:
        return None

    anchor = pd.read_csv(anchor_path)

    if list(anchor.columns) != ["Id", "SalePrice"]:
        raise ValueError("anchor CSV must contain columns: Id, SalePrice")

    if len(anchor) != len(test_id):
        raise ValueError("anchor CSV row count does not match test.csv")

    if not anchor["Id"].equals(test_id.reset_index(drop=True)):
        raise ValueError("anchor CSV Id order does not match test.csv")

    return anchor


def save_submission(path: str, test_id: pd.Series, price: np.ndarray) -> None:
    sub = pd.DataFrame({
        "Id": test_id,
        "SalePrice": price
    })

    if len(sub) != 1459:
        raise ValueError(f"invalid submission row count: {len(sub)}")

    if sub["SalePrice"].isna().any() or not np.isfinite(sub["SalePrice"]).all():
        raise ValueError("SalePrice contains invalid values")

    sub.to_csv(path, index=False)
    print(f"saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv")
    parser.add_argument("--test", default="test.csv")
    parser.add_argument("--sample", default="sample_submission.csv")
    parser.add_argument("--anchor", default=None)
    parser.add_argument("--prefix", default="house_prices_v9")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fast", action="store_true")

    args = parser.parse_args()

    print("loading data")
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)

    print("preprocessing data")
    X, y, X_test, test_id = preprocess(train, test)
    print(f"train shape: {X.shape}")
    print(f"test shape : {X_test.shape}")

    print("training models")
    models = make_models(fast=args.fast)
    print("models:", ", ".join(models.keys()))

    oof, test_pred, cv_df, names = get_oof(models, X, y, X_test, folds=args.folds)
    model_log_pred, weights = final_predictions(oof, test_pred, y, names, folds=args.folds)

    model_price = np.clip(np.expm1(model_log_pred), 30000, 950000)
    model_path = f"{args.prefix}_model_only.csv"
    save_submission(model_path, test_id, model_price)

    cv_df.to_csv(f"{args.prefix}_cv_report.csv", index=False)
    pd.DataFrame({
        "model": names,
        "weight": weights
    }).to_csv(f"{args.prefix}_weights.csv", index=False)

    anchor = read_anchor(args.anchor, test_id)

    if anchor is not None:
        anchor_price = anchor["SalePrice"].values.astype(float)

        floor_path = f"{args.prefix}_anchor_floor.csv"
        save_submission(floor_path, test_id, anchor_price)

        log_anchor = np.log1p(anchor_price)
        log_model = np.log1p(model_price)

        for alpha in [0.01, 0.02, 0.03, 0.05]:
            blended = np.expm1((1 - alpha) * log_anchor + alpha * log_model)
            blended = np.clip(blended, 30000, 950000)
            save_submission(
                f"{args.prefix}_conservative_blend_{int(alpha * 100):02d}.csv",
                test_id,
                blended
            )

        print(f"anchor output: {floor_path}")
        print("blend outputs: 01, 02, 03, 05")
    else:
        print("anchor file not provided")


if __name__ == "__main__":
    main()
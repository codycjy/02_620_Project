import xgboost as xgb
from xgboost_self import XGBoost
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import numpy as np
import time


def compare_implementations(X, y, problem_type="regression", dataset_name=""):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 参数设置
    base_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_samples_split": 2,
        "lambda_reg": 1.0,
    }

    print(f"\n=== {dataset_name} ===")
    print("XGBoost实现:")
    start_time = time.time()

    if problem_type == "classification":
        if len(np.unique(y)) == 2:
            custom_model = XGBoost(**base_params, task="binary")
        else:
            custom_model = XGBoost(**base_params, task="multi")
    else:
        custom_model = XGBoost(**base_params, task="reg")

    # 对于分类问题，确保标签是整数类型
    if problem_type == "classification":
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    custom_model.fit(X_train, y_train)
    custom_train_time = time.time() - start_time

    start_time = time.time()
    custom_train_pred = custom_model.predict(X_train)
    custom_test_pred = custom_model.predict(X_test)
    custom_predict_time = time.time() - start_time

    print("\n官方XGBoost实现:")
    if problem_type == "regression":
        official_model = xgb.XGBRegressor(**base_params)
    else:
        official_model = xgb.XGBClassifier(**base_params)

    start_time = time.time()
    official_model.fit(X_train, y_train)
    official_train_time = time.time() - start_time

    start_time = time.time()
    official_train_pred = official_model.predict(X_train)
    official_test_pred = official_model.predict(X_test)
    official_predict_time = time.time() - start_time

    # 评估和对比
    if problem_type == "regression":
        metrics = {
            "训练集MSE": {
                "实现": mean_squared_error(y_train, custom_train_pred),
                "官方实现": mean_squared_error(y_train, official_train_pred),
            },
            "测试集MSE": {
                "实现": mean_squared_error(y_test, custom_test_pred),
                "官方实现": mean_squared_error(y_test, official_test_pred),
            },
        }
    else:
        metrics = {
            "训练集准确率": {
                "实现": accuracy_score(y_train, custom_train_pred),
                "官方实现": accuracy_score(y_train, official_train_pred),
            },
            "测试集准确率": {
                "实现": accuracy_score(y_test, custom_test_pred),
                "官方实现": accuracy_score(y_test, official_test_pred),
            },
        }

        # 只对二分类问题计算AUC
        if len(np.unique(y)) == 2:
            custom_train_proba = custom_model.predict_proba(X_train)[:, 1]
            custom_test_proba = custom_model.predict_proba(X_test)[:, 1]
            official_train_proba = official_model.predict_proba(X_train)[:, 1]
            official_test_proba = official_model.predict_proba(X_test)[:, 1]

            metrics.update(
                {
                    "训练集AUC": {
                        "实现": roc_auc_score(y_train, custom_train_proba),
                        "官方实现": roc_auc_score(y_train, official_train_proba),
                    },
                    "测试集AUC": {
                        "实现": roc_auc_score(y_test, custom_test_proba),
                        "官方实现": roc_auc_score(y_test, official_test_proba),
                    },
                }
            )

    # 打印性能指标
    for metric_name, values in metrics.items():
        print(f"\n{metric_name}:")
        for impl_name, value in values.items():
            print(f"{impl_name}: {value:.4f}")

    # 打印时间指标
    print("\n训练时间 (秒):")
    print(f"实现: {custom_train_time:.4f}")
    print(f"官方实现: {official_train_time:.4f}")

    print("\n预测时间 (秒):")
    print(f"实现: {custom_predict_time:.4f}")
    print(f"官方实现: {official_predict_time:.4f}")

    # 特征重要性对比
    if hasattr(custom_model, "feature_importance") and hasattr(
        official_model, "feature_importances_"
    ):
        print("\n特征重要性对比:")
        custom_importance = custom_model.feature_importance()
        official_importance = official_model.feature_importances_

        print("\n特征\t实现\t官方实现\t差异")
        print("-" * 50)
        for i in range(len(custom_importance)):
            diff = abs(custom_importance[i] - official_importance[i])
            print(
                f"{i}\t{custom_importance[i]:.4f}\t{official_importance[i]:.4f}\t{diff:.4f}"
            )


# 测试
if __name__ == "__main__":
    # 1. 糖尿病数据集测试
    print("\n=== 糖尿病数据集(回归问题) ===")
    X, y = load_diabetes(return_X_y=True)
    compare_implementations(X, y, "regression", "糖尿病数据集(回归问题)")

    # 2. 鸢尾花数据集测试
    print("\n=== 鸢尾花数据集(多分类问题) ===")
    X, y = load_iris(return_X_y=True)
    compare_implementations(X, y, "classification", "鸢尾花数据集(多分类问题)")

    # 3. 乳腺癌数据集测试
    print("\n=== 乳腺癌数据集(二分类问题) ===")
    X, y = load_breast_cancer(return_X_y=True)
    compare_implementations(X, y, "classification", "乳腺癌数据集(二分类问题)")

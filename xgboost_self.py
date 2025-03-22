import numpy as np


class Node:
    def __init__(self, value=None):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = value
        self.gain = None


class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2, lambda_reg=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg
        self.root = None
        self.n_features = None

    def fit(self, X, g, h):
        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, g, h)

    def _calculate_leaf_value(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.lambda_reg)

    def _calculate_gain(self, g_left, h_left, g_right, h_right, g, h):
        gain = 0.5 * (
            np.sum(g_left) ** 2 / (np.sum(h_left) + self.lambda_reg)
            + np.sum(g_right) ** 2 / (np.sum(h_right) + self.lambda_reg)
            - np.sum(g) ** 2 / (np.sum(h) + self.lambda_reg)
        )
        return gain

    def _find_best_split(self, X, g, h):
        best_gain = -np.inf
        best_feature, best_threshold = None, None

        for feature_idx in range(self.n_features):
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = ~left_idxs

                if np.sum(left_idxs) == 0 or np.sum(right_idxs) == 0:
                    continue

                g_left, h_left = g[left_idxs], h[left_idxs]
                g_right, h_right = g[right_idxs], h[right_idxs]

                gain = self._calculate_gain(g_left, h_left, g_right, h_right, g, h)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _grow_tree(self, X, g, h, depth=0):
        n_samples = X.shape[0]

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.all(g == g[0])
        ):
            return Node(value=self._calculate_leaf_value(g, h))

        feature_idx, threshold, gain = self._find_best_split(X, g, h)

        if feature_idx is None or gain <= 0:
            return Node(value=self._calculate_leaf_value(g, h))

        node = Node()
        node.feature_index = feature_idx
        node.threshold = threshold
        node.gain = gain

        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs

        node.left = self._grow_tree(X[left_idxs], g[left_idxs], h[left_idxs], depth + 1)
        node.right = self._grow_tree(
            X[right_idxs], g[right_idxs], h[right_idxs], depth + 1
        )

        return node

    def predict(self, X):
        return np.array([self._predict_sample(x) for x in X])

    def _predict_sample(self, x):
        node = self.root
        while node.left:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


class XGBoost:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        lambda_reg=1.0,
        task="reg",
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.lambda_reg = lambda_reg
        self.task = task  # 'reg' for regression, 'binary' for binary classification, 'multi' for multi-classification
        self.trees = []
        self.base_prediction = None
        self.n_classes = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _calculate_gradients(self, y_true, y_pred):
        if self.task == "reg":
            # 均方误差损失
            g = y_pred - y_true
            h = np.ones_like(y_true)
        elif self.task == "binary":
            # 对数损失
            pred_prob = self._sigmoid(y_pred)
            g = pred_prob - y_true
            h = pred_prob * (1 - pred_prob)
        else:  # 多分类
            # 计算softmax概率
            pred_prob = self._softmax(y_pred)
            # 梯度: p_i - y_i
            g = pred_prob - y_true
            # 二阶导数: p_i * (1 - p_i)
            h = pred_prob * (1 - pred_prob)
        return g, h

    def fit(self, X, y):
        if self.task == "reg":
            self.base_prediction = np.mean(y)
            y_pred = np.full_like(y, self.base_prediction, dtype=np.float64)

            for _ in range(self.n_estimators):
                g, h = self._calculate_gradients(y, y_pred)

                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    lambda_reg=self.lambda_reg,
                )
                tree.fit(X, g, h)
                self.trees.append(tree)

                update = tree.predict(X)
                y_pred += self.learning_rate * update

        elif self.task == "binary":
            self.base_prediction = 0
            y = y.astype(float)
            y_pred = np.full_like(y, self.base_prediction, dtype=np.float64)

            for _ in range(self.n_estimators):
                g, h = self._calculate_gradients(y, y_pred)

                tree = DecisionTree(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    lambda_reg=self.lambda_reg,
                )
                tree.fit(X, g, h)
                self.trees.append(tree)

                update = tree.predict(X)
                y_pred += self.learning_rate * update

        else:  # 多分类
            # 确定类别数
            if len(y.shape) == 1:
                # 如果y是标签形式，转为one-hot编码
                self.n_classes = len(np.unique(y))
                y_one_hot = np.zeros((y.shape[0], self.n_classes))
                for i in range(len(y)):
                    y_one_hot[i, y[i]] = 1
                y = y_one_hot
            else:
                self.n_classes = y.shape[1]

            # 为每个类别创建森林
            self.trees = [[] for _ in range(self.n_classes)]
            self.base_prediction = np.zeros(self.n_classes)

            # 初始预测值
            y_pred = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)

            for _ in range(self.n_estimators):
                # 计算所有类别的梯度
                g, h = self._calculate_gradients(y, y_pred)

                # 为每个类别训练一棵树
                for k in range(self.n_classes):
                    tree = DecisionTree(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        lambda_reg=self.lambda_reg,
                    )
                    tree.fit(X, g[:, k], h[:, k])
                    self.trees[k].append(tree)

                    # 更新预测值
                    update = tree.predict(X)
                    y_pred[:, k] += self.learning_rate * update

    def predict(self, X):
        if self.task == "reg" or self.task == "binary":
            y_pred = np.full(X.shape[0], self.base_prediction, dtype=np.float64)
            for tree in self.trees:
                update = tree.predict(X)
                y_pred += self.learning_rate * update

            if self.task == "binary":
                return (self._sigmoid(y_pred) >= 0.5).astype(int)
            return y_pred

        else:  # 多分类
            # 初始化预测矩阵: 样本数 x 类别数
            y_pred = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)

            # 对每个类别的每棵树进行预测
            for k in range(self.n_classes):
                for tree in self.trees[k]:
                    update = tree.predict(X)
                    y_pred[:, k] += self.learning_rate * update

            # 返回最高概率的类别索引
            return np.argmax(y_pred, axis=1)

    def predict_proba(self, X):
        """返回概率预测"""
        if self.task == "binary":
            y_pred = np.full(X.shape[0], self.base_prediction, dtype=np.float64)
            for tree in self.trees:
                update = tree.predict(X)
                y_pred += self.learning_rate * update

            proba = self._sigmoid(y_pred)
            return np.vstack((1 - proba, proba)).T

        elif self.task == "multi":
            # 初始化预测矩阵: 样本数 x 类别数
            y_pred = np.zeros((X.shape[0], self.n_classes), dtype=np.float64)

            # 对每个类别的每棵树进行预测
            for k in range(self.n_classes):
                for tree in self.trees[k]:
                    update = tree.predict(X)
                    y_pred[:, k] += self.learning_rate * update

            # 使用softmax转换为概率
            return self._softmax(y_pred)

        else:
            raise ValueError("predict_proba只适用于分类问题")

    def feature_importance(self):
        if not self.trees:
            return None

        if self.task == "reg" or self.task == "binary":
            n_features = self.trees[0].n_features
            importances = np.zeros(n_features)

            for tree in self.trees:
                self._update_feature_importance(tree.root, importances)

            total = np.sum(importances)
            if total > 0:
                importances = importances / total

            return importances

        else:  # 多分类
            n_features = self.trees[0][0].n_features
            importances = np.zeros(n_features)

            # 汇总所有类别的所有树的特征重要性
            for k in range(self.n_classes):
                for tree in self.trees[k]:
                    self._update_feature_importance(tree.root, importances)

            total = np.sum(importances)
            if total > 0:
                importances = importances / total

            return importances

    def _update_feature_importance(self, node, importances, weight=1.0):
        if node.left is None:
            return

        importances[node.feature_index] += node.gain * weight

        self._update_feature_importance(node.left, importances, weight * 0.5)
        self._update_feature_importance(node.right, importances, weight * 0.5)

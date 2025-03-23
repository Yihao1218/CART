import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None, confidence=None):
        self.feature_idx = feature_idx  # 特征索引
        self.threshold = threshold      # 阈值
        self.left = left               # 左子树
        self.right = right             # 右子树
        self.value = value             # 叶节点的值
        self.confidence = confidence    # 叶节点的置信度

class CART:
    def __init__(self, max_depth=None, min_samples_split=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None
        
    def fit(self, X, y):
        """训练决策树"""
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           n_labels == 1:
            leaf_value, confidence = self._most_common_label(y)
            return Node(value=leaf_value, confidence=confidence)
            
        # 寻找最佳分割点
        best_feature_idx, best_threshold = self._best_split(X, y)
        
        if best_feature_idx is None:
            leaf_value, confidence = self._most_common_label(y)
            return Node(value=leaf_value, confidence=confidence)
            
        # 根据最佳分割点划分数据
        left_idxs = X[:, best_feature_idx] <= best_threshold
        right_idxs = ~left_idxs
        
        # 递归构建左右子树
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return Node(best_feature_idx, best_threshold, left, right)
        
    def _best_split(self, X, y):
        """寻找最佳分割点"""
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                gain = self._information_gain(y, X[:, feature_idx], threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    
        return best_feature_idx, best_threshold
        
    def _information_gain(self, y, X_column, threshold):
        """计算信息增益"""
        parent_gini = self._gini(y)
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = sum(left_idxs), sum(right_idxs)
        if n_l == 0 or n_r == 0:
            return 0
        gini_l = self._gini(y[left_idxs])
        gini_r = self._gini(y[right_idxs])
        child_gini = (n_l/n) * gini_l + (n_r/n) * gini_r
        return parent_gini - child_gini
        
    def _gini(self, y):
        """计算基尼指数"""
        proportions = np.bincount(y.astype(int)) / len(y)
        return 1 - np.sum(proportions ** 2)
        
    def _most_common_label(self, y):
        """返回最常见的类别及其置信度"""
        counter = Counter(y)
        most_common = counter.most_common(1)[0]
        return most_common[0], most_common[1] / len(y)
        
    def predict(self, X):
        """预测样本类别"""
        return np.array([self._traverse_tree(x, self.root) for x in X])
        
    def _traverse_tree(self, x, node):
        """遍历决策树进行预测"""
        if node.value is not None:
            return node.value
            
        if x[node.feature_idx] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
        
    def get_node_count(self):
        """获取节点数量"""
        return self._count_nodes(self.root)
        
    def _count_nodes(self, node):
        """递归计算节点数量"""
        if node is None:
            return 0
        return 1 + self._count_nodes(node.left) + self._count_nodes(node.right)
        
    def get_depth(self):
        """获取树的深度"""
        return self._get_depth(self.root)
        
    def _get_depth(self, node):
        """递归计算树的深度"""
        if node is None:
            return 0
        if node.value is not None:
            return 1
        return 1 + max(self._get_depth(node.left), self._get_depth(node.right))
        
    def predict_proba(self, X):
        """预测样本属于每个类别的概率"""
        probas = []
        for x in X:
            node = self.root
            while node.value is None:
                if x[node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # 获取叶节点中各类别的概率分布
            proba = np.zeros(self.n_classes)
            proba[int(node.value)] = node.confidence
            probas.append(proba)
        return np.array(probas)

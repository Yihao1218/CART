import numpy as np
from copy import deepcopy
from cart import CART, Node

class PrePruningCART(CART):
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0.0):
        super().__init__(max_depth, min_samples_split)
        self.min_gain = min_gain
        
    def _best_split(self, X, y):
        """重写寻找最佳分割点的方法，加入预剪枝"""
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        # 计算不分割时的准确率（选择样本最多的类别）
        base_acc = np.max(np.bincount(y.astype(int))) / len(y)
        
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # 计算分割后的子集
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = ~left_idxs
                
                # 确保两个子集都不为空
                if sum(left_idxs) == 0 or sum(right_idxs) == 0:
                    continue
                    
                gain = self._information_gain(y, X[:, feature_idx], threshold)
                
                if gain > best_gain:
                    # 计算分割后的准确率
                    left_counts = np.bincount(y[left_idxs].astype(int))
                    right_counts = np.bincount(y[right_idxs].astype(int))
                    
                    # 确保每个子集都有足够的样本
                    if len(left_counts) == 0 or len(right_counts) == 0:
                        continue
                        
                    left_acc = np.max(left_counts) / sum(left_idxs)
                    right_acc = np.max(right_counts) / sum(right_idxs)
                    split_acc = (sum(left_idxs) * left_acc + sum(right_idxs) * right_acc) / len(y)
                    
                    # 如果分割后的准确率提升不够，则不进行分割
                    if split_acc - base_acc > self.min_gain:
                        best_gain = gain
                        best_feature_idx = feature_idx
                        best_threshold = threshold
                    
        return best_feature_idx, best_threshold


class PostPruningCART(CART):
    def prune(self, X_val, y_val):
        """对训练好的决策树进行后剪枝"""
        self.root = self._prune_node(self.root, X_val, y_val)
        
    def _prune_node(self, node, X_val, y_val):
        """递归对节点进行后剪枝"""
        if node.value is not None:  # 叶节点
            return node
            
        # 获取当前节点对应的验证集样本
        left_idxs = X_val[:, node.feature_idx] <= node.threshold
        right_idxs = ~left_idxs
        
        # 递归处理左右子树
        if sum(left_idxs) > 0:
            node.left = self._prune_node(node.left, X_val[left_idxs], y_val[left_idxs])
        if sum(right_idxs) > 0:
            node.right = self._prune_node(node.right, X_val[right_idxs], y_val[right_idxs])
        
        # 尝试剪枝
        accuracy_before = self._get_accuracy(X_val, y_val)
        
        # 保存当前节点的信息
        feature_idx = node.feature_idx
        threshold = node.threshold
        left = node.left
        right = node.right
        
        # 将当前节点变为叶节点
        node.value, node.confidence = self._most_common_label(y_val)
        node.feature_idx = None
        node.threshold = None
        node.left = None
        node.right = None
        accuracy_after = self._get_accuracy(X_val, y_val)
        
        # 如果剪枝后准确率没有提升，则恢复原状
        if accuracy_after <= accuracy_before:
            node.value = None
            node.feature_idx = feature_idx
            node.threshold = threshold
            node.left = left
            node.right = right
        return node
        
    def _get_accuracy(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


import numpy as np

class DataSplitter:
    @staticmethod
    def holdout_split(X, y, test_size=0.2, random_state=None):
        """留出法划分数据集"""
        if random_state is not None:
            np.random.seed(random_state)
            
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        test_size = int(test_size * n_samples)
        
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def k_fold_split(X, y, n_splits=5, random_state=None):
        """K折交叉验证划分数据集"""
        if random_state is not None:
            np.random.seed(random_state)
            
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_splits
        
        folds = []
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i < n_splits - 1 else n_samples
            test_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            folds.append((X_train, X_test, y_train, y_test))
            
        return folds
    
    @staticmethod
    def bootstrap_split(X, y, n_bootstraps=100, random_state=None):
        """自助法划分数据集"""
        if random_state is not None:
            np.random.seed(random_state)
            
        n_samples = len(X)
        bootstraps = []
        
        for _ in range(n_bootstraps):
            # 有放回采样
            train_indices = np.random.choice(n_samples, n_samples, replace=True)
            # 未被采样的样本作为测试集
            test_indices = np.array(list(set(range(n_samples)) - set(train_indices)))
            
            X_train = X[train_indices]
            X_test = X[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]
            
            bootstraps.append((X_train, X_test, y_train, y_test))
            
        return bootstraps

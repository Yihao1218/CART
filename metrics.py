import numpy as np

class Metrics:
    @staticmethod
    def accuracy_score(y_true, y_pred):
        """计算准确率"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def error_rate(y_true, y_pred):
        """计算错误率"""
        return 1 - Metrics.accuracy_score(y_true, y_pred)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        """计算混淆矩阵"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        classes = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(classes)

        conf_matrix = np.zeros((n_classes, n_classes), dtype=int)

        class_to_index = {cls: idx for idx, cls in enumerate(classes)}

        for yt, yp in zip(y_true, y_pred):
            conf_matrix[class_to_index[yt], class_to_index[yp]] += 1

        return conf_matrix

    @staticmethod
    def precision_recall(y_true, y_pred, average='macro'):
        """计算查准率和查全率"""
        conf_matrix = Metrics.confusion_matrix(y_true, y_pred)
        n_classes = conf_matrix.shape[0]
        
        precisions = np.zeros(n_classes)
        recalls = np.zeros(n_classes)
        
        for i in range(n_classes):
            tp = conf_matrix[i, i]
            fp = np.sum(conf_matrix[:, i]) - tp
            fn = np.sum(conf_matrix[i, :]) - tp
            
            precisions[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recalls[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            
        if average == 'macro':
            return np.mean(precisions), np.mean(recalls)
        return precisions, recalls


    @staticmethod
    def calculate_pr_curve(y_true, y_score):
        """计算P-R曲线"""
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # 获取所有不同的预测分数作为阈值
        thresholds = np.unique(y_score)
        thresholds = np.sort(thresholds)[::-1]  # 从大到小排序
        
        # 初始化
        precision = []
        recall = []

        # 对每个阈值计算Precision和Recall
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            precision.append(p)
            recall.append(r)

        return np.array(precision), np.array(recall), thresholds
    @staticmethod
    def pr_curve(y_true, y_score):
        n_classes = y_score.shape[1]
        
        # 为每个类别计算PR曲线
        pr_curves = []
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            precision, recall, _ = Metrics.calculate_pr_curve(y_true_binary, y_score[:, i])
            pr_curves.append((recall, precision))
            
        return pr_curves 
    @staticmethod
    def auc_score(recalls, precisions):
        """计算AUC值"""
        # 使用梯形法则计算曲线下面积
        return np.trapz(precisions, recalls)


    @staticmethod
    def calculate_roc_curve(y_true, y_score):
        """计算ROC曲线"""
        # 将预测分数和真实标签转换为numpy数组
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # 获取正例和反例的数量
        m_plus = np.sum(y_true == 1)  # 正例数量
        m_minus = np.sum(y_true == 0)  # 反例数量
        
        # 按预测分数从大到小排序
        sorted_indices = np.argsort(-y_score)
        y_true_sorted = y_true[sorted_indices]
        y_score_sorted = y_score[sorted_indices]
        
        # 初始化TPR和FPR
        tpr = [0.0]  # 初始点为 (0, 0)
        fpr = [0.0]
        
        # 初始化TP和FP的计数
        tp = 0
        fp = 0
        
        # 遍历每个样本，逐步调整阈值
        for i in range(len(y_score_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1  # 真正例
            else:
                fp += 1  # 假正例
            
            # 计算当前的TPR和FPR
            tpr.append(tp / m_plus)
            fpr.append(fp / m_minus)
        
        return np.array(fpr), np.array(tpr), y_score_sorted


    @staticmethod
    def roc_curve(y_true, y_score):
        n_classes = y_score.shape[1]
        
        # 为每个类别计算ROC曲线
        roc_curves = []
        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            fpr, tpr, _ = Metrics.calculate_roc_curve(y_true_binary, y_score[:, i])
            auc = Metrics.auc_score(fpr, tpr)
            roc_curves.append((fpr, tpr, auc))
            
        return roc_curves


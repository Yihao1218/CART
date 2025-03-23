import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


class TreeVisualizer:
    def __init__(self, feature_names=None, class_names=None, feature_values=None):
        self.feature_names = feature_names
        self.class_names = class_names
        self.feature_values = feature_values

    def plot_tree(self, tree, filename='tree'):
        """绘制决策树图"""
        dot = Digraph(comment='Decision Tree')
        dot.attr(rankdir='TB')

        def add_nodes(node, parent_id=None, edge_label=None, node_id=None):
            if node_id is None:
                node_id = str(id(node))

            if node.value is not None:
                # 叶节点
                label = f'Class: {self.class_names[int(node.value)]}\nConfidence: {node.confidence:.2f}' if self.class_names else f'Class: {node.value}\nConfidence: {node.confidence:.2f}'
                dot.node(node_id, label, shape='box')
            else:
                # 内部节点
                feature_name = self.feature_names[node.feature_idx] if self.feature_names else f'Feature {node.feature_idx}'
                feature_label = self.feature_values[feature_name][int(node.threshold)]
                label = f'{feature_name} <= {feature_label}?'
                dot.node(node_id, label, shape='oval')

            if parent_id:
                dot.edge(parent_id, node_id, edge_label)

            if node.left:
                add_nodes(node.left, node_id, 'Yes', str(id(node.left)))
            if node.right:
                add_nodes(node.right, node_id, 'No', str(id(node.right)))

        add_nodes(tree.root)
        dot.render(filename, view=True, format='png')


class PerformanceVisualizer:
    @staticmethod
    def plot_confusion_matrix(conf_matrix, class_names=None, filename='confusion_matrix.png'):
            """绘制混淆矩阵"""
            plt.figure(figsize=(10, 8))
            plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')

            if class_names:
                tick_marks = np.arange(len(class_names))
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

            # 在每个单元格中添加数值
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    plt.text(j, i, format(conf_matrix[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if conf_matrix[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()


    @staticmethod
    def plot_pr_curves(pr_curves, class_names=None, filename='pr_curves.png'):
        """
        绘制PR曲线
        
        Parameters:
        -----------
        pr_curves : list of tuples
            每个类别的PR曲线点，每个元素为(recall, precision)元组
        class_names : list of str, optional
            类别名称列表
        filename : str, optional
            保存图片的文件名
        """
        plt.figure(figsize=(10, 8))

        for i, (recalls, precisions) in enumerate(pr_curves):
            # 计算平均精确率（AP）
            ap = np.mean(precisions)
            
            # 绘制曲线
            label = f'{class_names[i]} (AP = {ap:.3f})' if class_names else f'Class {i} (AP = {ap:.3f})'
            plt.plot(recalls, precisions, label=label)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_roc_curves(roc_curves, class_names=None, filename='roc_curves.png'):
        """
        绘制ROC曲线
        
        Parameters:
        -----------
        roc_curves : list of tuples
            每个类别的ROC曲线点，每个元素为(fpr, tpr, auc)元组
        class_names : list of str, optional
            类别名称列表
        filename : str, optional
            保存图片的文件名
        """
        plt.figure(figsize=(10, 8))

        for i, (fprs, tprs, auc) in enumerate(roc_curves):
            # 绘制曲线
            label = f'{class_names[i]} (AUC = {auc:.3f})' if class_names else f'Class {i} (AUC = {auc:.3f})'
            plt.plot(fprs, tprs, label=label)

        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

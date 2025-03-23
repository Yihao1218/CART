import numpy as np

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        self.feature_values = {
            'buying': ['vhigh', 'high', 'med', 'low'],
            'maint': ['vhigh', 'high', 'med', 'low'],
            'doors': ['2', '3', '4', '5more'],
            'persons': ['2', '4', 'more'],
            'lug_boot': ['small', 'med', 'big'],
            'safety': ['low', 'med', 'high']
        }
        self.class_values = ['unacc', 'acc', 'good', 'vgood']
        
    def load_data(self):
        """加载数据并进行预处理"""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                data.append(line.strip().split(','))
        
        data = np.array(data)
        X = data[:, :-1]  # 特征
        y = data[:, -1]   # 标签
        
        # 将类别特征转换为数值
        X_encoded = np.zeros((X.shape[0], X.shape[1]))
        for i, feature in enumerate(self.feature_names):
            values = self.feature_values[feature]
            for j, value in enumerate(values):
                X_encoded[X[:, i] == value, i] = j
                
        # 将类别标签转换为数值
        y_encoded = np.zeros(y.shape)
        for i, value in enumerate(self.class_values):
            y_encoded[y == value] = i
            
        return X_encoded, y_encoded
    
    def decode_labels(self, y_encoded):
        """将数值标签转换回原始类别"""
        y_decoded = np.array([self.class_values[int(i)] for i in y_encoded])
        return y_decoded

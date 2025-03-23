import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from data_split import DataSplitter
from cart import CART
from pruning import PrePruningCART, PostPruningCART
from metrics import Metrics
from visualization import TreeVisualizer, PerformanceVisualizer

def evaluate_model(model, X_test, y_test, class_names, prefix=''):
    """评估模型性能并保存可视化结果"""
    # 预测
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    
    # 计算性能指标
    accuracy = Metrics.accuracy_score(y_test, y_pred)
    error_rate = Metrics.error_rate(y_test, y_pred)
    conf_matrix = Metrics.confusion_matrix(y_test, y_pred)
    precision, recall = Metrics.precision_recall(y_test, y_pred)
    pr_curves = Metrics.pr_curve(y_test, y_score)
    roc_curves = Metrics.roc_curve(y_test, y_score)
    
    # 打印性能指标
    print(f"\n{prefix} Model Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Error Rate: {error_rate:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # 可视化
    vis = PerformanceVisualizer()
    vis.plot_confusion_matrix(conf_matrix, class_names, f'{prefix.lower()}_confusion_matrix.png')
    vis.plot_pr_curves(pr_curves, class_names, f'{prefix.lower()}_pr_curves.png')
    vis.plot_roc_curves(roc_curves, class_names, f'{prefix.lower()}_roc_curves.png')
    
    return accuracy, error_rate, precision, recall

def main():
    # 加载数据
    data_loader = DataLoader('car2/car_1000.txt')
    X, y = data_loader.load_data()
    
    # 特征名称和类别名称
    feature_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    class_names = ['unacc', 'acc', 'good', 'vgood']
    feature_values={
        'buying': ['vhigh', 'high', 'med', 'low'],
        'maint': ['vhigh', 'high', 'med', 'low'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high']
    }
    # 创建可视化器
    tree_vis = TreeVisualizer(feature_names, class_names,feature_values)
    
    # 1. 留出法
    print("\n=== Holdout Method ===")
    X_train, X_test, y_train, y_test = DataSplitter.holdout_split(X, y, test_size=0.2, random_state=42)
    
    # 基础CART
    cart = CART(max_depth=50)
    cart.fit(X_train, y_train)
    tree_vis.plot_tree(cart, 'holdout_cart')
    evaluate_model(cart, X_test, y_test, class_names, 'Holdout CART')
    
    # 预剪枝CART
    pre_cart = PrePruningCART(max_depth=50,min_gain=0.0)
    pre_cart.fit(X_train, y_train)
    tree_vis.plot_tree(pre_cart, 'holdout_pre_pruning_cart')
    evaluate_model(pre_cart, X_test, y_test, class_names, 'Holdout Pre-pruning CART')
    
    # 后剪枝CART
    post_cart = PostPruningCART(max_depth=50)
    post_cart.fit(X_train, y_train)
    post_cart.prune(X_test, y_test)
    tree_vis.plot_tree(post_cart, 'holdout_post_pruning_cart')
    evaluate_model(post_cart, X_test, y_test, class_names, 'Holdout Post-pruning CART')
    
    
    # 2. K折交叉验证
    print("\n=== K-fold Cross Validation ===")
    k = 5
    folds = DataSplitter.k_fold_split(X, y, n_splits=k, random_state=42)
    
    results = {
        'CART': {'acc': [], 'err': [], 'prec': [], 'rec': []},
        'Pre-pruning': {'acc': [], 'err': [], 'prec': [], 'rec': []},
        'Post-pruning': {'acc': [], 'err': [], 'prec': [], 'rec': []}
    }
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(folds):
        print(f"\nFold {i+1}:")
        
        # 基础CART
        cart = CART(max_depth=100)
        cart.fit(X_train, y_train)
        tree_vis.plot_tree(cart, f'kfold_cart_fold{i+1}')
        acc, err, prec, rec = evaluate_model(cart, X_test, y_test, class_names, f'K-fold CART (Fold {i+1})')
        results['CART']['acc'].append(acc)
        results['CART']['err'].append(err)
        results['CART']['prec'].append(prec)
        results['CART']['rec'].append(rec)
        
        # 预剪枝CART
        pre_cart = PrePruningCART(max_depth=100, min_gain=0.0)
        pre_cart.fit(X_train, y_train)
        if i == 0:
            tree_vis.plot_tree(pre_cart, f'kfold_pre_pruning_cart_fold{i+1}')
        acc, err, prec, rec = evaluate_model(pre_cart, X_test, y_test, class_names, f'K-fold Pre-pruning CART (Fold {i+1})')
        results['Pre-pruning']['acc'].append(acc)
        results['Pre-pruning']['err'].append(err)
        results['Pre-pruning']['prec'].append(prec)
        results['Pre-pruning']['rec'].append(rec)
        
        # 后剪枝CART
        post_cart = PostPruningCART(max_depth=100)
        post_cart.fit(X_train, y_train)
        post_cart.prune(X_test, y_test)
        if i == 0:
            tree_vis.plot_tree(post_cart, f'kfold_post_pruning_cart_fold{i+1}')
        acc, err, prec, rec = evaluate_model(post_cart, X_test, y_test, class_names, f'K-fold Post-pruning CART (Fold {i+1})')
        results['Post-pruning']['acc'].append(acc)
        results['Post-pruning']['err'].append(err)
        results['Post-pruning']['prec'].append(prec)
        results['Post-pruning']['rec'].append(rec)
    
    # 打印K折交叉验证平均结果
    print("\nK-fold Cross Validation Average Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Average Accuracy: {np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}")
        print(f"Average Error Rate: {np.mean(metrics['err']):.4f} ± {np.std(metrics['err']):.4f}")
        print(f"Average Precision: {np.mean(metrics['prec']):.4f} ± {np.std(metrics['prec']):.4f}")
        print(f"Average Recall: {np.mean(metrics['rec']):.4f} ± {np.std(metrics['rec']):.4f}")
    
    # 3. 自助法
    print("\n=== Bootstrap Method ===")
    bootstraps = DataSplitter.bootstrap_split(X, y, n_bootstraps=5, random_state=41)
    
    results = {
        'CART': {'acc': [], 'err': [], 'prec': [], 'rec': []},
        'Pre-pruning': {'acc': [], 'err': [], 'prec': [], 'rec': []},
        'Post-pruning': {'acc': [], 'err': [], 'prec': [], 'rec': []}
    }
    
    for i, (X_train, X_test, y_train, y_test) in enumerate(bootstraps):
        print(f"\nBootstrap {i+1}:")
        
        # 基础CART
        cart = CART(max_depth=100)
        cart.fit(X_train, y_train)
        if i == 0:
            tree_vis.plot_tree(cart, f'bootstrap_cart_{i+1}')
        acc, err, prec, rec = evaluate_model(cart, X_test, y_test, class_names, f'Bootstrap CART ({i+1})')
        results['CART']['acc'].append(acc)
        results['CART']['err'].append(err)
        results['CART']['prec'].append(prec)
        results['CART']['rec'].append(rec)
        
        # 预剪枝CART
        pre_cart = PrePruningCART(max_depth=100, min_gain=0.01)
        pre_cart.fit(X_train, y_train)
        if i == 0:
            tree_vis.plot_tree(pre_cart, f'bootstrap_pre_pruning_cart_{i+1}')
        acc, err, prec, rec = evaluate_model(pre_cart, X_test, y_test, class_names, f'Bootstrap Pre-pruning CART ({i+1})')
        results['Pre-pruning']['acc'].append(acc)
        results['Pre-pruning']['err'].append(err)
        results['Pre-pruning']['prec'].append(prec)
        results['Pre-pruning']['rec'].append(rec)
        
        # 后剪枝CART
        post_cart = PostPruningCART(max_depth=100)
        post_cart.fit(X_train, y_train)
        post_cart.prune(X_test, y_test)
        if i == 0:
            tree_vis.plot_tree(post_cart, f'bootstrap_post_pruning_cart_{i+1}')
        acc, err, prec, rec = evaluate_model(post_cart, X_test, y_test, class_names, f'Bootstrap Post-pruning CART ({i+1})')
        results['Post-pruning']['acc'].append(acc)
        results['Post-pruning']['err'].append(err)
        results['Post-pruning']['prec'].append(prec)
        results['Post-pruning']['rec'].append(rec)
        
    # 打印自助法平均结果
    print("\nBootstrap Average Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"Average Accuracy: {np.mean(metrics['acc']):.4f} ± {np.std(metrics['acc']):.4f}")
        print(f"Average Error Rate: {np.mean(metrics['err']):.4f} ± {np.std(metrics['err']):.4f}")
        print(f"Average Precision: {np.mean(metrics['prec']):.4f} ± {np.std(metrics['prec']):.4f}")
        print(f"Average Recall: {np.mean(metrics['rec']):.4f} ± {np.std(metrics['rec']):.4f}")
    
if __name__ == '__main__':
    main()

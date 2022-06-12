import numpy as np
import matplotlib.pyplot as plt

# Custom classes
import sys
sys.path.insert(0,'..')
from meta_learning import evaluator


def plot_results(y, predicted_y, title, idx, subplot_index=(3, 3)):
    r2 = evaluator.evaluate(y, predicted_y, 'r2')
    mse = evaluator.evaluate(y, predicted_y, 'mse')
    std = np.std(y - predicted_y)
    x = range(len(y))

    plt.subplot(*subplot_index, idx + 1)
    plt.plot(x, y, label="expected")
    plt.plot(x, predicted_y, label="predicted")
    plt.title(title)
    plt.legend()

    return {'r2': r2, 'mse': mse, 'std': std}


def offline_train(update_params, learner, validation_base):
    learner._update_params(**update_params)
    learner.meta_base = learner._get_last_performances(learner.meta_base)
    learner._train_meta_model()

    model = learner.meta_model
    metric = learner.meta_label_metric
    val_base = learner._get_last_performances(validation_base)

    X, y_true = val_base.drop(learner.performance_metrics, axis=1), val_base[metric]
    y_pred = model.predict(X)
    return y_true, y_pred
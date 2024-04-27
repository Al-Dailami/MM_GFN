from sklearn import metrics
import numpy as np

class CustomBins:
    inf = 1e18
    bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
    nbins = len(bins)

def get_bin_custom(x, nbins, one_hot=False):
    for i in range(nbins):
        a = CustomBins.bins[i][0]
        b = CustomBins.bins[i][1]
        if a <= x < b:
            if one_hot:
                onehot = np.zeros((CustomBins.nbins,))
                onehot[i] = 1
                return onehot
            return i
    return None

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(4/24, y_true))) * 100  # this stops the mape being a stupidly large value when y_true happens to be very small

def mean_squared_logarithmic_error(y_true, y_pred):
    return np.mean(np.square(np.log(y_true/y_pred)))

def compute_regression_metrics(y_true, predictions, verbose=1, elog=None):
    print('==> Length of Stay:')
    y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
    prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if elog is not None:
        elog.print('Custom bins confusion matrix:')
        elog.print(cf)
    if verbose:
        print('Custom bins confusion matrix:')
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, predictions)
    msle = mean_squared_logarithmic_error(y_true, predictions)
    rmsle = np.sqrt(msle)
    r2 = metrics.r2_score(y_true, predictions)

    if elog is not None:
        elog.print('Mean absolute deviation (MAD) = {}'.format(mad))
        elog.print('Mean squared error (MSE) = {}'.format(mse))
        elog.print('Root mean squared error (RMSE) = {}'.format(rmse))
        elog.print('Mean absolute percentage error (MAPE) = {}'.format(mape))
        elog.print('Mean squared logarithmic error (MSLE) = {}'.format(msle))
        elog.print('Root mean squared logarithmic error (RMSLE) = {}'.format(rmsle))
        elog.print('R^2 Score = {}'.format(r2))
        elog.print('Cohen kappa score = {}'.format(kappa))
    if verbose:
        print('Mean absolute deviation (MAD) = {}'.format(mad))
        print('Mean squared error (MSE) = {}'.format(mse))
        print('Root mean squared error (RMSE) = {}'.format(rmse))
        print('Mean absolute percentage error (MAPE) = {}'.format(mape))
        print('Mean squared logarithmic error (MSLE) = {}'.format(msle))
        print('Root mean squared logarithmic error (RMSLE) = {}'.format(rmsle))
        print('R^2 Score = {}'.format(r2))
        print('Cohen kappa score = {}'.format(kappa))

    return [mad, mse, rmse, mape, msle, rmsle, r2, kappa]

def compute_binary_metrics(y_true, prediction_probs, verbose=1, elog=None):
    print('==> Mortality:')
    prediction_probs = np.array(prediction_probs)
    prediction_probs = np.transpose(np.append([1 - prediction_probs], [prediction_probs], axis=0))
    predictions = prediction_probs.argmax(axis=1)
    cf = metrics.confusion_matrix(y_true, predictions, labels=range(2))
    if elog is not None:
        elog.print('Confusion matrix:')
        elog.print(cf)
    elif verbose:
        print('Confusion matrix:')
        print(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])

    auroc = metrics.roc_auc_score(y_true, prediction_probs[:, 1])
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, prediction_probs[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    f1macro = metrics.f1_score(y_true, predictions, average='macro')

    results = {'Accuracy': acc, 'Precision Survived': prec0, 'Precision Died': prec1, 'Recall Survived': rec0,
               'Recall Died': rec1, 'Area Under the Receiver Operating Characteristic curve (AUROC)': auroc,
               'Area Under the Precision Recall curve (AUPRC)': auprc, "minpse": minpse, 'F1 score (macro averaged)': f1macro}
    
    if elog is not None:
        for key in results:
            elog.print('{} = {}'.format(key, results[key]))
    if verbose:
        for key in results:
            print('{} = {}'.format(key, results[key]))

#     return results
    return [acc, prec0, prec1, rec0, rec1, auroc, auprc, minpse, f1macro]
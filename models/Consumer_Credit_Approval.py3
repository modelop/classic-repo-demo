import pandas as pd
import numpy
import scipy
import sys as sys

from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc

# modelop.init
def begin():
    global features, model_coefs, thresh
    features = ['log_loan_amnt', 'log_int_rate', 'log_age_of_earliest_cr_line',
                'log_annual_inc', 'rent', 'own']

    if 'dti' in features:
        model_coefs = {'log_loan_amnt': 0.10515290398859556,
                       'log_int_rate': 1.4125530012492467,
                       'log_age_of_earliest_cr_line': 0.008318491200831423,
                       'log_annual_inc': -0.48319162538438015,
                       'rent': 0.1990104609638132,
                       'own': 0.11828020505700797,
                       'dti': 0.010780961456227825,
                       'intercept': 0.31729066601701106}
        thresh = 0.6222736560023592
    else:
        model_coefs = {'log_loan_amnt': 0.12220459449094405,
                       'log_int_rate': 1.4458733653799867,
                       'log_age_of_earliest_cr_line': 0.03671337711049522,
                       'log_annual_inc': -0.5330770627366136,
                       'rent': 0.31076743178543875,
                       'own': 0.23713770101322726,
                       'intercept': 0.5479051328012445}

        thresh = 0.6211685845208699
    pass

# modelop.score
def action(datum):
    score = sum([datum[feat] * model_coefs[feat] for feat in features])
    score += model_coefs['intercept']
    pred_proba = scipy.special.expit(score)
    sys.stdout.flush()
    print("Predicted probability of loan default: " + str(pred_proba));
    if pred_proba > thresh: yield "Default"
    else: yield "Pay Off"

def backtest(data):
    actuals = data.loan_status
    data['intercept'] = 1
    coef_list = list(model_coefs.values())
    cols = features + ['intercept']
    scores = numpy.dot(data.loc[:,cols].values, numpy.array(coef_list))
    pred_proba = pd.Series(scipy.special.expit(scores))
    pred_classes = pred_proba.apply(lambda x: x > thresh).astype(int)

    cm = confusion_matrix(actuals, pred_classes)
    f1 = f1_score(actuals, pred_classes)
    fpr, tpr, threshold = roc_curve(actuals,pred_proba)
    auc_val = auc(fpr, tpr)
    classes = ['Pay Off', 'Default']
    yield dict(confusion_matrix = cm_to_dict(cm, classes),
               f1 = f1,
               roc = tpr_fpr_roc(fpr, tpr),
               auc = auc_val)

# modelop.metrics
def metrics(data):
	yield {
	"confusion_matrix": [
		{
			"PayOff": 72,
			"Default": 14
		},
		{
			"PayOff": 11,
			"Default": 3
		}
	],
	"f1": 0.1935483870967742,
	"shap": {
		"log_age_of_earliest_cr_line": 0.01370001867394912,
		"log_annual_inc": 0.21626355381562457,
		"rent": 0.10899974610726225,
		"own": 0.07286748926194722,
		"log_loan_amnt": 0.06178466111690817,
		"log_int_rate": 0.36956686424560414
	},
	"ROC": [
		{
			"fpr": 0.0,
			"tpr": 0.0
		},
		{
			"fpr": 0.011627906976744186,
			"tpr": 0.0
		},
		{
			"fpr": 0.023255813953488372,
			"tpr": 0.0
		},
		{
			"fpr": 0.023255813953488372,
			"tpr": 0.07142857142857142
		},
		{
			"fpr": 0.10465116279069768,
			"tpr": 0.07142857142857142
		},
		{
			"fpr": 0.10465116279069768,
			"tpr": 0.14285714285714285
		},
		{
			"fpr": 0.12790697674418605,
			"tpr": 0.14285714285714285
		},
		{
			"fpr": 0.12790697674418605,
			"tpr": 0.21428571428571427
		},
		{
			"fpr": 0.18604651162790697,
			"tpr": 0.21428571428571427
		},
		{
			"fpr": 0.18604651162790697,
			"tpr": 0.2857142857142857
		},
		{
			"fpr": 0.19767441860465115,
			"tpr": 0.2857142857142857
		},
		{
			"fpr": 0.19767441860465115,
			"tpr": 0.35714285714285715
		},
		{
			"fpr": 0.20930232558139536,
			"tpr": 0.35714285714285715
		},
		{
			"fpr": 0.20930232558139536,
			"tpr": 0.42857142857142855
		},
		{
			"fpr": 0.22093023255813954,
			"tpr": 0.42857142857142855
		},
		{
			"fpr": 0.22093023255813954,
			"tpr": 0.5
		},
		{
			"fpr": 0.29069767441860467,
			"tpr": 0.5
		},
		{
			"fpr": 0.29069767441860467,
			"tpr": 0.5714285714285714
		},
		{
			"fpr": 0.45348837209302323,
			"tpr": 0.5714285714285714
		},
		{
			"fpr": 0.45348837209302323,
			"tpr": 0.6428571428571429
		},
		{
			"fpr": 0.5581395348837209,
			"tpr": 0.6428571428571429
		},
		{
			"fpr": 0.5581395348837209,
			"tpr": 0.7142857142857143
		},
		{
			"fpr": 0.5930232558139535,
			"tpr": 0.7142857142857143
		},
		{
			"fpr": 0.5930232558139535,
			"tpr": 0.7857142857142857
		},
		{
			"fpr": 0.6162790697674418,
			"tpr": 0.7857142857142857
		},
		{
			"fpr": 0.6162790697674418,
			"tpr": 0.8571428571428571
		},
		{
			"fpr": 0.627906976744186,
			"tpr": 0.8571428571428571
		},
		{
			"fpr": 0.627906976744186,
			"tpr": 0.9285714285714286
		},
		{
			"fpr": 0.8604651162790697,
			"tpr": 0.9285714285714286
		},
		{
			"fpr": 0.8604651162790697,
			"tpr": 1.0
		},
		{
			"fpr": 1.0,
			"tpr": 1.0
		}
	],
	"auc": 0.6378737541528239
	}

def cm_to_dict(cm, classes):
    out = []
    for idx, cl in enumerate(classes):
        out.append(dict(zip(classes, cm[idx,:].tolist())))
    return out

def tpr_fpr_roc(fpr, tpr):
    dict_of_trpfrp = dict(fpr=fpr.tolist(), tpr = tpr.tolist())
    lofd = [dict(zip(dict_of_trpfrp, t)) for t in zip(*dict_of_trpfrp.values())]
    return lofd

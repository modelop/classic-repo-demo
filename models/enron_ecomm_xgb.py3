# fastscore.recordsets.0: true
# fastscore.recordsets.1: true
# fastscore.module-attached: gensim
# fastscore.module-attached: xgboost

import sys
import pickle
try:
    import gensim
except ImportError:
    pass
import pandas as pd
import numpy
try:
    import xgboost
except ImportError:
    pass


def pad_vectors(list_of_vecs):
    for j in range(len(list_of_vecs), 2**2):
        list_of_vecs.append(numpy.zeros(2**2))
    return numpy.array(list_of_vecs)


def preprocess(df):
    tokens = df.content.astype(str).apply(lambda x: list(gensim.utils.tokenize(x)))
    tokens = tokens.apply(lambda x: x[:2**2])
    vectorized = tokens.apply(lambda x: list(map(lambda y: ft_model.wv[y], x)))
    padded_vectors = vectorized.apply(pad_vectors)
    padded_vectors = padded_vectors.apply(lambda x: x.flatten())
    data = padded_vectors.to_list()
    data = numpy.array(data)
    data = pd.DataFrame(data, index = padded_vectors.index, columns = range(2**4))
    return data


# modelop.init
def conditional_begin():
    if 'xgboost_not_present' in sys.modules:
        begin()


def begin():
    global threshold, xgb_model, ft_model
    xgb_model_artifacts = pickle.load(open('xgb_model_artifacts.pkl', 'rb'))
    threshold = xgb_model_artifacts['threshold']
    ft_model = xgb_model_artifacts['ft_model']

    xgb_model = xgboost.XGBClassifier()
    xgb_model.load_model('xgb_model.model')
    pass


# modelop.score
def action(df):
   
    cleaned = preprocess(df)
    pred_proba = xgb_model.predict_proba(cleaned)[:,1]
    pred_proba = pd.Series(pred_proba, index = df.index)
    preds = pred_proba.apply(lambda x: x > threshold).astype(int)

    output = pd.concat([df, preds], axis=1)
    output.columns = ['id', 'content', 'prediction']
    yield output


# modelop.metrics
def metrics(datum):
    yield {
    "ROC": [
        {
            "fpr": 0,
            "tpr": 0
        },
        {
            "fpr": 0,
            "tpr": 0.3333333333333333
        },
        {
            "fpr": 0.0375,
            "tpr": 0.3333333333333333
        },
        {
            "fpr": 0.0375,
            "tpr": 0.6666666666666666
        },
        {
            "fpr": 0.1875,
            "tpr": 0.6666666666666666
        },
        {
            "fpr": 0.1875,
            "tpr": 1
        },
        {
            "fpr": 0.25,
            "tpr": 1
        },
        {
            "fpr": 0.275,
            "tpr": 1
        },
        {
            "fpr": 0.5375,
            "tpr": 1
        },
        {
            "fpr": 0.5625,
            "tpr": 1
        },
        {
            "fpr": 0.65,
            "tpr": 1
        },
        {
            "fpr": 0.675,
            "tpr": 1
        },
        {
            "fpr": 0.75,
            "tpr": 1
        },
        {
            "fpr": 0.9,
            "tpr": 1
        },
        {
            "fpr": 1,
            "tpr": 1
        }
    ],
    "auc": 0.9249999999999999,
    "f2_score": 0.5263157894736842,
    "cost_per_f2_point": 1.16,
    "confusion_matrix": [
        {
            "Compliant": 75,
            "Non-Compliant": 5
        },
        {
            "Compliant": 1,
            "Non-Compliant": 2
        }
    ],
    "shap" : {

    },
    "bias" : {
        "attributeAudited": "Gender",
        "referenceGroup": "Male",
        "fairnessThreshold": "80%",
        "fairnessMeasures": [
            {
                "label": "Equal Parity",
                "result": "Failed",
                "group": "Female",
                "disparity": 0.67
            },
            {
                "label": "Proportional Parity",
                "result": "Passed",
                "group": None,
                "disparity": 1.1
            },
            {
                "label": "False Positive Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 1.17
            },
            {
                "label": "False Discovery Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 0.82
            },
            {
                "label": "False Negative Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 1.1
            },
            {
                "label": "False Omission Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 0.88
            }
        ]
        }
    }



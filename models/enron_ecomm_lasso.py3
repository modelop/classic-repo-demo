# fastscore.recordsets.0: true
# fastscore.recordsets.1: false

import pickle
try:
    import nltk
except ImportError:
    pass
try:
    import gensim
except ImportError:
    pass
import functools
import pandas as pd
import scipy
import sys

def remove_proper_nouns(string):
    list_of_words = string.split()
    tagged_low = nltk.tag.pos_tag(list_of_words)
    removed_proper_nouns = list(filter(lambda x: x[1] != 'NNP', tagged_low))
    untagged_low = list(map(lambda x: x[0], removed_proper_nouns))
    return " ".join(untagged_low)

def preprocess(series):

    removed_proper_nouns = series.astype(str).apply(remove_proper_nouns)
    CUSTOM_FILTERS = [lambda x: x.lower(), 
                      gensim.parsing.preprocessing.strip_tags, 
                      gensim.parsing.preprocessing.strip_punctuation]

    preprocessing = gensim.parsing.preprocess_string
    preprocessing_filters = functools.partial(preprocessing, filters=CUSTOM_FILTERS)
    removed_punctuation = removed_proper_nouns.apply(preprocessing)

    stopword_remover = gensim.parsing.preprocessing.remove_stopwords
    stopword_remover_list = lambda x: list(map(stopword_remover, x))
    cleaned = removed_punctuation.apply(stopword_remover_list)

    filter_short_words = lambda x: list(filter(lambda y: len(y) > 1, x))
    cleaned = cleaned.apply(filter_short_words)

    filter_non_alpha = lambda x: list(filter(lambda y: y.isalpha(), x))
    cleaned = cleaned.apply(filter_non_alpha)
    
    stemmer = nltk.stem.porter.PorterStemmer()
    list_stemmer = lambda x: list(map(lambda y: stemmer.stem(y), x))
    cleaned = cleaned.apply(list_stemmer)

    return cleaned

def pad_sparse_matrix(sp_mat, length, width):
    sp_data = (sp_mat.data, sp_mat.indices, sp_mat.indptr)
    padded = scipy.sparse.csr_matrix(sp_data, shape=(length, width))
    return padded


# modelop.init
def conditional_begin():
    if 'gensim' in sys.modules:
        begin()


def begin():
    global lasso_model_artifacts 
    lasso_model_artifacts = pickle.load(open('lasso_model_artifacts.pkl', 'rb'))
    pass


# modelop.score
def action(x):
    lasso_model = lasso_model_artifacts['lasso_model']
    dictionary = lasso_model_artifacts['dictionary']
    threshold = lasso_model_artifacts['threshold']
    tfidf_model = lasso_model_artifacts['tfidf_model']
    
    cleaned = preprocess(x.content)
    corpus = cleaned.apply(dictionary.doc2bow)
    corpus_sparse = gensim.matutils.corpus2csc(corpus).transpose()
    corpus_sparse_padded = pad_sparse_matrix(sp_mat = corpus_sparse, 
                                             length=corpus_sparse.shape[0], 
                                             width = len(dictionary))
    sys.stdout.flush()
    tfidf_vectors = tfidf_model.transform(corpus_sparse_padded)

    probabilities = lasso_model.predict_proba(tfidf_vectors)[:,1]

    predictions = pd.Series(probabilities > threshold, index=x.index).astype(int)
    output = pd.concat([x, predictions], axis=1)
    output.columns = ['content', 'id', 'prediction']
    # yield output
    yield output.to_dict(orient="records")


# modelop.metrics
def metrics(datum):
    yield {
    "ROC": [
        {
            "fpr": 0,
            "tpr": 0
        },
        {
            "fpr": 0.0125,
            "tpr": 0
        },
        {
            "fpr": 0.0125,
            "tpr": 0.3333333333333333
        },
        {
            "fpr": 0.025,
            "tpr": 0.3333333333333333
        },
        {
            "fpr": 0.025,
            "tpr": 0.6666666666666666
        },
        {
            "fpr": 0.275,
            "tpr": 0.6666666666666666
        },
        {
            "fpr": 0.275,
            "tpr": 1
        },
        {
            "fpr": 0.4375,
            "tpr": 1
        },
        {
            "fpr": 0.4625,
            "tpr": 1
        },
        {
            "fpr": 0.5,
            "tpr": 1
        },
        {
            "fpr": 0.75,
            "tpr": 1
        },
        {
            "fpr": 0.8125,
            "tpr": 1
        },
        {
            "fpr": 0.8625,
            "tpr": 1
        },
        {
            "fpr": 0.925,
            "tpr": 1
        },
        {
            "fpr": 0.95,
            "tpr": 1
        },
        {
            "fpr": 0.975,
            "tpr": 1
        },
        {
            "fpr": 1,
            "tpr": 1
        }
    ],
    "auc": 0.8958333333333333,
    "f2_score": 0.625,
    "confusion_matrix": [
        {
            "Compliant": 78,
            "Non-Compliant": 2
        },
        {
            "Compliant": 1,
            "Non-Compliant": 2
        }
    ],
    "shap" : {
        "strategi": 3.2520682958749694,
        "indic": 0.7102795675925868,
        "net": 0.3541640953338466,
        "research": 2.340609152949877,
        "market": 1.6368844559196445,
        "version": 1.9101731885557005,
        "good": 1.3562284993616274,
        "mail": 0.6928553560354906,
        "bid": 2.8162293879415072,
        "data": 3.387712516393189,
        "report": 0.47704432407553465,
        "greet": 2.754767300818468,
        "said": 4.54813893155902,
        "approach": 5.799971287318539,
        "develop": 3.3643937399880004,
        "law": 2.196729052019291,
        "normal": 2.681564509767407,
        "pocket": 1.2258033392225045,
        "trader": 0.8864414506427811,
        "trigger": 1.9309807702640416,
        "label": 2.885248298256796,
        "softwar": 1.995336285501284
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
                "disparity": 0.75
            },
            {
                "label": "Proportional Parity",
                "result": "Passed",
                "group": None,
                "disparity": 1.05
            },
            {
                "label": "False Positive Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 0.97
            },
            {
                "label": "False Discovery Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 0.88
            },
            {
                "label": "False Negative Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 1.05
            },
            {
                "label": "False Omission Rate Parity",
                "result": "Passed",
                "group": "Female",
                "disparity": 0.93
            }
        ]
        }
    }




import numpy
import pickle
import pandas as pd


# modelop.init
def begin():
    global xgb_model, means, stdevs, shap_explainer, garageyrbuilt_mean, features

    artifacts = pickle.load(open('/fastscore/xgb_shap_artifacts.pkl','rb'))
    xgb_model = artifacts['xgb_model']
    shap_explainer = artifacts['shap_explainer']
    means = artifacts['means']
    stdevs = artifacts['stdevs']
    garageyrbuilt_mean = artifacts['garageyrbuilt_mean'] #Used for imputing a missing
    features = artifacts['features'] #XGBoost is finicky about order
    pass


# modelop.score
def action(datum):
    standard_datum = datum
    for k in means.keys():
        standard_datum[k] = (datum[k] - means[k]) / stdevs[k]

    if not datum['GarageYrBlt']:
        datum['GarageYrBlt'] = garageyrbuilt_mean

    idx = standard_datum.pop("Id")

    pd_datum = pd.DataFrame(standard_datum, index=[idx], columns = features)
    prediction_raw = xgb_model.predict(pd_datum)[0]
    prediction = numpy.exp(prediction_raw)

    shap_values = shap_explainer.shap_values(pd_datum.values)[0]
    shap_values = pd.Series(shap_values, index=features)

    yield dict(Id = idx, xgboost_pred = prediction, shap_values=shap_values.to_dict())

# modelop.metrics
def metrics(datum):
    yield """
    {
    "rmse": 0.16593233547965738,
    "mae": 0.12163949222544038,
    "explained_variance": 0.8337244947859745,
    "r_squared": 0.8326815731399975,
    "shap": {
        "FirstFlrSF": 0.13801132708612732,
        "SecondFlrSF": 0.02738545211258212,
        "GarageYrBlt": 0.009966999923768783,
        "YearBuilt": 0.12276532809950753,
        "GarageCars": 0.07883269740135544,
        "GarageArea": 0.25306942718400616,
        "TotalBsmtSF": 0.10114252998063433,
        "LotFrontage": 0.004683047129816955,
        "LotArea": 0.03082993027062292,
        "no_garage": 0.0,
        "finished_garage": 0.06533796171781912,
        "unfinished_garage": 0.0
    }
}
    """




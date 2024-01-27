from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

from logistic import LogisticRegression
from SVM import SVM
from LRclassificationMetrics import ClassificationMetrics
from standarize import standardize_data
from scale import scale


app = Flask(__name__, static_url_path="/static")


# Load the trained model from the .pkl file
# with open("D:/BreastCancerPrediction/LRmodel.pkl", "rb") as file:
#     model = LogisticRegression.load_model("D:/BreastCancerPrediction/LRmodel.pkl")


with open("D:/BreastCancerPrediction/SVM_model.pkl", "rb") as file:
    model = SVM.load_model("D:/BreastCancerPrediction/SVM_model.pkl")

# print(model)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        print(request)
        print(features)
        final_features = [np.array(features)]
        final_features = standardize_data(final_features)

        print(final_features)
        prediction_proba = model.predict_proba(final_features)
        benign_prob = prediction_proba[0]
        print([benign_prob])

        output = model.predict(final_features)
        print(output)

        if np.any(output == 1):
            res_val = f"The patient is predicted as Malignant with a probability of {benign_prob:.2%}."
        else:
            res_val = f"The patient is predicted as Benign with a probability of {1 - benign_prob:.2%}."

        return render_template("predict.html", prediction="{}".format(res_val))

    elif request.method == "GET":
        return render_template("predict.html")


@app.route("/data")
def data():
    return render_template("data.html")


@app.route("/faq")
def faq():
    return render_template("faq.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == "__main__":
    app.run(debug=True)

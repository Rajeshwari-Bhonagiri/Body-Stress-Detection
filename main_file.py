from flask import Flask, render_template, url_for, request
from sklearn.externals import joblib
import os
import numpy as np
import pickle
#import seiral

app = Flask(__name__, static_folder='static')

@app.route("/")
def index():
    return render_template('home.html')


@app.route('/result', methods=['POST', 'GET'])
def result():
    age = int(request.form['age'])
    sex = int(request.form['sex'])
    bp = float(request.form['bp'])
    hb = int(request.form['hb'])
    temp = float(request.form['temp'])
    
    x = np.array([age, sex, bp, hb, temp]).reshape(1, -1)

    scaler_path = os.path.join(os.path.dirname(__file__), 'models/scaler.pkl')
    scaler = None
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    x = scaler.transform(x)

    model_path = os.path.join(os.path.dirname(__file__), 'models/rfc.sav')
    clf = joblib.load(model_path)

    y = clf.predict(x)
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print(y)

    # No heart disease
    if y == 0:
        return render_template('nostress.html')

    # y=1,2,4,4 are stages of heart disease
    else:
        return render_template('stress.htm', stage=int(y))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == "__main__":
    app.run(debug=True)

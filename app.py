import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import csv

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
    '''

    f = request.form['file']
    data = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data.append(row)
    return render_template('data.html', data=data)



if __name__ == "__main__":
    app.run(debug=True)
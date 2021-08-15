import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import csv
import os

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
loaded_model = pickle.load(open("finalized_model.yml", 'rb'))

max_wavelength = 1600
min_wavelength = 950

a = round((max_wavelength - 901)/3.5)
b = round((min_wavelength - 901)/3.5)

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

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
    '''
    f = request.files['file']
    data = []
    with open(f, encoding="utf8") as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data.append(row)

    return render_template('data.html', data=data)
'''
    file = request.files["file"]
    file.save(os.path.join("uploads", file.filename))
    path_name= './uploads/'+file.filename
    test = pd.read_csv(path_name,skiprows=21)
    test = test.iloc[b:(a+1),1:2]
    test = np.asarray(test)
    test = pd.DataFrame(test.T)
    result = loaded_model.predict(test)
    comment = 'Kết quả là: Mẫu' + str(result)
    return render_template("index.html", message=comment)

if __name__ == "__main__":
    app.run(debug=True)
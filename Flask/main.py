import os
import json
import flask
import joblib
import numpy as np
from keras.models import load_model
        
app = flask.Flask(__name__)
app.secret_key = " "

@app.route('/') 
def index():
    return flask.render_template("main.html")

@app.route('/', methods=['POST'])
def my_form_post():
    text = flask.request.form['input']
    I = np.array(json.loads(text))
    I = np.reshape(I, (1, ) + I.shape)
    model = load_model('Models/model.h5')
    scaler = joblib.load('Models/scaler.pkl', 'r')
    prediction = model.predict(I)
    prediction = scaler.inverse_transform(prediction)
    output = 'The Stock Predictions are : <br><br>Open : ${} <br>Close : ${}'.format(prediction[0][0], prediction[0][1])
    return output

port = int(os.environ.get('PORT', 10000))
app.run(host = '0.0.0.0', port = port)

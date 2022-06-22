from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('LinearRegression.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    print(request.form)
    # taking data from the form
    features = [float(x) for x in request.form.values()]
    # keeping the features in an array
    feature_arr = [np.array(features)]
    # print(feature_arr)
    # performing prediction on our model
    prediction = model.predict(feature_arr)
   # outcome = round(prediction[0], 2)

    return render_template('index.html', pred=' Profit expectation: 🤗\n{}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask,render_template,url_for,request
import numpy as np
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

def valuePredictor(to_predict_list, model_ensemble, model_qda, model_rf):
    size = len(to_predict_list)
    to_predict = np.array(to_predict_list).reshape(1,size)

    preds = model_qda.predict(to_predict)
    # 0 : operating, 1 : closed
    pred = preds[0]
    if pred == 0:
        result = 'operating'
    elif pred == 1:
        preds = model_rf.predict(to_predict)
    # 0 : operating, 1 : acquired, 2 : closed, 3 : ipo
        pred = preds[0]
        if pred == 0:
            result = 'operating'
        elif pred == 1:
            result = 'acquired'
        elif pred == 2:
            result = 'closed'
        elif pred == 3:
            result = 'ipo'

    return result

@app.route('/predict',methods=['POST'])
def predict():
    model_ensemble = pickle.load(open('./models/ensemble.pkl', 'rb'))
    model_qda = pickle.load(open('./models/qda.pkl', 'rb'))
    model_rf = pickle.load(open('./models/rf.pkl', 'rb'))

    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = valuePredictor(to_predict_list, model_ensemble, model_qda, model_rf)
        pred = str(result)
        return render_template('result.html', prediction=pred)



if __name__ =='__main__':
    app.run(debug=True)
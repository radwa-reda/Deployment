from flask import Flask, render_template, request
from flask.json import jsonify
import joblib
app = Flask(__name__)

model = joblib.load('lastmymodelRandomForestClassifier.h5')
scaler= joblib.load('lastmyscalerRandomForestClassifier.h5')


@app.route('/', methods=['GET'])
def home():
    return render_template('/Home.html')
@app.route('/index', methods=['GET'])
def pro():
    return render_template('/index.html')
@app.route('/predictAPI', methods=['POST'])
def API():
    gender=float(request.args.get('gender'))
    age=float(request.args.get('age'))
    cigarettes=float(request.args.get('cigarettes'))
    is_high_blood_pressure=float(request.args.get('is_high_blood_pressure'))
    blood_pressure_med_treatment= float(request.args.get('blood_pressure_med_treatment'))
    is_diabetes=float(request.args.get('is_diabetes'))
    total_cholesterol=float(request.args.get('total_cholesterol'))
    systolic_blood_pressure=float(request.args.get('systolic_blood_pressure'))
    Diastolic_b_p=float(request.args.get('Diastolic_b_p'))
    glucose=float(request.args.get('glucose'))
    
   
   
     
    
    #inp_data = [float(n) for n in inp_data]

    predict_disease = model.predict(scaler.transform([[gender,age,cigarettes,is_high_blood_pressure,is_diabetes,blood_pressure_med_treatment,total_cholesterol,systolic_blood_pressure,Diastolic_b_p,glucose]]))
    predict_disease_a= predict_disease.tolist()
    #return render_template('/predict.html', predict_disease=predict_disease)
    return jsonify ({'predict_disease' :predict_disease_a})

@app.route('/predict', methods=['GET'])
def predict():
    gender=float(request.args.get('gender'))
    age=float(request.args.get('age'))
    cigarettes=float(request.args.get('cigarettes'))
    is_high_blood_pressure=float(request.args.get('is_high_blood_pressure'))
    blood_pressure_med_treatment= float(request.args.get('blood_pressure_med_treatment'))
    is_diabetes=float(request.args.get('is_diabetes'))
    total_cholesterol=float(request.args.get('total_cholesterol'))
    systolic_blood_pressure=float(request.args.get('systolic_blood_pressure'))
    Diastolic_b_p=float(request.args.get('Diastolic_b_p'))
    glucose=float(request.args.get('glucose'))
    
    
    #inp_data = [float(n) for n in inp_data]

    predict_disease = model.predict(scaler.transform([[gender,age,cigarettes,is_high_blood_pressure,is_diabetes,blood_pressure_med_treatment,total_cholesterol,systolic_blood_pressure,Diastolic_b_p,glucose]]))
    
    return render_template('/predict.html', predict_disease=predict_disease)
    #return str([[ gender,age,cigarettes,total_cholesterol,systolic_blood_pressure,Diastolic_b_p,is_high_blood_pressure,blood_pressure_med_treatment,glucose,is_diabetes]])

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1')


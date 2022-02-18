from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getdelay',methods=['POST','GET'])
def get_delay():
    if request.method=='POST':
        result=request.form
		
        print(result["gender"])

        lst = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

        data = {'gender':[result["gender"]],
                'SeniorCitizen':[result["SeniorCitizen"]],
                'Partner':[result["Partner"]],
                'Dependents':[result["Dependents"]],
                 'tenure':[result["tenure"]],
                 'PhoneService':[result["PhoneService"]],
                 'MultipleLines':[result["MultipleLines"]],
                 'InternetService':[result["InternetService"]],
                 'OnlineSecurity':[result["OnlineSecurity"]],
                 'OnlineBackup':[result["OnlineBackup"]],
                 'DeviceProtection':[result["DeviceProtection"]],
                 'TechSupport':[result["TechSupport"]],
                 'StreamingTV':[result["StreamingTV"]],
                 'StreamingMovies':[result["StreamingMovies"]],
                 'Contract':[result["Contract"]],
                 'PaperlessBilling':[result["PaperlessBilling"]],
                 'PaymentMethod':[result["PaymentMethod"]],
                 'MonthlyCharges':[result["MonthlyCharges"]],
                 'TotalCharges':[result["TotalCharges"]],
                }

        df = pd.DataFrame(data)
        
        new_dtypes = {"SeniorCitizen": int,
                        "tenure": int,
                    "MonthlyCharges": float,
                    "TotalCharges": float
                        }

        df = df.astype(new_dtypes)
        


        pkl_file = open('pipeline.pkl', 'rb')
        pipeline = pickle.load(pkl_file)
        prediction = pipeline.predict(df)
        proba = round(pipeline.predict_proba(df)[0][1],3)
        
        return render_template('result.html', prediction=prediction, proba=proba)

    
if __name__ == '__main__':
	app.run(debug=True)
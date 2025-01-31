from datetime import datetime
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd


app=Flask(__name__)
#API Endpoint
@app.route('/')
def home():
    return "<h1>Sales Prediction Model</h1>"

# Load the model from the pickle file
with open("gradient_boosting_regression_model_tuned.pkl", "rb") as f:
    loaded_gb_model = pickle.load(f)


@app.route('/predict',methods=['GET','POST']) 
def predict():
    if request.method=='GET':
        return "GET METHOD"
    else:
        req_data=request.get_json()
        Store_Type=req_data['Store_Type']
        Location_Type=req_data['Location_Type']
        Region_Code=req_data['Region_Code']
        Holiday=req_data['Holiday']
        Store_id=req_data['Store_id']
        Discount=0
        if(req_data['Discount']=="Yes"):
            Discount=1
        Date=req_data['Date']
        # Convert string to datetime object
        date_obj = datetime.strptime(Date, "%Y-%m-%d")

        # Extract components
        Year = date_obj.year
        Month = date_obj.month
        Day = date_obj.day
        WeekDay = date_obj.weekday()  # Monday = 0, Sunday = 6
        Is_weekend = 1 if WeekDay >= 5 else 0  # 1 for Saturday & Sunday
        
        input_data=[{
                    "Store_Type": Store_Type,
                    "Location_Type": Location_Type,
                    "Region_Code": Region_Code,
                    "Discount": Discount,
                    "WeekDay": WeekDay,
                    "Month": Month,
                    "Holiday": Holiday,
                    'Store_id': Store_id
                    }]       
        df = pd.DataFrame(input_data)  # Convert to DataFrame

        new_test_preds = loaded_gb_model.predict(df)
        print(new_test_preds)
        return jsonify({"Total Sales Predicted": (new_test_preds*new_test_preds).tolist()})
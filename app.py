from flask import Flask,request,render_template
import pandas as pd
import numpy as np
import pickle
import sklearn
# decisionTree=pickle.load(open('dtr.pkl','rb'))
Cyp=pickle.load(open('rf_model.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))
recommendation = pickle.load(open('crop_recommendation_model.pkl', 'rb'))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
fertilizer_model = pickle.load(open('fertilizer_model.pkl', 'rb'))
fertilizer_label_encoders = pickle.load(open('label_encoders1.pkl', 'rb'))

app=Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')
@app.route("/guide")
def guide():
    return render_template("measurement_guide.html")
@app.route("/predict",methods=['POST'])
def predict():
    if request.method=='POST':
        # Year=int(request.form['Year'])
        average_rain_fall_mm_per_year=float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes=float(request.form['pesticides_tonnes'])
        avg_temp=float(request.form['avg_temp'])
        Area=request.form['Area']
        Item=request.form['Item']
        # features=np.array([[average_rain_fall_mm__per_year,pesticides_tonnes,avg_temp,Area,Item]],dtype=object)
        features = pd.DataFrame([[average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                                columns=['average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])
        transformed_features=preprocessor.transform(features)
        prediction = Cyp.predict(transformed_features)[0]  
        return render_template('index.html',prediction=round(prediction,2))
@app.route("/recommend",methods=['POST'])
def recommend():
    if request.method=='POST':
        N=float(request.form['N'])
        P=float(request.form['P'])
        K=float(request.form['K'])
        temperature=float(request.form['temperature'])
        humidity=float(request.form['humidity'])
        ph=float(request.form['ph'])
        rainfall=float(request.form['rainfall'])
        crop_features=np.array([[N,P,K,temperature,humidity,ph,rainfall]],dtype=float)
        predicted_label=recommendation.predict(crop_features)[0]
        recommended_crop = label_encoder.inverse_transform([predicted_label])[0]
        # recommended_crop=recommendation.predict(crop_features)[0]
        return render_template('index.html',recommended_crop=recommended_crop)
@app.route("/fertilizer", methods=["POST"])
def recommend_fertilizer():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        soil_type = request.form['soil_type']
        crop_type = request.form['crop_type']
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        soil_encoded = fertilizer_label_encoders['Soil Type'].transform([soil_type])[0]
        crop_encoded = fertilizer_label_encoders['Crop Type'].transform([crop_type])[0]
        sample = [[temperature, humidity, moisture, soil_encoded, crop_encoded, N, P, K]]
        predicted_label = fertilizer_model.predict(sample)[0]
        fertilizer_name = fertilizer_label_encoders['Fertilizer Name'].inverse_transform([predicted_label])[0]

        return render_template('index.html', fertilizer_name=fertilizer_name)
if __name__=="__main__":
    app.run(debug=True)
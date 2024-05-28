from flask import Flask,render_template,request
import pickle
import numpy as np

app = Flask(__name__)

with open("one_hot_encoder.pkl", "rb") as f:
    one_hot_encoder = pickle.load(f)

with open("yj.pkl", "rb") as f:
    yeo_johnson_transformer = pickle.load(f)

with open("standard_scaler.pkl", "rb") as f:
    standard_scaler = pickle.load(f)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

categorical_columns = [
    'age_group','hours_per_week_bins','workclass','education','marital.status','occupation','relationship','race','sex'
]

numerical_columns=[
    'fnlwgt_t','education.num_t','net_capital_t'
]

def preprocess_input(data):
    # Extract and transform categorical features
    categorical_features = [data[col] for col in categorical_columns]
    categorical_features = np.array(categorical_features).reshape(1, -1)
    categorical_features_encoded = one_hot_encoder.transform(categorical_features).toarray()

 # Extract and transform numerical features one by one using Yeo-Johnson and then scale them
    numerical_features = [float(data[col]) for col in numerical_columns]
    numerical_features = np.array(numerical_features).reshape(1, -1)
    numerical_features_transformed = np.hstack([
        yeo_johnson_transformer.transform(np.array([x]).reshape(-1, 1)) for x in numerical_features[0]
    ])

    # Extract and transform is_native feature (not one-hot encoded)
    is_native = int(data['is_native'])  
    is_native = np.array(is_native).reshape(1, -1)

    # Combine processed features
    features = np.hstack((is_native,numerical_features_transformed,categorical_features_encoded))
    features_scaled = standard_scaler.transform(features)
    return features_scaled

@app.route('/',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        data=request.form.to_dict()
        processed_data=preprocess_input(data)
        prediction=model.predict(processed_data)
        prediction_result="Income >=50k" if prediction[0]==1 else "Income <50k"
        return render_template('index.html',prediction=prediction_result)
    return render_template('index.html',prediction=None)

if __name__=='__main__':
    app.run(debug=True)
        
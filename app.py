from flask import Flask, request, render_template
import pickle
import os
import cv2

app = Flask(__name__)
upload_path = "upload"
ml_path = "ml"

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/predict',methods=["Get","POST"])
def predict():
    new_file = request.files['file']
    target_path = os.path.join(upload_path, new_file.filename)
    new_file.save(target_path)

    image = cv2.imread(target_path, 0)

    model_path = os.path.join(ml_path, 'model.pkl')
    with open(model_path, 'rb') as file:
        classifier = pickle.load(file)
    prediction = classifier.predict(image)

    return f"Resultado: Es un {prediction.tolist()[0]}"


if __name__ == '__main__':
    if os.path.exists(upload_path) == False:
        os.makedirs(upload_path)
    if os.path.exists(ml_path) == False:
        os.makedirs(ml_path)
    app.run(debug=True, port=5002)

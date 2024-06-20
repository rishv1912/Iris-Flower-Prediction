from flask import Flask,render_template,redirect,url_for,request
import joblib
import pandas as pd
import csv
import os


app = Flask(__name__)

model = joblib.load(open('best_model.pkl','rb'))


@app.route('/')
def index():
    result = ''
    return render_template('index.html',**locals())



CSV_FILE = 'UserDatabse/user_data.csv'

# Ensuring that the directory exists
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)

# Define a function to append data to the CSV file
def append_to_csv(data, filename):
    # Check if the file exists to write the header
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            # Write header only if file does not exist
            writer.writerow(["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])
        writer.writerow(data)



@app.route('/predict',methods=['POST','GET'])
def submit():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]

        # Writing to a CSV file
        append_to_csv([sepal_length, sepal_width, petal_length, petal_width, prediction], CSV_FILE)

 
    
    return render_template('index.html',**locals()) # using **locals() instead of dictionary for context and sending it the frontend is sometimes hectic. It saves the time and automatically sees and send all the data to index.html file





if __name__ == '__main__':
    app.run(debug=True)

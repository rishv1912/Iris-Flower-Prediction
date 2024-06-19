from flask import Flask,render_template,redirect,url_for,request
import joblib
import pandas as pd
import csv


app = Flask(__name__)

model = joblib.load(open('best_model.pkl','rb'))


@app.route('/')
def index():
    result = ''
    return render_template('index.html',**locals())


# user_predicted_values = { 'sepal_length':[],
#                          'sepal_width':[],
#                          'petal_length':[],
#                          'petal_width':[],
#                          'species':[]}


# creating the dataframe for storing later.
# user_data = pd.DataFrame(columns=['sepal_length',
#                          'sepal_width',
#                          'petal_length',
#                          'petal_width',
#                          'species'])


@app.route('/predict',methods=['POST','GET'])
def submit():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]

        # user_predicted_values = { 'sepal_length':[],
        #                  'sepal_width':[],
        #                  'petal_length':[],
        #                  'petal_width':[],
        #                  'species':[]}
        
        user_predicted_values = { 'sepal_length':sepal_length,
                         'sepal_width':sepal_width,
                         'petal_length':petal_length,
                         'petal_width':petal_width,
                         'species':prediction}


        # user_predicted_values['sepal_length'].append(sepal_length)
        # user_predicted_values['sepal_width'].append(sepal_width)
        # user_predicted_values['petal_length'].append(petal_length)
        # user_predicted_values['petal_width'].append(petal_length)
        # user_predicted_values['species'].append(prediction)

        # user_predicted_dataframe = pd.DataFrame(user_predicted_values)
        # pd.concat([user_data,user_predicted_dataframe],axis=1)
        # print(user_data)

        filename = "UserDatabase/user_data.csv"

        # Writing to a CSV file
        with open(filename, mode='w', newline='') as file:
            # Create a writer object
            writer = csv.DictWriter(file, fieldnames=["sepal_length", "sepal_width", "petal_length","petal_width","species"])

            # Write the header
            writer.writeheader()

            # Write the data
            for row in user_predicted_values:
                writer.writerow(row)

        print(f"Data has been written to {filename}")




        
        
        
 
    # using **locals() instead of dictionary for context and sending it the frontend is sometimes hectic. It saves the time and automatically sees and send all the data to index.html file

    return render_template('index.html',**locals())

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask,render_template,redirect,url_for,request
import joblib

app = Flask(__name__)

model = joblib.load(open('best_model.pkl','rb'))


@app.route('/')
def index():
    result = ''
    return render_template('index.html',**locals())


@app.route('/predict',methods=['POST','GET'])
def submit():
    if request.method == 'POST':
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]


    return render_template('index.html',**locals())

if __name__ == '__main__':
    app.run(debug=True)

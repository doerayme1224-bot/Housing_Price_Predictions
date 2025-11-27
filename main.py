import pickle # imports pickle, to use the saved model
from flask import Flask,request,app, jsonify, render_template # import flask, we are making a flask app which is why
import numpy as np # numpy, we use the library in the code below
import pandas as pd # imports pandas, also used in the below code

app=Flask(__name__) # instantiating the flask app with the __name__ as a parameter in order to (in simple terms) create a web app

## Load the model
regmodel=pickle.load(open('pkl_file','rb')) # we open the pkl_file model that we created earlier, using pickle, and store it in the var `regmodel` for later use

@app.route('/') # first app for the web app
def home(): # creates a function that renders the home.html file for the web app when called
    return render_template("base.html")

@app.route('/predict_api', methods=['post']) # second part of the web app, which creates a REST API endpoint for a web app 'accepts an http pull request'
def predict_api(): # creates a function that works when a post request is sent to the endpoint
    data=request.json['data'] # parses the JSON data into a pythoin dictionary and stores it in data
    print(data)
    input=np.array(list(data.values())).reshape(1,-1) # gets a list from the request, converts the values to a list then a NumPy array, then reshapes the arrays to be two dimensional
    output=regmodel.predict(input) # predictss the data given from the post request data to make a prediction on the data
    print(output[0]) # takes teh prediction of the model from the array, and prints it's output
    return jsonify(output[0]) # returns the prediction from the data back to the client

@app.route('/predict', methods=['POST']) # tells flask to run the function below whenever a POST request is made
def predict(): # the function
    input=[float(x) for x in request.form.values()] # makes a python list of the user input values
    print(input)
    input_array = np.array(input).reshape(1, -1) # converts the list to a numpy array and reshapes the array into a 2 dimensional array
    output=regmodel.predict(input_array)[0] # where the prediction of the model happens and accesses only the preedicted value
    return render_template("base.html", prediction_test="The estimated cost of the house is {} USD".format(output)) # using the home template, returns a predicted value in a string with a place holder, and formats it as an output

if __name__=="__main__":
    app.run(debug=True)
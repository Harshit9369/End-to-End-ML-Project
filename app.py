# Flask App Routing

from flask import Flask

app = Flask(__name__)

@app.route("/", methods=["GET"])
def welcome():
    return "<h1>Welcome to Flask App!</h1>"
@app.route("/index", methods=["GET"]) # by default method is GET only. 
def index():
    return "Welcome to Index Page!"

@app.route("/success")
def success():
    return "Welcome to Success Page!"



if __name__ == "__main__": # entry point for the program
    app.run(debug=True) # run the flask app in debug mode(reload automatically)
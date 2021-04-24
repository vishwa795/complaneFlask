from flask import Flask

from fastText import ft;

app = Flask(__name__)

@app.route('/fastText')
def fastText():
    return "This is fastText"



@app.route('/keyBert')
def keyBert():
    return "This is keyBert"



@app.route('/postitionRank')
def positionRank():
    return "This is positionRank"



@app.route('/annoy')
def annoy():
    return "This is Annoy"

if __name__ == "__main__":
    app.run(debug=True)
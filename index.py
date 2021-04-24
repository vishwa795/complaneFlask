from flask import Flask

import fasttext

app = Flask(__name__)

x = 4
y =3+x

@app.route('/fastText')
def fastText():
    #use fastText here
    return "This is fastText "+str(y)



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
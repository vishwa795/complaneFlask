from flask import Flask, request

import fasttext
from keybert import KeyBERT
import json

app = Flask(__name__)

x = 4
y =3+x
keyBERT_model = KeyBERT('distilbert-base-nli-mean-tokens')

@app.route('/fastText')
def fastText():
    #use fastText here
    return "This is fastText "+str(y)



@app.route('/keyBert', methods = ['POST'])
def keyBert():
    if request.method == 'POST':
        doc = request.form['doc']
        keywords = keyBERT_model.extract_keywords(doc)
        return json.dumps(keywords)



@app.route('/postitionRank')
def positionRank():
    return "This is positionRank"



@app.route('/annoy')
def annoy():
    return "This is Annoy"

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request

import fasttext
from keybert import KeyBERT
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import json

app = Flask(__name__)

keyBERT_model = KeyBERT('distilbert-base-nli-mean-tokens')
BERT_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

embed_dim = 768
tree_Telecom = AnnoyIndex(embed_dim, "angular")   # Department_of_Telecommunications
tree_IncomeTax = AnnoyIndex(embed_dim, "angular") # Central_Board_of_Direct_Taxes_(Income_Tax)
tree_Labour = AnnoyIndex(embed_dim, "angular")    # Ministry_of_labour_and_Employment
tree_Finance = AnnoyIndex(embed_dim, "angular")   # Department_of_Financial_Services_(Banking_Division)
tree_Welfare = AnnoyIndex(embed_dim, "angular")   # Department_of_Ex_Servicemen_Welfare
tree_IndirectTax = AnnoyIndex(embed_dim, "angular")       # Central_Board_of_Indirect_Taxes_and_Customs

def treeBuild(n = 20):
    tree_Telecom.build(n)
    tree_IncomeTax.build(n)
    tree_Labour.build(n)
    tree_Finance.build(n)
    tree_Welfare.build(n)
    tree_IndirectTax.build(n)

def treeUnbuild():
    tree_Telecom.unbuild()
    tree_IncomeTax.unbuild()
    tree_Labour.unbuild()
    tree_Finance.unbuild()
    tree_Welfare.unbuild()
    tree_IndirectTax.unbuild()

treeBuild()

def getKeywordList(KeywordsObject):
    keywordsList = []
    for keyword_map in KeywordsObject:
        print(keyword_map)
        keywordsList.append(keyword_map["keyword"])
    return keywordsList


@app.route('/fastText', methods = ['POST'])
def fastText():
    #use fastText here
    return "This is fastText "



@app.route('/keyBert', methods = ['POST'])
def keyBert():
    if request.method == 'POST':
        doc = request.form['doc']
        keywords = keyBERT_model.extract_keywords(doc)
        return json.dumps(keywords)



@app.route('/multipartiteRank', methods = ['POST'])
def multipartiteRank():
    return "This is multipartiteRank"


@app.route('/annoy/train', methods = ['POST'])
def annoyTrain():
    if request.method == 'POST':

        # [
        #     {
        #     "keyword": "tax",
        #     "department": "Central_Board_of_Direct_Taxes_(Income_Tax)",
        #     "index": 10
        #     },
        #     {
        #         "keyword": "refunds",
        #         "department": "Central_Board_of_Direct_Taxes_(Income_Tax)",
        #         "index": 11
        #     }
        # ]
        

        data = json.dumps(request.json)
        keyword_set_array = json.loads(data)

        keyword_list = getKeywordList(keyword_set_array)

        embedding_bert=BERT_model.encode(keyword_list)

        treeUnbuild()

        for i in range(len(keyword_set_array)):
            if(keyword_set_array[i]["department"]=="Department_of_Telecommunications"):
                tree_Telecom.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Central_Board_of_Direct_Taxes_(Income_Tax)"):
                tree_IncomeTax.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Ministry_of_labour_and_Employment"):
                tree_Labour.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Department_of_Financial_Services_(Banking_Division)"):
                tree_Finance.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Department_of_Ex_Servicemen_Welfare"):
                tree_Welfare.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Central_Board_of_Indirect_Taxes_and_Customs"):
                tree_IndirectTax.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])
        
        
        treeBuild(20)
        success = {
            "status":"success"
        }
        return json.dumps(success)


@app.route('/annoy/findIdentical', methods = ['POST'])
def annoyFindIdentical():
    if request.method == 'POST':
        # {
        #     "keyword": "tax"
        #     "department": "Central_Board_of_Direct_Taxes_(Income_Tax)"
        # }
        keyword_map = request.json
        keytosearch = BERT_model.encode([keyword_map["keyword"]])
        relatedKeywordIndexes = []
        if(keyword_map["department"]=="Department_of_Telecommunications"):
           relatedKeywordIndexes = tree_Telecom.get_nns_by_vector(keytosearch[0],10)

        elif(keyword_map["department"]=="Central_Board_of_Direct_Taxes_(Income_Tax)"):
            relatedKeywordIndexes = tree_IncomeTax.get_nns_by_vector(keytosearch[0],10)

        elif(keyword_map["department"]=="Ministry_of_labour_and_Employment"):
            relatedKeywordIndexes = tree_Labour.get_nns_by_vector(keytosearch[0],10)

        elif(keyword_map["department"]=="Department_of_Financial_Services_(Banking_Division)"):
            relatedKeywordIndexes = tree_Finance.get_nns_by_vector(keytosearch[0],10)

        elif(keyword_map["department"]=="Department_of_Ex_Servicemen_Welfare"):
            relatedKeywordIndexes = tree_Welfare.get_nns_by_vector(keytosearch[0],10)
            
        elif(keyword_map["department"]=="Central_Board_of_Indirect_Taxes_and_Customs"):
            relatedKeywordIndexes = tree_IndirectTax.get_nns_by_vector(keytosearch[0],10)
        
        return json.dumps(relatedKeywordIndexes)


if __name__ == "__main__":
    app.run(debug=True)
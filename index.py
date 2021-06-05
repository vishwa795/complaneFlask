from flask import Flask, request

import fasttext
from flask.wrappers import Response
from keybert import KeyBERT
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import json
import fasttext
from torch.functional import tensordot
import pke
import uuid
import string
from nltk.corpus import stopwords
import os

app = Flask(__name__)

keyBERT_model = KeyBERT('distilbert-base-nli-mean-tokens')
BERT_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

embed_dim = 768
tree_Telecom = AnnoyIndex(embed_dim,'angular')
tree_Telecom.load('models/annoy/tree_Telecom.ann') # Department_of_Telecommunications

tree_IncomeTax = AnnoyIndex(embed_dim,'angular')# Central_Board_of_Direct_Taxes_(Income_Tax)
tree_IncomeTax.load('models/annoy/tree_IncomeTax.ann')

tree_Labour = AnnoyIndex(embed_dim,'angular')    # Ministry_of_labour_and_Employment
tree_Labour.load('models/annoy/tree_Labour.ann')

tree_Finance = AnnoyIndex(embed_dim,'angular')   # Department_of_Financial_Services_(Banking_Division)
tree_Finance.load('models/annoy/tree_Finance.ann')

tree_Welfare = AnnoyIndex(embed_dim,'angular')   # Department_of_Ex_Servicemen_Welfare
tree_Welfare.load('models/annoy/tree_Welfare.ann')

tree_IndirectTax = AnnoyIndex(embed_dim,'angular') # Central_Board_of_Indirect_Taxes_and_Customs
tree_IndirectTax.load('models/annoy/tree_IndirectTax.ann')


classification_model = fasttext.load_model('models/fastText/fasttext_Without_Others.bin')

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

def treeUnload():
    tree_Telecom.unload()
    tree_IncomeTax.unload()
    tree_Labour.unload()
    tree_Finance.unload()
    tree_Welfare.unload()
    tree_IndirectTax.unload()

def saveAnnoyTrees():
    os.remove('models/annoy/tree_Telecom.ann')
    tree_Telecom.save('models/annoy/tree_Telecom.ann')
    tree_IncomeTax.save('models/annoy/tree_IncomeTax.ann')
    tree_Labour.save('models/annoy/tree_Labour.ann')
    tree_Finance.save('models/annoy/tree_Finance.ann')
    tree_Welfare.save('models/annoy/tree_Welfare.ann')
    tree_IndirectTax.save('models/annoy/tree_IndirectTax.ann')

def getKeywordList(KeywordsObject):
    keywordsList = []
    for keyword_map in KeywordsObject:
        print(keyword_map)
        keywordsList.append(keyword_map["keyword"])
    return keywordsList


@app.route('/fastText', methods = ['POST'])
def fastText():
    # {
    # "complaint":"First, document embeddings are extracted with BERT to get a document-level representation. 
    # Then, word embeddings are extracted for N-gram words/phrases. Finally, we use cosine similarity to find
    #  the words/phrases that are the most similar to the document",
    #  ... any other sent data will be returned back in response as it is
    # }
    if(request.method == 'POST'):
        print(request.json)
        data = request.json
        complaint = data["complaint"]
        target = classification_model.predict(complaint)
        target_dept = target[0][0]
        target_dept = target_dept.replace("__label__","")
        confidence_score = target[1][0]
        data["department_predicted"] = target_dept
        data["predicted_confidence"] = str(confidence_score)
        response = Response()
        response.status_code = 200;
        response.content_type = "application/json"
        response.data = json.dumps(data)
        return response
    return "This is fastText "


@app.route('/extractKeywords', methods = ['POST'])
def keywordExtractor():
    # {
    # "complaint":"First, document embeddings are extracted with BERT to get a document-level representation. 
    # Then, word embeddings are extracted for N-gram words/phrases. Finally, we use cosine similarity to find
    #  the words/phrases that are the most similar to the document"
    #  ... any other sent data will be returned back in response as it is
    # }
    if request.method == 'POST':
        data = request.json
        complaint = data['complaint']

        filename =str(uuid.uuid1())+".txt"
        f = open(filename,"w")
        f.write(complaint)
        f.close()

        # 1. create a MultipartiteRank extractor.
        extractor = pke.unsupervised.MultipartiteRank()

        # 2. load the content of the document.
        extractor.load_document(input=filename)

        # 3. select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        pos = {'NOUN', 'PROPN', 'ADJ'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-','sir','please']
        stoplist += stopwords.words('english')
        additional_stoplist = ['a', 'able', 'about', 'above', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'again', 'against', 'ah', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apparently', 'approximately', 'are', 'aren', 'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'at', 'auth', 'available', 'away', 'awfully', 'b', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'between', 'beyond', 'biol', 'both', 'brief', 'briefly', 'but', 'by', 'c', 'ca', 'came', 'can', 'cannot', "can't", 'cause', 'causes', 'certain', 'certainly', 'co', 'com', 'come', 'comes', 'contain', 'containing', 'contains', 'could', 'couldnt', 'd', 'date', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', 'done', "don't", 'down', 'downwards', 'due', 'during', 'e', 'each', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 'especially', 'et', 'et-al', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'except', 'f', 'far', 'few', 'ff', 'fifth', 'first', 'five', 'fix', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'found', 'four', 'from', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'h', 'had', 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes', 'hi', 'hid', 'him', 'himself', 'his', 'hither', 'home', 'how', 'howbeit', 'however', 'hundred', 'i', 'id', 'ie', 'if', "i'll", 'im', 'immediate', 'immediately', 'importance', 'important', 'in', 'inc', 'indeed', 'index', 'information', 'instead', 'into', 'invention', 'inward', 'is', "isn't", 'it', 'itd', "it'll", 'its', 'itself', "i've", 'j', 'just', 'k', 'keep\tkeeps', 'kept', 'kg', 'km', 'know', 'known', 'knows', 'l', 'largely', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'lets', 'like', 'liked', 'likely', 'line', 'little', "'ll", 'look', 'looking', 'looks', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'mg', 'might', 'million', 'miss', 'ml', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'mug', 'must', 'my', 'myself', 'n', 'na', 'name', 'namely', 'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'now', 'nowhere', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'ord', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'page', 'pages', 'part', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 'put', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'respectively', 'resulted', 'resulting', 'results', 'right', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sec', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sent', 'seven', 'several', 'shall', 'she', 'shed', "she'll", 'shes', 'should', "shouldn't", 'show', 'showed', 'shown', 'showns', 'shows', 'significant', 'significantly', 'similar', 'similarly', 'since', 'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure\tt', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that've", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'thered', 'therefore', 'therein', "there'll", 'thereof', 'therere', 'theres', 'thereto', 'thereupon', "there've", 'these', 'they', 'theyd', "they'll", 'theyre', "they've", 'think', 'this', 'those', 'thou', 'though', 'thoughh', 'thousand', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'v', 'value', 'various', "'ve", 'very', 'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wants', 'was', 'wasnt', 'way', 'we', 'wed', 'welcome', "we'll", 'went', 'were', 'werent', "we've", 'what', 'whatever', "what'll", 'whats', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whim', 'whither', 'who', 'whod', 'whoever', 'whole', "who'll", 'whom', 'whomever', 'whos', 'whose', 'why', 'widely', 'willing', 'wish', 'with', 'within', 'without', 'wont', 'words', 'world', 'would', 'wouldnt', 'www', 'x', 'y', 'yes', 'yet', 'you', 'youd', "you'll", 'your', 'youre', 'yours', 'yourself', 'yourselves', "you've", 'z', 'zero']
        stoplist += additional_stoplist
        extractor.candidate_selection(pos=pos, stoplist=stoplist)

        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1,
                                    threshold=0.74,
                                    method='average')

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=5)
        keywords = keyBERT_model.extract_keywords(complaint,stop_words=stoplist)
        list_keywords = []
        for i in range(len(keywords)):
            list_keywords.append(keywords[i][0])
        for i in range(len(keyphrases)):
            list_keywords.append(keyphrases[i][0])
        
        print(list_keywords)
        result = list(set(list_keywords))
        data['keywords'] = result

        response = Response()
        response.content_type = "application/json"
        response.data = json.dumps(data)
        os.remove(filename)
        return response
    return "ONLY POST METHOD IS ALLOWED"
        


@app.route('/annoy/addKeywords', methods = ['POST'])
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

        
        tree_Telecom.save('models/annoy/tree_Telecom.ann')
        tree_IncomeTax.save('models/annoy/tree_IncomeTax.ann')
        tree_Labour.save('models/annoy/tree_Labour.ann')
        tree_Finance.save('models/annoy/tree_Finance.ann')
        tree_Welfare.save('models/annoy/tree_Welfare.ann')
        tree_IndirectTax.save('models/annoy/tree_IndirectTax.ann')
        success = {
            "status":"success"
        }
        response = Response()
        response.content_type = "application/json"
        response.data = json.dumps(success)
        response.status_code = 200
        return response

@app.route('/annoy/retrain', methods = ['POST'])
def annoyRetrain():

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
        

        keyword_set_array = request.json

        keyword_list = getKeywordList(keyword_set_array)

        embedding_bert=BERT_model.encode(keyword_list)

        tree_Finance_new = AnnoyIndex(embed_dim,"angular")
        tree_IncomeTax_new = AnnoyIndex(embed_dim,"angular")
        tree_Labour_new = AnnoyIndex(embed_dim,"angular")
        tree_Welfare_new = AnnoyIndex(embed_dim,"angular")
        tree_Telecom_new = AnnoyIndex(embed_dim,"angular")
        tree_IndirectTax_new = AnnoyIndex(embed_dim,"angular")

        for i in range(len(keyword_set_array)):
            if(keyword_set_array[i]["department"]=="Department_of_Telecommunications"):
                tree_Telecom_new.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Central_Board_of_Direct_Taxes_(Income_Tax)"):
                tree_IncomeTax_new.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Ministry_of_labour_and_Employment"):
                tree_Labour_new.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Department_of_Financial_Services_(Banking_Division)"):
                tree_Finance_new.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Department_of_Ex_Servicemen_Welfare"):
                tree_Welfare_new.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])

            elif(keyword_set_array[i]["department"]=="Central_Board_of_Indirect_Taxes_and_Customs"):
                tree_IndirectTax_new.add_item(int(keyword_set_array[i]["index"]), embedding_bert[i])
        
        tree_Telecom_new.build(20)
        tree_IncomeTax_new.build(20)
        tree_Labour_new.build(20)
        tree_Finance_new.build(20)
        tree_Welfare_new.build(20)
        tree_IndirectTax_new.build(20)

        global tree_Finance,tree_IncomeTax,tree_Labour,tree_Telecom,tree_Welfare,tree_IndirectTax
        tree_Telecom.unload()
        tree_IncomeTax.unload()
        tree_Labour.unload()
        tree_Finance.unload()
        tree_Welfare.unload()
        tree_IndirectTax.unload()

        tree_Telecom = 0


        tree_Finance = tree_Finance_new
        tree_IncomeTax = tree_IncomeTax_new
        tree_Labour = tree_Labour_new
        tree_Telecom = tree_Telecom_new
        tree_Welfare = tree_Welfare_new
        tree_IndirectTax = tree_Finance_new

        
        success = {
            "status":"success"
        }
        response = Response()
        response.content_type = "application/json"
        response.data = json.dumps(success)
        response.status_code = 200
        return response


         
        


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
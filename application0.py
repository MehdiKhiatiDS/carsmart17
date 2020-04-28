from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib 
from joblib import load
import numpy as np
import pandas as pd


# loAd pipeline

pipeline = load('assets/pipeline3.joblib')
print('Model Loaded!')
# EB looks for an 'application' callable by default.
application = Flask(__name__)
@application.route('/')
def home():
        return "This is a RESTful web service! Append a model to the URL (for example: <code>/FerrariEnzo</code>) to predict the price of a car! neat huh"

# @application.route('/<user_input>') 
# def predict(user_input):
#         # model = joblib.load('assets/pipeline1.joblib')
#         return pipeline.predict(user_input)
    
@application.route('/test', methods=['POST'])
def test():
    # data = request.get_json(force=True)
    # values = data.values()
    # values = list(values)
    # values = np.array(values)
    # pipe = pipeline.predict([values])
   
    # preds = pipeline.predict([np.array(list(data.values()))])
    # data_values = data.values()
    # output = preds[0] 
    # return values
    # return str(values)
    data = {
    "Year":3,
    "Milleage":12000,
    "City":"San Diego",
    "State":"CA",
    "Vin":"9VFEE4TG",
    "Make":"Acura",
    "Model":"L\ILX6"
    }
    year=data['Year']
    milleage = data['Milleage']
    city = data['City']
    state = data['State']
    vin = data['Vin']
    make = data['Make']
    model = data['Model']
    df = pd.DataFrame({
    'Year':[year],
    "Milleage":[milleage],
    "City":[city],
    "State":[state],
    "Vin":[vin],
    "Make":[make],
    "Model":[model]
    })
    df
    # data = request.get_json(force=True)
    # data = pd.DataFrame.from_dict(data)
    # data = {'col_1': ['Year'], 'col_2': ['Mileage'], 'col3':['City'], 'col4':['State'], 'col5':['Vin'], 'col6':['Make'], 'col7':['Model']}
    # data = pd.DataFrame.from_dict(data)
    preds = pipeline.predict([data])
    return data
  



# ['Price', 'Year', 'Mileage', 'City', 'State', 'Vin', 'Make', 'Model']




    # return jsonify(preds)   
        
# data.values()
# def predict_user(user1_name, user2_name, tweet_text):
#     """Determine and return which user is more likely to say a given Tweet."""

#     user1 = User.query.filter(User.name == user1_name).one()
#     user2 = User.query.filter(User.name == user2_name).one()
#     user1_embeddings = np.array([tweet.embedding for tweet in user1.tweets])
#     user2_embeddings = np.array([tweet.embedding for tweet in user2.tweets])
#     embeddings = np.vstack([user1_embeddings, user2_embeddings])
#     labels = np.concatenate([np.ones(len(user1.tweets)),
#                              np.zeros(len(user2.tweets))])
    
#     knnc = KNeighborsClassifier(weights='distance', metric='cosine').fit(embeddings, labels)
#     tweet_embedding = BASILICA.embed_sentence(tweet_text, model='twitter')
#     return knnc.predict(np.array(tweet_embedding).reshape(1, -1))
        # price_predicted = [(preds['model'] == preds[1][0][0])] 
        # dict_set = [{
        # 'pipeline' : x[0]
    
        # }
    #     for x in preds[['model']].values]

    #     json_preds = json.dumps(dict_set)

    #     return json_preds
    # return app

# print a nice greeting.
# def say_hello(username = "World"):
#     return '<p>Hello %s!</p>\n' % username

# some bits of text for the page.
# header_text = '''
#     <html>\n<head> <title>EB Flask Test</title> </head>\n<body>'''
# instructions = '''
#     <p><em>Hint</em>: This is a RESTful web service! Append a 
#     to the URL (for example: <code>/19000</code>) to predict the price
#     of a car.</p>\n'''
# home_link = '<p><a href="/">Back</a></p>\n'
# footer_text = '</body>\n</html>'

# EB looks for an 'application' callable by default.
# application = Flask(__name__)

# add a rule for the index page.
# application.add_url_rule('/', 'index', (lambda: header_text +
    # say_hello() + instructions + footer_text))

# add a rule when the page is accessed with a name appended to the site
# URL.
# application.add_url_rule('/<username>', 'hello', (lambda username:
    # header_text + say_hello(username) + home_link + footer_text))

# run the app.
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    application.debug = True
    application.run()
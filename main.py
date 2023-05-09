from datetime import datetime
from flask import Flask, render_template, request, url_for
import pickle
import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame

nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords

from flask_sqlalchemy import SQLAlchemy

popular_hotels = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
hotels = pickle.load(open('hotels.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))
final_hotels =pickle.load(open('final_hotels.pkl','rb'))

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/codebase'
db = SQLAlchemy(app)






class Contacts(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    phone_num = db.Column(db.String(12), nullable=False)
    mes = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)
    email = db.Column(db.String(20), nullable=False)

@app.route('/contact',methods = ['GET', 'POST'])
def contact():
    if (request.method== 'POST'):
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('phone')
        message = request.form.get('message')
        entry = Contacts(name=name, phone_num = phone, mes = message, date= datetime.now(),email = email )
        db.session.add(entry)
        db.session.commit()
    return render_template('contact.html')

@app.route('/')
def index():
    return render_template('newindex.html',
                           hotel_name=list(popular_hotels['hotel_name'].values),
                           avg_rating=list(popular_hotels['avg_rating'].values),
                           hotel_experience=list(popular_hotels['hotel_experience'].values),
                           address=list(popular_hotels['address'].values),
                           country=list(popular_hotels['country'].values)
                           )


@app.route('/Recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_hotels', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    index = np.where(pt.index == user_input)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []

    for i in similar_items:
        item = []
        temp_df = hotels[hotels['hotel_name'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('hotel_name')['hotel_name'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['hotel_rating'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['hotel_experience'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['address'].values))
        item.extend(list(temp_df.drop_duplicates('hotel_name')['country'].values))

        data.append(item)

        print(data)

    #print(data)

    return render_template('recommend.html', data=data)



@app.route('/about')
def about():
    return render_template('about.html')




@app.route('/map')
def map():
    return render_template('hotel.html')


@app.route('/airport')
def airport():
    return render_template('airports.html')
#print(final_hotels)

@app.route('/RecommendNew')
def recommend_ui1():
    return render_template('recommend_new.html')


@app.route('/recommend_finalhotels', methods=['POST'])
def recommend1():
    query1 = request.form.get('user_input1')
    query2 = request.form.get('user_input2')

    data = []

    final_hotels = pd.read_csv('final_hotels.csv')

    amenities = final_hotels['amenities']
    address = final_hotels['address']
    avg_rating = final_hotels['avg_rating']

    # Create a TF-IDF vectorizer object
    tfidf_vectorizer = TfidfVectorizer()
    tfidf =  TfidfVectorizer()

    # Fit the vectorizer on the dataset
    tfidf_vectorizer.fit(amenities)
    tfidf_vectorizer.fit(address)
    #tfidf.fit_transform(avg_rating)

    # Transform the dataset to TF-IDF features
    X = tfidf_vectorizer.transform(amenities)
    Y = tfidf_vectorizer.transform(address)


    # # Example text queries

    # query1 = "'Pool', 'Restaurant', 'Fitness Centre with Gym / Workout Room', 'Spa', 'Room service', 'Bar/Lounge', 'Banquet Room', 'Breakfast Available', 'Business Centre with Internet Access', 'Concierge', 'Conference Facilities', 'Dry Cleaning', 'Heated pool', 'Hot Tub', 'Indoor pool', 'Laundry Service', 'Meeting rooms', 'Multilingual Staff', 'Non-smoking hotel', 'Paid Internet', 'Paid Wifi', 'Public Wifi', 'Wheelchair Access', 'Family Rooms', 'Non-smoking rooms', 'Suites'"
    # query2 = "6740 Fallsview Blvd Niagara Falls Ontario"

    #a = X.toarray()
    #tfidf_vectorizer.get_feature_names()
    query1 = query1.translate(str.maketrans('', '', string.punctuation))
    query2 = query2.translate(str.maketrans('', '', string.punctuation))

    stop_words = set(stopwords.words('english'))

    # Tokenize the text
    token1 = nltk.word_tokenize(query1)
    token2 = nltk.word_tokenize(query2)

    # Remove stop words
    filtered_token1 = [token for token in token1 if token.lower() not in stop_words]
    filtered_token2 = [token for token in token2 if token.lower() not in stop_words]

    # Join the filtered tokens back into a string
    filtered_text1 = " ".join(filtered_token1)
    filtered_text2 = " ".join(filtered_token2)

    # Transform the queries to TF-IDF features
    query1_vec = tfidf_vectorizer.transform([filtered_text1])
    query2_vec = tfidf_vectorizer.transform([filtered_text2])


    user_amenities_score = query1_vec.toarray().flatten()
    user_address_score = query2_vec.toarray().flatten()
    user_feature_vector = np.concatenate([user_amenities_score, user_address_score])

    # Calculate the cosine similarity between the queries and the dataset
    cosine_similarities_query1 = cosine_similarity(query1_vec, X)
    cosine_similarities_query2 = cosine_similarity(query2_vec, Y)

    # Print the cosine similarities
    #print(cosine_similarities_query1[0])
    #print(cosine_similarities_query2[0])
    similarity_scores = cosine_similarity(pt)
    final_hotels['similarity_amenities'] = cosine_similarities_query1[0]
    final_hotels['similarity_address'] = cosine_similarities_query2[0]

    weights = [0.4, 0.6]

    # Create a new column with the weighted average of the three ratings
    final_hotels['similarity scores'] = (final_hotels['similarity_amenities'] * weights[0] +
                                    final_hotels['similarity_address'] * weights[1]) / sum(weights)

    final_hotels = final_hotels.sort_values(by='similarity scores', ascending=False)
    final_hotels = final_hotels.drop(['similarity_amenities','similarity_address'],axis = 1)

    #similarity_scoress = final_hotels['similarity_scores']
    #content based recommendations
    top_hotels = final_hotels.nlargest(50, 'similarity scores')['hotel_name'].tolist()
    top_hotel = final_hotels.nlargest(1, 'similarity scores')['hotel_name'].values[0]
    top_hotel1 = str(top_hotel)
    final = set(top_hotels)
    final = list(final)

    # print(query1, query2)
    print(final)

    # final - Hotel names
    # iterate each hotel - rating, ameti.... = list(variable) = data

    return render_template('recommend_new.html', data=final)


app.run(debug=True)
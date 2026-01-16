from flask import Flask, request, jsonify
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_cors import CORS, cross_origin
import requests

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Funzione per ottenere i nomi dei paesi dall'API Restcountries
def get_country_names():
    try:
        response = requests.get('https://restcountries.com/v3.1/all')
        data = response.json()
        country_names = [country['name']['common'] for country in data]
        return country_names
    except Exception as e:
        print(f"Errore durante la richiesta dell'API: {e}")
        return []

# Funzione per generare suggerimenti di ricerca basati sui nomi dei paesi
def generate_search_suggestions(query):
    country_names = get_country_names()

    # Filtra i nomi dei paesi che iniziano con la query di ricerca
    suggestions = [country for country in country_names if country.lower().startswith(query.lower())]

    return suggestions[:5]  # Restituisci solo i primi 5 suggerimenti

# Endpoint per la ricezione delle richieste dal frontend
@app.route('/predict', methods=['POST', 'OPTIONS'])  # Aggiungiamo 'OPTIONS' tra i metodi supportati
@cross_origin()
def predict():
    print("ciao sono dentro predict")
    if request.method == 'OPTIONS':
        # Se il metodo è OPTIONS, restituisci una risposta vuota con i giusti header
        # Restituisci una risposta vuota con gli header CORS appropriati
        # response = app.make_response()
        response = jsonify({'some': 'data'})
        response.headers['Access-Control-Allow-Origin'] = '*'  # Abilita CORS per tutte le origini
        response.headers['Access-Control-Allow-Methods'] = 'POST'  # Specifica i metodi consentiti
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Specifica gli header consentiti
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    if request.method == 'POST':
        data = request.get_json()
        query = data['search_text']  # Ottieni la query di ricerca dalla richiesta

        # Genera suggerimenti di ricerca basati sulla query
        suggestions = generate_search_suggestions(query)

        # Restituisci i suggerimenti al frontend
        return jsonify({'suggestions': suggestions})

# Funzione per caricare il modello SVM serializzato
def load_model():
    # Determina il percorso del file pkl
    file_path = os.path.join(os.getcwd(), 'modello_svm.pkl')
    print(file_path)

    
    # Carica il modello serializzato
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Funzione per caricare il vettorizzatore TF-IDF
def load_tfidf_vectorizer():
    # Ottieni il percorso della directory del modulo
    module_dir = os.path.dirname(__file__)
    # Costruisci il percorso completo del file pkl
    file_path = os.path.join(module_dir, 'tfidf_vectorizer (2).pkl')
    
    # Carica il vettorizzatore TF-IDF serializzato
    with open(file_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


# Funzione per classificare un commento
def classify_comment_from_string(comment, model, vectorizer):
    # Trasforma il commento in un vettore TF-IDF
    comment_vector = vectorizer.transform([comment])
    
    # Applica il modello per ottenere la previsione
    prediction = model.predict(comment_vector)[0]
    
    # Interpretazione della previsione
    if prediction == 1:
        result = "razzista"
    else:
        result = "non razzista"
    print("prediction: {} result: {}".format(prediction,result))
    return prediction

# Funzione per il test manuale
def manual_test(model, vectorizer):
    # Esegui il test manuale
    comment = input("Inserisci il commento da classificare: ")
    prediction = classify_comment_from_string(comment, model, vectorizer)

    # Stampa il risultato
    print("Il commento '{}' è classificato come: {}".format(comment, prediction))

# Endpoint per la classificazione del commento
@app.route('/classify', methods=['POST', 'OPTIONS'])
@cross_origin()
def classify_comment():
    print('Message received')

    data = request.get_json()
    comment = data['comment']
    
    # Carica il modello SVM
    model = load_model()

    # Carica il vettorizzatore TF-IDF
    vectorizer = load_tfidf_vectorizer()

    # Classifica il commento
    prediction = classify_comment_from_string(comment, model, vectorizer)

    # Crea la risposta da inviare al client
    response = jsonify({'comment': comment, 'prediction': prediction})

    print('Comment "{}" is classified as {}'.format(comment, prediction))

    return response


if __name__ == '__main__':

    app.run(debug=True, port=5000)

    

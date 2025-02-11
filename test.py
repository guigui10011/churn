import requests
import json

# Définir l'URL de l'API
URL = "http://127.0.0.1:5000/predict"

# Données d'entrée pour le test
test_data = {
    "Age": 40,
    "Account_Manager": 0,
    "Years": 5,
    "Num_Sites": 20
}

# Envoyer la requête POST
response = requests.post(URL, json=test_data)

# Vérifier la réponse
if response.status_code == 200:
    print("Test réussi ✅")
    print("Réponse de l'API:", json.dumps(response.json(), indent=4))
else:
    print("Échec du test ❌")
    print("Code d'erreur:", response.status_code)
    print("Message:", response.text)
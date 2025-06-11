import pandas as pd
import re

# Reload the CSV
file_path = "C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Legy Data/Collaborative_Filtering/Data/cleaned_dakar_reviews01.csv"
df = pd.read_csv(file_path)
df = df.dropna(subset=['Review']).reset_index(drop=True)

# Redefine the preference keywords
preference_keywords = {
    'cares_about_cleanliness': [
        'propre', 'hygiène', 'sale', 'saleté', 'nettoyage', 'immaculé', 'bactéries', 'désinfecté', 'malpropre'
    ],
    
    'cares_about_service_speed': [
        'rapide', 'attente', 'lent', 'service rapide', 'retard', 'long', 'tardif', 'immédiat', 'servi rapidement'
    ],
    
    'cares_about_price': [
        'prix', 'abordable', 'cher', 'coûteux', 'bon marché', 'argent', 'tarif', 'facture', 'addition élevée', 'économique'
    ],
    
    'cares_about_food_quality': [
        'délicieux', 'savoureux', 'frais', 'froid', 'cuisiné', 'exquis', 'goût', 'insipide', 'réchauffé', 'moisi', 'parfumé'
    ],
    
    'cares_about_atmosphere': [
        'ambiance', 'calme', 'bruyant', 'musique', 'décor', 'cadre', 'environnement', 'vue', 'chaleureux', 'intime', 'cosy', 'bruyamment'
    ],
    
    'cares_about_staff': [
        'amical', 'accueillant', 'impoli', 'serveur', 'personnel', 'accueil', 'service client', 'souriant', 'bavard', 'professionnel', 'serveuse'
    ],
}


# Preference detection function
def detect_preferences(review):
    detected = []
    review_lower = review.lower()
    for preference, keywords in preference_keywords.items():
        if any(re.search(r'\b' + re.escape(word) + r'\b', review_lower) for word in keywords):
            detected.append(preference)
    return detected

# Apply preference detection
df['Preferences'] = df['Review'].apply(lambda x: detect_preferences(x))

# Display a sample of the results
print(df[['Review', 'Preferences']].head())

# (Optional) Save to CSV if you want
df.to_csv("dakar_reviews_with_preferences.csv", index=False, encoding='utf-8-sig')

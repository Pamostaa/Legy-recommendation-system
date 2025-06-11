import pandas as pd
import ast

df = pd.read_csv("C:/Users/21655/OneDrive - Ministere de l'Enseignement Superieur et de la Recherche Scientifique/Desktop/Legy Data/Sentiment analysis/dakar_reviews_with_preferences.csv")
df['Preferences'] = df['Preferences'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# Define the four relevant preferences for Food Delivery
preference_labels = [
    'cares_about_food_quality',
    'cares_about_service_speed',
    'cares_about_price',
    'cares_about_cleanliness'
]

# Initialize the label columns
for pref in preference_labels:
    df[pref] = df['Preferences'].apply(lambda prefs: 1 if pref in prefs else 0)

df_final = df[['Review'] + preference_labels]

# Save the file
df_final.to_csv('labeled_reviews.csv', index=False, encoding='utf-8-sig')

print(df_final.head())

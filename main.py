# %% [markdown]
# # Fertilizer Recommendation System
# 
# ## Project Overview
# 
# Sistem ini mengimplementasikan **content-based fertilizer recommendation system** menggunakan teknik machine learning untuk merekomendasikan pupuk paling sesuai berdasarkan kondisi tanah, jenis tanaman, dan parameter lingkungan.
# 
# ## Key Features
# 
# - **Rekomendasi Berbasis Data**: Menggunakan dataset 3.100 record pupuk dengan parameter lingkungan dan tanah
# - **Content-Based Filtering**: Menggunakan TF-IDF vectorization dan cosine similarity untuk rekomendasi
# - **Feature Engineering Kategorikal**: Mengubah nilai numerik menjadi kategori bermakna (low, medium, high)
# - **Input Fleksibel**: Mendukung input parsial - pengguna dapat mengisi parameter yang relevan saja
# 
# ## System Components
# 
# ### 1. Data Processing
# 
# - Memuat dataset pupuk beserta parameter lingkungan (Temperature, Moisture, Rainfall, pH, NPK, Carbon)
# - Membuat fitur kategorikal dengan membagi nilai numerik menjadi tiga tingkat
# - Menghasilkan feature string gabungan dari jenis tanah, tanaman, dan kondisi lingkungan kategorikal
# 
# ### 2. Machine Learning Pipeline
# 
# - **TF-IDF Vectorization**: Mengubah fitur kategorikal menjadi vektor numerik
# - **Cosine Similarity**: Mengukur kemiripan antara query pengguna dan record pupuk
# - **Ranking System**: Mengembalikan 5 rekomendasi pupuk teratas beserta skor kemiripan
# 
# ### 3. Recommendation Engine
# 
# - Sistem input fleksibel, pengguna dapat menentukan:
#   - Jenis tanah (Loamy, Peaty, Acidic Soil, dll)
#   - Jenis tanaman (rice, corn, wheat, dll)
#   - Kondisi lingkungan (opsional, kategori low/medium/high)
# - Output berupa rekomendasi pupuk terurut lengkap dengan ID, nama pupuk, skor kemiripan, jenis tanah, dan tanaman
# 
# ## Technical Implementation
# 
# - **Library**: pandas, numpy, scikit-learn, warnings
# - **Algoritma**: Content-based filtering dengan TF-IDF & cosine similarity
# - **Struktur Data**: 3.100 record × 8 Fitur
# - **Format Output**: Rekomendasi terstruktur (ID, jenis pupuk, skor, tanah, tanaman)
# 
# Sistem ini membantu petani dan profesional pertanian mengambil keputusan pemupukan yang tepat sesuai kondisi spesifik lahan dan tanaman.
# 

# %% [markdown]
# # Load Data
# 

# %%
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

df = pd.read_csv('data.csv').drop(columns=['Remark'])
df['ID'] = range(1, len(df) + 1)
df.head()

# %% [markdown]
# # Create Categorical Feature
# 

# %%
# Function to convert numerical values to categorical (low, medium, high)
def categorize_value(value, column):
    q33 = df[column].quantile(0.33)
    q67 = df[column].quantile(0.67)

    if value <= q33:
        return 'low'
    elif value <= q67:
        return 'medium'
    else:
        return 'high'

# Create a copy of the dataframe for categorical features
df_categorical = df.copy()

# %% [markdown]
# # TF-IDF
# 

# %%
# Convert numerical columns to categorical
numerical_cols = ['Temperature', 'Moisture', 'Rainfall', 'PH', 'Nitrogen', 'Phosphorous', 'Potassium', 'Carbon']
for col in numerical_cols:
    df_categorical[col + '_cat'] = df[col].apply(lambda x: categorize_value(x, col))

# Create feature string for content-based filtering
df_categorical['features'] = (df_categorical['Soil'] + ' ' +
                            df_categorical['Crop'] + ' ' +
                            df_categorical['Temperature_cat'] + '_temperature ' +
                            df_categorical['Moisture_cat'] + '_moisture ' +
                            df_categorical['Rainfall_cat'] + '_rainfall ' +
                            df_categorical['PH_cat'] + '_ph ' +
                            df_categorical['Nitrogen_cat'] + '_nitrogen ' +
                            df_categorical['Phosphorous_cat'] + '_phosphorous ' +
                            df_categorical['Potassium_cat'] + '_potassium ' +
                            df_categorical['Carbon_cat'] + '_carbon')

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df_categorical['features'])

# %% [markdown]
# # Recommendation Model
# 

# %%
def recommend_fertilizer(soil, crop, temperature_level=None, moisture_level=None, rainfall_level=None,
                        ph_level=None, nitrogen_level=None, phosphorous_level=None,
                        potassium_level=None, carbon_level=None):
    """
    Recommend fertilizers based on soil type, crop, and optional environmental parameters.

    Args:
        soil (str): Type of soil (e.g., 'Loamy Soil', 'Peaty Soil', 'Acidic Soil')
        crop (str): Type of crop (e.g., 'rice', 'wheat', 'corn')
        temperature_level (str, optional): Temperature category ('low', 'medium', 'high')
        moisture_level (str, optional): Moisture category ('low', 'medium', 'high')
        rainfall_level (str, optional): Rainfall category ('low', 'medium', 'high')
        ph_level (str, optional): pH category ('low', 'medium', 'high')
        nitrogen_level (str, optional): Nitrogen category ('low', 'medium', 'high')
        phosphorous_level (str, optional): Phosphorous category ('low', 'medium', 'high')
        potassium_level (str, optional): Potassium category ('low', 'medium', 'high')
        carbon_level (str, optional): Carbon category ('low', 'medium', 'high')

    Returns:
        list: List of dictionaries containing top 5 fertilizer recommendations. Each dictionary contains: id, fertilizer, similarity_score, soil, crop
    """
    # Create query string with only provided parameters
    query_parts = [soil, crop]

    if temperature_level:
        query_parts.append(f"{temperature_level}_temperature")
    if moisture_level:
        query_parts.append(f"{moisture_level}_moisture")
    if rainfall_level:
        query_parts.append(f"{rainfall_level}_rainfall")
    if ph_level:
        query_parts.append(f"{ph_level}_ph")
    if nitrogen_level:
        query_parts.append(f"{nitrogen_level}_nitrogen")
    if phosphorous_level:
        query_parts.append(f"{phosphorous_level}_phosphorous")
    if potassium_level:
        query_parts.append(f"{potassium_level}_potassium")
    if carbon_level:
        query_parts.append(f"{carbon_level}_carbon")

    query = " ".join(query_parts)

    # Transform query to TF-IDF vector
    query_vector = tfidf.transform([query])

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(query_vector, tfidf_matrix)

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[0]))

    # Sort by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 5 recommendations
    top_recommendations = sim_scores[:5]

    # Get fertilizer recommendations
    recommendations = []
    for idx, score in top_recommendations:
        recommendations.append({
            'id': df_categorical.iloc[idx]['ID'],
            'fertilizer': df_categorical.iloc[idx]['Fertilizer'],
            'similarity_score': score,
            'soil': df_categorical.iloc[idx]['Soil'],
            'crop': df_categorical.iloc[idx]['Crop']
        })

    return recommendations

# %% [markdown]
# # Evaluation
# 

# %%
from sklearn.metrics import precision_score, recall_score, f1_score
import random

def evaluate_recommendation_system(test_cases, top_k=5):
    """
    Evaluasi sistem rekomendasi dengan test_cases.
    Setiap test_case harus berupa dict dengan parameter input dan ground truth fertilizer.
    test_cases: list of dict, masing-masing dict minimal punya 'soil', 'crop', dan 'fertilizer' (ground truth)
    top_k: jumlah rekomendasi teratas yang dicek

    Return: DataFrame hasil evaluasi dan precision@k, MRR, NDCG
    """
    results = []
    hits = 0
    mrr_scores = []
    ndcg_scores = []

    for case in test_cases:
        # Copy input tanpa fertilizer (ground truth)
        input_params = {k: v for k, v in case.items() if k != 'fertilizer'}
        ground_truth = case.get('fertilizer')
        recs = recommend_fertilizer(**input_params)
        top_fertilizers = [r['fertilizer'] for r in recs[:top_k]]
        hit = int(ground_truth in top_fertilizers)
        hits += hit

        # Calculate reciprocal rank for MRR
        reciprocal_rank = 0
        for i, fertilizer in enumerate(top_fertilizers):
            if fertilizer == ground_truth:
                reciprocal_rank = 1 / (i + 1)
                break
        mrr_scores.append(reciprocal_rank)

        # Calculate NDCG
        # Create relevance scores (1 for ground truth, 0 for others)
        relevance_scores = [1 if fertilizer == ground_truth else 0 for fertilizer in top_fertilizers]

        # Calculate DCG
        dcg = 0
        for i, rel in enumerate(relevance_scores):
            if rel > 0:
                dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Calculate IDCG (Ideal DCG) - best possible ranking
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0
        for i, rel in enumerate(ideal_relevance):
            if rel > 0:
                idcg += rel / np.log2(i + 2)

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

        results.append({
            'input': input_params,
            'ground_truth': ground_truth,
            'recommended': top_fertilizers,
            'hit@{}'.format(top_k): hit,
            'reciprocal_rank': reciprocal_rank,
            'ndcg': ndcg
        })

    # Calculate precision@k, MRR, and NDCG
    precision_at_k = hits / len(results)
    mrr = sum(mrr_scores) / len(mrr_scores)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)

    print(f'Precision@{top_k}: {precision_at_k:.4f}')
    print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')
    print(f'Normalized Discounted Cumulative Gain (NDCG): {avg_ndcg:.4f}')

    return pd.DataFrame(results)

# Membuat test_cases otomatis dari df_categorical (misal ambil 10 baris acak)
sample_cases = df_categorical.sample(500, random_state=42)
cat_fields = [
    'temperature_level', 'moisture_level', 'rainfall_level',
    'ph_level', 'nitrogen_level', 'phosphorous_level',
    'potassium_level', 'carbon_level'
]
cat_map = {
    'temperature_level': 'Temperature_cat',
    'moisture_level': 'Moisture_cat',
    'rainfall_level': 'Rainfall_cat',
    'ph_level': 'PH_cat',
    'nitrogen_level': 'Nitrogen_cat',
    'phosphorous_level': 'Phosphorous_cat',
    'potassium_level': 'Potassium_cat',
    'carbon_level': 'Carbon_cat'
}

test_cases = []
for _, row in sample_cases.iterrows():
    case = {
        'soil': row['Soil'],
        'crop': row['Crop'],
        'fertilizer': row['Fertilizer']
    }
    # Pilih secara random berapa banyak fitur kategori yang ingin digunakan (0-8)
    n_cat = random.randint(4, 8)
    selected_fields = random.sample(cat_fields, n_cat)
    for field in selected_fields:
        case[field] = row[cat_map[field]]
    test_cases.append(case)

df_eval = evaluate_recommendation_system(test_cases, top_k=5)
display(df_eval)

# %% [markdown]
# # Usage Test
# 

# %%
# Example usage
recommendations = recommend_fertilizer(
    soil="Loamy Soil",
    crop="rice",
    temperature_level="high",
    moisture_level="high",
    # rainfall_level="high",
    # ph_level="medium",
    # nitrogen_level="high",
    # phosphorous_level="high",
    # potassium_level="low",
    # carbon_level="medium"
)

print("Fertilizer Recommendations:")
print("=" * 50)
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. ID: {rec['id']}")
    print(f"   Fertilizer: {rec['fertilizer']}")
    print(f"   Similarity Score: {rec['similarity_score']:.4f}")
    print(f"   Soil: {rec['soil']}, Crop: {rec['crop']}")
    print("-" * 30)



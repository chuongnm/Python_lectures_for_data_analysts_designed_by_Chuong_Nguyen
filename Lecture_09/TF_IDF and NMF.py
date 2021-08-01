import pandas as pd
review = pd.read_csv('C:/Users/chuon/Desktop/MCI/PY26SA7LV2/Lecture_09/review_food.csv')


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
doc_term_matrix = tfidf_vect.fit_transform(review['Text'].values.astype('U'))



#NMF non negative matrix factoring
from sklearn.decomposition import NMF

nmf = NMF(n_components=5, random_state=42)
nmf.fit(doc_term_matrix )
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

le = LabelEncoder()
train_df['label_encoded'] = le.fit_transform(train_df['label'])
val_df['label_encoded'] = le.transform(val_df['label'])
test_df['label_encoded'] = le.transform(test_df['label'])

print(f"Classes encodées: {dict(zip(le.classes_, range(len(le.classes_))))}")

## tfidf = TfidfVectorizer(max_features=40000, min_df=2, ngram_range=(1, 1))

tfidf = TfidfVectorizer(max_features=40000, min_df=2, ngram_range=(1, 1))

X_train = tfidf.fit_transform(train_df['text']).toarray()
X_val = tfidf.transform(val_df['text']).toarray()
X_test = tfidf.transform(test_df['text']).toarray()

y_train = train_df['label_encoded'].values
y_val = val_df['label_encoded'].values
y_test = test_df['label_encoded'].values

print(f"Forme X_train: {X_train.shape}")
print(f"Vocabulaire TF-IDF: {len(tfidf.get_feature_names_out())} mots")

## OU tfidf = TfidfVectorizer(max_features=40000, min_df=2,max_df=0.85, ngram_range=(1, 1))

tfidf = TfidfVectorizer(max_features=40000, min_df=2,max_df=0.85, ngram_range=(1, 1))

X_train = tfidf.fit_transform(train_df['text']).toarray()
X_val = tfidf.transform(val_df['text']).toarray()
X_test = tfidf.transform(test_df['text']).toarray()

y_train = train_df['label_encoded'].values
y_val = val_df['label_encoded'].values
y_test = test_df['label_encoded'].values

print(f"Forme X_train: {X_train.shape}")
print(f"Vocabulaire TF-IDF: {len(tfidf.get_feature_names_out())} mots")

"""
Classes encodées: {'femme': 0, 'homme': 1}
Forme X_train: (852, 40000)
Vocabulaire TF-IDF: 40000 mots

"""
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from .utils import preprocess, create_representation

data = pd.read_csv('./models/data.csv')

data['tokens'] = data['question'].apply(preprocess)

all_words = set(word for tokens in data['tokens'] for word in tokens)
for tokens in data['tokens']:
    for word in tokens:
        all_words.add(word)

# Create a bag of words representation 
data['bow'] = data['tokens'].apply(lambda words: create_representation(words, all_words))

pipeline = Pipeline([
    ('vectorizer', DictVectorizer()),
    ('classifier', MultinomialNB())
])
X_train, X_test, Y_train, Y_test = train_test_split(data['bow'], data['correct_answer'], test_size=0.2, random_state=42)
model = pipeline.fit(X_train, Y_train)

def generate_answer(request):
    request = create_representation(preprocess(request), all_words)
    predicted_answer = model.predict([request])[0]
    return predicted_answer
print(generate_answer("What are the two most common silicates?"))
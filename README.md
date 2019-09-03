

```python
import pandas as pd
import gensim
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
```


```python
# preprocessing
df = pd.read_excel('Glossary.xls')
eng_df = df[df['language'] == 'eng']
eng_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>language</th>
      <th>subject_id</th>
      <th>keyword</th>
      <th>simplified_definition</th>
      <th>detail_definition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>eng</td>
      <td>0</td>
      <td>BoP account</td>
      <td>NaN</td>
      <td>A BoP account is a statistical statement that ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>eng</td>
      <td>0</td>
      <td>CPI(A), CPI(B), CPI(C) and Composite CPI</td>
      <td>NaN</td>
      <td>The Composite CPI reflects the impact of consu...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>eng</td>
      <td>0</td>
      <td>Death</td>
      <td>NaN</td>
      <td>A death refers to the permanent disappearance ...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>eng</td>
      <td>0</td>
      <td>Domestic household</td>
      <td>NaN</td>
      <td>Consist of a group of persons who live togethe...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>eng</td>
      <td>0</td>
      <td>Employed persons</td>
      <td>NaN</td>
      <td>Refer to those persons aged &gt;=15 who have been...</td>
    </tr>
  </tbody>
</table>
</div>




```python
all_text = eng_df['detail_definition'].drop_duplicates()
all_text[:5]
```




    1    A BoP account is a statistical statement that ...
    3    The Composite CPI reflects the impact of consu...
    5    A death refers to the permanent disappearance ...
    7    Consist of a group of persons who live togethe...
    9    Refer to those persons aged >=15 who have been...
    Name: detail_definition, dtype: object




```python
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     C:\Users\Toby\AppData\Roaming\nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    




    True




```python
lemmatizer = nltk.stem.WordNetLemmatizer()
# Define a function to perform both stemming and tokenization
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    # Filter out raw tokens to remove noise
    filtered_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if re.search('[a-zA-Z]', token)]
    
    # Stem the filtered_tokens
    stems = list(chain.from_iterable(word.split('/') for word in filtered_tokens))
    
    return filtered_tokens
```


```python
tokenized_corpus = [tokenize(entry) for entry in all_text]
tokenized_corpus[5]
```




    ['export',
     'of',
     'service',
     'are',
     'the',
     'sale',
     'of',
     'service',
     'to',
     'the',
     'rest',
     'of',
     'the',
     'world']




```python
from gensim.models import word2vec
# Set values for various parameters
feature_size = 100    # Word vector dimensionality  
window_context = 30          # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3   # Downsample setting for frequent words

w2v_model = word2vec.Word2Vec(tokenized_corpus, size=feature_size, 
                          window=window_context, min_count=min_word_count,
                          sample=sample, iter=50)
```


```python
# view similar words based on gensim's model
similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=5)]
                  for search_term in ['index', 'population', 'employee', 'service']}
similar_words
```




    {'index': ['obtained', 'volume', 'continuous', 'chain', 'aggregate'],
     'population': ['force', 'serf', 'by-censuses', 'census', 'thirty'],
     'employee': ['salary', 'wage', 'compensation', 'employment', 'mandatory'],
     'service': ['agency', 'support', 'trade-related', 'scientific', 'vii']}




```python
from sklearn.manifold import TSNE

words = sum([[k] + v for k, v in similar_words.items()], [])
# words = w2v_model.wv.index2word
wvs = w2v_model.wv[words]

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(14, 8))
plt.scatter(T[:, 0], T[:, 1], c='blue', edgecolors='grey')
for label, x, y in zip(labels, T[:, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')
```


![png](output_8_0.png)



```python
#similarity
w2v_model.wv.similarity('index', 'volume')
```




    0.8732481



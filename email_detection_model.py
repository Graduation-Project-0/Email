#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


df1 = pd.read_csv(r"D:\Downloads\Emails\spam1.csv")
df2 = pd.read_csv(r"D:\Downloads\Emails\combined_data.csv")
df3 = pd.read_csv(r"D:\Downloads\Emails\Phishing_Email.csv")
df4 = pd.read_csv(r"D:\Downloads\Emails\enronSpamSubset.csv")
df5 = pd.read_csv(r"D:\Downloads\Emails\lingSpam.csv")
df6 = pd.read_csv(r"D:\Downloads\Emails\completeSpamAssassin.csv")
df7 = pd.read_csv(r"D:\Downloads\Emails\spam_Emails_data.csv")


# In[5]:


df3.drop('Unnamed: 0', axis = 'columns', inplace = True)
df4.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis = 'columns', inplace = True)
df5.drop('Unnamed: 0', axis = 'columns', inplace = True)
df6.drop('Unnamed: 0', axis = 'columns', inplace = True)


# In[6]:


df1 = df1.rename(columns={'Message':'email', 'Category':'label'})
df2 = df2.rename(columns={'text':'email', 'label':'label'})
df3 = df3.rename(columns={'Email Text':'email', 'Email Type':'label'})
df4 = df4.rename(columns={'Body':'email', 'Label':'label'})
df5 = df5.rename(columns={'Body':'email', 'Label':'label'})
df6 = df6.rename(columns={'Body':'email', 'Label':'label'})
df7 = df7.rename(columns={'text':'email', 'label':'label'})


# In[7]:


def label_value(value):
    if value in ['Spam', '1', 'Phishing Email', 'spam']:
        return 1
    else:
        return 0


# In[8]:


df1['label'] = df1['label'].apply(label_value)
df2['label'] = df2['label'].apply(label_value)
df3['label'] = df3['label'].apply(label_value)
df4['label'] = df4['label'].apply(label_value)
df5['label'] = df5['label'].apply(label_value)
df6['label'] = df6['label'].apply(label_value)
df7['label'] = df7['label'].apply(label_value)


# In[9]:


df = pd.concat([df1,df2,df3,df4,df5,df6,df7], ignore_index= True)


# In[10]:


df


# In[11]:


df = df.drop_duplicates(['email'])


# In[12]:


df


# In[13]:


df = df.dropna()


# In[14]:


df


# In[15]:


num = df['label'].value_counts()
mini = num.min()


# In[16]:


mini


# In[17]:


df = df.groupby('label').sample(n=mini, random_state = 10)
print(df['label'].value_counts())


# In[18]:


df.reset_index(drop = True, inplace = True)


# In[19]:


df


# In[20]:


df['email'] = df['email'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[21]:


df['email'].head(15)


# In[22]:


import string


# In[23]:


def remove_punc(email):
    for punctuation in string.punctuation:
        email = email.replace(punctuation, '')
    return email
df['email'] = df['email'].apply(remove_punc)


# In[24]:


df['email'].head(15)


# In[25]:


import re


# In[26]:


def remove_not_txt(email):
    return re.sub(r'\d+', '', email)
    df['email'] = re.sub('[^a-zA-Z]', ' ', email)
df['email'] = df['email'].apply(remove_not_txt)


# In[27]:


df['email'].head(15)


# In[28]:


def remove_links(email):
    return re.sub(r'http\S+|www\S+', '', email)
df['email'] = df['email'].apply(remove_links)


# In[29]:


df['email'].head(15)


# In[30]:


import nltk


# In[31]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def remove_stopwords(email):
    words = email.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)
df['email'] = df['email'].apply(remove_stopwords)


# In[32]:


df['email'].head(15)


# In[33]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# In[34]:


lemmatizer = WordNetLemmatizer()
def lemmatize_txt(email):
    words = nltk.word_tokenize(email)
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)
df['email'] = df['email'].astype(str).apply(lemmatize_txt)


# In[35]:


x = df['email']
y = df['label']


# In[36]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features= 382766, ngram_range=(1,2), sublinear_tf = True, min_df = 3, max_df = 0.9)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)


# In[38]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[63]:


svm = LinearSVC(C=0.5)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print("accuracy:\n", accuracy_score(y_test, y_pred))
print("confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("classification report:\n", classification_report(y_test, y_pred))


# In[62]:


from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

model = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features= 382766, ngram_range=(1,2), sublinear_tf = True, min_df = 3, max_df = 0.9)),
    ('svm', LinearSVC(C=0.5))
])
scores = cross_val_score(model, x, y, cv=10, scoring='accuracy')

print("Accuracy per fold:", scores)
print("Average accuracy:", np.mean(scores))


# In[50]:


import joblib


# In[64]:


joblib.dump(svm, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


# In[66]:


get_ipython().system('jupyter nbconvert --to script email_detection_model.ipynb')


# In[ ]:





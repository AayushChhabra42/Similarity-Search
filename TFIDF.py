import numpy as np

a = "purple is the best city in the forest".split()
b = "there is an art to getting your way and throwing bananas on to the street is not it".split(" ")
c = "it is not often you find soggy bananas on the street".split(" ")

docs=[a,b,c]

def tfidf(word,document):
    #Term Frequency
    tf=document.count(word)/len(document)
    #Inverse Document Frequency
    idf=np.log10(len(docs)/sum([1 for doc in docs if word in doc]))
    return round(tf*idf,4)

print(tfidf('is',a))
print(tfidf('forest',a))

vocab=set(a+b+c)

vec_a=[]
vec_b=[]
vec_c=[]

for word in vocab:
    vec_a.append(tfidf(word,a))
    vec_b.append(tfidf(word,b))
    vec_c.append(tfidf(word,c))

print(vec_a)
print(vec_b)
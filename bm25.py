import numpy as np

a = "purple is the best city in the forest".split()
b = "there is an art to getting your way and throwing bananas on to the street is not it".split()
c = "it is not often you find soggy bananas on the street".split()
d = "green should have smelled more tranquil but somehow it just tasted rotten".split()
e = "joyce enjoyed eating pancakes with ketchup".split()
f = "as the asteroid hurtled toward earth becky was upset her dentist appointment had been canceled".split()

docs=[a,b,c,d,e,f]

avgdl=sum(len(sentence) for sentence in docs)/len(docs)
N=len(docs)

def bm25(word,sentence,k=1.2,b=0.75):
    freq=sentence.count(word)
    tf=(freq*(k+1))/(freq+k*(1-b+b*(len(sentence)/avgdl)))

    N_q=sum([1 for doc in docs if word in doc]) 

    idf=np.log(((N-N_q+0.5)/(N_q+0.5))+1)

    return round(tf*idf,4)

print(bm25('purple',a))
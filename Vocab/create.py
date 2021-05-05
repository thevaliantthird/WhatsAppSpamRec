import os
import numpy as np

with open("pos.npy","rb") as f:
    a = np.load(f)
with open("neg.npy","rb") as f:
    b = np.load(f)

a = np.vstack((a,b))

t = open("oxford.txt","r",encoding = 'charmap')

def normalise_text (text):
    text = text.lower() # lowercase
    text = text.replace(r"\#","") # replaces hashtags
    text = text.replace(r"http\S+","URL")  # remove URL addresses
    text = text.replace(r"@","")
    text = text.replace(r"[^A-Za-z0-9()!?\'\`\"]", " ")
    text = text.replace("\s{2,}", " ")
    text = text.replace(r"\n"," ")
    text = text.replace(r"IMG-S+","IMAGE")
    text = text.strip()
    return text

s = t.read()
s = normalise_text(s)
L = []
iPrev = -1
for i in range(0,len(s)):
    if i-iPrev > 200 and s[i]==' ':
        L.append(s[iPrev+1:i])
        iPrev = i

print(len(L))

a = np.vstack((a,np.expand_dims(np.array(L),axis = 1)))

np.save('dat.npy',a)        

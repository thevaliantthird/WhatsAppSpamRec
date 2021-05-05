import os
import numpy as np


f = open("past.txt","r",encoding = 'charmap')

s = f.read()

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


s = normalise_text(s)

l = s.split(' ')
dict = {}

for s in l:
    dict[s] = True

print(len(dict))

import math
import string
import re
import ast
import os
import sys

#function to preprocess each string by 
#1.removing punctuation 
#2.converting each string to lower case 
#3.splitting each string to a list of words
def tokenize(s):
    s=s.translate(str.maketrans('', '', string.punctuation))
    s=s.lower()
    strlist = re.sub("[^\w]", " ",  s).split()
    return strlist

#calculating posterior probability of each review for classification
#posterior probability is the product of prior probablity and conditinal probabilty of each word in review
#calculations done on Log scale
#assign class based on max posterior probability

#function to assign class for positive/neagtive classification
def TestSenti(vocab,prior,cond_prob,s):
    s=tokenize(s)
    w=[i for i in s if i in vocab]
    score_pos=math.log(prior['pos'])
    score_neg=math.log(prior['neg'])
    for i in w:
        score_pos+=math.log(cond_prob['pos'][i])
        score_neg+=math.log(cond_prob['neg'][i])
    if score_pos>score_neg:
        return 'positive'
    else:
        return 'negative'

#function to assign class for truthful/deceptive classification
def TestTruth(vocab,prior,cond_prob,s):
    s=tokenize(s)
    w=[i for i in s if i in vocab]
    score_dec=math.log(prior['dec'])
    score_truth=math.log(prior['truth'])
    for i in w:
        score_dec+=math.log(cond_prob['dec'][i])
        score_truth+=math.log(cond_prob['truth'][i])
    
    if score_dec>score_truth:
       
        return 'deceptive'
    else:
        
        return 'truthful'


#file containing test dataset
test_file = sys.argv[1]

#loading parameters learnt from the training dataset
filehandle=[]
with open('nbmodel.txt', 'r') as file:
    for f in file:
        filehandle.append(f)  
vocab=ast.literal_eval(filehandle[0])
prior=ast.literal_eval(filehandle[1])
cond_prob=ast.literal_eval(filehandle[2])
vocab1=ast.literal_eval(filehandle[3])
prior1=ast.literal_eval(filehandle[4])
cond_prob1=ast.literal_eval(filehandle[5])

#cassification fo test dataset asand writing classes into output file 
with open('nboutput.txt', 'w') as output_file:    
    for dirpath,sub,files in os.walk(test_file):
        for filename in files:
            if filename.endswith('.txt') and 'README' not in filename:
                    file = open(os.path.join(dirpath, filename), 'r')
                    data = file.read()
                    
                    predict_senti=TestSenti(vocab,prior,cond_prob,data)
                    predict_truth=TestTruth(vocab1,prior1,cond_prob1,data)
                    output_file.write(predict_truth+' '+predict_senti+' '+dirpath+'/' + filename+'\n')
    
output_file.close()







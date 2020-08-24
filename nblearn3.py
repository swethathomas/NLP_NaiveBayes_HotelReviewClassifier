import os
import string
import re
from collections import Counter
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


#function to calculate probailities for Navive Bayes Classifier
#The function calculates probabilities for the positive/negative sentiment classification
#Probabilities calculated:- 1.Prior Probabilities 2.Conditional Probabilities
def NaiveBayesSenti(document):
    N=len(document)
    N_pos=0
    N_neg=0
    docs=[]
    Tpos=0
    Tneg=0
    pos=[]
    neg=[]
    pos_dic={}
    neg_dic={}
    cond_prob={}
    prior={}
    wordlist=[]
    #for each review create tuple contating prepreocessed review and class(positive/negative) it belongs to
    for place, item in enumerate(document):
        if document[place][1]=="pos":
            N_pos+=1
        if document[place][1]=="neg":
            N_neg+=1
        docs.append((tokenize(item[0]),document[place][1]))
        #create list of all the words in each review
        for i in docs[place][0]:
            wordlist.append(i)
            
    #calculate prior probabilities
    prior_pos=N_pos/N
    prior_neg=N_neg/N
    
    #remove most frequently occuring words as these are stop words
    ctr= Counter(wordlist) 
    least = 0
    most=800
    vocab=[d for d in wordlist if ctr[d] > least and ctr[d] < most]
    
    #create a vocabulary: set of all the words encountered in the reviews 
    vocab=set(vocab)

    #craete a list fo all words seen in positive reviews 
    #craete a list fo all words seen in negative reviews 
    
    for review in docs:
        if review[1]=="pos":
            for i in review[0]:
                pos.append(i)
                #count number of words in positive reviews that occur in the vocab
                if i in vocab:
                    Tpos=Tpos+1
        
        if review[1]=="neg":
            for i in review[0]:
                neg.append(i)
                #count number of words in negative reviews that occur in the vocab
                if i in vocab:
                    Tneg+=1
                    
    Tpos+=len(vocab)
    Tneg+=len(vocab)
    
    #calculate frequency of words in positive/negative reviews 
    pos_ctr= Counter(pos)
    neg_ctr= Counter(neg)
   
    #calculating conditional probabilities for each word in the reviews 
    for i in vocab:
        pos_dic[i]=0
        neg_dic[i]=0
        #Performing add one smoothing
        if i in pos_ctr:
            pos_dic[i]+=(pos_ctr[i]+1)
        else:
                pos_dic[i]+=1
            
        pos_dic[i]=pos_dic[i]/Tpos
    
        #Performing add one smoothing
        if i in neg_ctr:
            neg_dic[i]+=(neg_ctr[i]+1)
        else:
            neg_dic[i]+=1
            
        neg_dic[i]=neg_dic[i]/Tneg
        
    #return conditional and prior probability of each class
    cond_prob['pos']=pos_dic
    cond_prob['neg']=neg_dic
    prior['pos']=prior_pos
    prior['neg']=prior_neg
    return vocab,prior,cond_prob

#function to calculate probailities for Navive Bayes Classifier
#The function calculates probabilities for the truthful/deceptive classification
#Probabilities calculated:- 1.Prior Probabilities 2.Conditional Probabilities
def NaiveBayesTruth(document):
    N=len(document)
    N_dec=0
    N_truth=0
    docs=[]
    Tdec=0
    Ttruth=0
    dec=[]
    truth=[]
    dec_dic={}
    truth_dic={}
    cond_prob={}
    prior={}
    wordlist=[]
    #for each review create tuple contating prepreocessed review and class(truthful/deceptive) it belongs to
    for place, item in enumerate(document):
        if document[place][1]=="dec":
            N_dec+=1
        if document[place][1]=="truth":
            N_truth+=1
        docs.append((tokenize(item[0]),document[place][1]))
        #create list of all the words in each review
        for i in docs[place][0]:
            wordlist.append(i)
            
   #calculate prior probabilities
    prior_dec=N_dec/N
    prior_truth=N_truth/N
    
    #remove most frequently occuring words as these are stop words
    ctr= Counter(wordlist) 
    least = 0
    most=800
    vocab=[d for d in wordlist if ctr[d] > least and ctr[d] < most]
    
    #create a vocabulary: set of all the words encountered in the reviews 
    vocab=set(vocab)

    #craete a list fo all words seen in truthful reviews 
    #craete a list fo all words seen in deceptive reviews 
    for review in docs:
        if review[1]=="dec":
            for i in review[0]:
                dec.append(i)
                #count number of words in deceptive reviews that occur in the vocab
                if i in vocab:
                    Tdec=Tdec+1
        
        if review[1]=="truth":
            for i in review[0]:
                truth.append(i)
                #count number of words in truthful reviews that occur in the vocab
                if i in vocab:
                    Ttruth+=1
                    
    Tdec+=len(vocab)
    Ttruth+=len(vocab)
    
    #calculate frequency of words in truthful/deceptive reviews
    dec_ctr= Counter(dec)
    truth_ctr= Counter(truth)
   
    
    for i in vocab:
        dec_dic[i]=0
        truth_dic[i]=0
        #perform add one smoothing
        if i in dec_ctr:
            dec_dic[i]+=(dec_ctr[i]+1)
        else:
            
                dec_dic[i]+=1
            
        dec_dic[i]=dec_dic[i]/Tdec
        #perfrom add one smoothing
        if i in truth_ctr:
            truth_dic[i]+=(truth_ctr[i]+1)
        else:
            truth_dic[i]+=1
            
        truth_dic[i]=truth_dic[i]/Ttruth
    
    #return conditional and prior probability of each class
    cond_prob['dec']=dec_dic
    cond_prob['truth']=truth_dic
    prior['dec']=prior_dec
    prior['truth']=prior_truth
    return vocab,prior,cond_prob


#stores reviews for positive/negative classification
document1=[]
#stores reviews for truthful/deceptive classfication
document2=[]

#give folder containg reviews as input
train_file = sys.argv[1]

for dirpath,sub,files in os.walk(train_file):
    for filename in files:
        if filename.endswith('.txt') and 'README' not in filename:
            if 'positive' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document1.append((data,"pos"))
            if 'negative' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document1.append((data,"neg"))
                
            if 'truthful' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document2.append((data,"truth"))
            if 'deceptive' in dirpath:
                file = open(os.path.join(dirpath, filename), 'r')
                data = file.read()
                document2.append((data,"dec"))



#caluculate prior and conditional prbabilities for each classification
vocab,prior,cond_prob=NaiveBayesSenti(document1)
vocab1,prior1,cond_prob1=NaiveBayesTruth(document2)


#wrtie these calculated probabilities to a text file 
with open('nbmodel.txt', 'w') as filehandle:
    filehandle.write(str(vocab)+'\n')
    filehandle.write(str(prior)+'\n')
    filehandle.write(str(cond_prob)+'\n')
    filehandle.write(str(vocab1)+'\n')
    filehandle.write(str(prior1)+'\n')
    filehandle.write(str(cond_prob1)+'\n')
filehandle.close()

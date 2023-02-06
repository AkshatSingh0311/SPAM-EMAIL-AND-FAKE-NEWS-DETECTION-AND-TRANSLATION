import numpy as np 
import pandas as pd
import re
import string
import itertools
from string import digits
from nltk.tokenize import word_tokenize
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier,Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

def preprocessing(data): #Preprocessing the data
    
    #Converting the data to lower case
    data=data.apply(lambda x: x.lower())
    
    # Remove all numbers from text
    remove_digits = str.maketrans('','',digits)
    data=data.apply(lambda x: x.translate(remove_digits))
    
    #Removing the special characters using regex
    special=set(string.punctuation)
    data=data.apply(lambda x: ''.join(i for i in x if i not in special))
    
    # Remove extra spaces
    data=data.apply(lambda x: x.strip())
    
    return data

def english_hindi_convert(): #Translating English to Hindi
    
    df=pd.read_csv(r'C:\Users\utsav\Downloads\Hindi_English_Truncated_Corpus.csv')
    df=df[df['source']=='ted']
    df=df.drop('source',axis=1)

    #Removing null and duplicate values, if present
    df=df[~pd.isnull(df['english_sentence'])]
    df.drop_duplicates(inplace=True)
    df=df[~pd.isnull(df['hindi_sentence'])]
    df.drop_duplicates(inplace=True)

    #Converting the data to lower case
    df['english_sentence']=df['english_sentence'].apply(lambda x: x.lower())
    df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: x.lower())

    # Remove all numbers from text
    remove_digits = str.maketrans('','',digits)
    df['english_sentence']=df['english_sentence'].apply(lambda x: x.translate(remove_digits))
    df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: x.translate(remove_digits))
    df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

    #Removing the special characters using regex
    special=set(string.punctuation)
    df['english_sentence']=df['english_sentence'].apply(lambda x: ''.join(i for i in x if i not in special))
    df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: ''.join(i for i in x if i not in special))

    # Remove extra spaces
    df['english_sentence']=df['english_sentence'].apply(lambda x: x.strip())    
    df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: x.strip())    

    #Keeping senetences with length at least 20 for efficincy
    df['length_eng_sentence']=df['english_sentence'].apply(lambda x:len(x.split(" ")))
    df['length_hin_sentence']=df['hindi_sentence'].apply(lambda x:len(x.split(" ")))
    df=df[df['length_eng_sentence']<=20]
    df=df[df['length_hin_sentence']<=20]    
    
    #Tokenizing the data
    df['english_sentence']=df['english_sentence'].apply(lambda x: word_tokenize(x))
    df['hindi_sentence']=df['hindi_sentence'].apply(lambda x: word_tokenize(x))

    # Get English and Hindi Vocabulary
    all_eng_words=set()
    for eng in df['english_sentence']:
        for word in eng:
            if word not in all_eng_words:
                all_eng_words.add(word)

    all_hindi_words=set()
    for hin in df['hindi_sentence']:
        for word in hin:
            if word not in all_hindi_words:
                all_hindi_words.add(word)

    all_eng_words=list(all_eng_words) #List of all English words
    all_hindi_words=list(all_hindi_words) #List of all Hindi words
    m=min(len(all_eng_words),len(all_hindi_words))
    adict={} #Dictionary to map English words to Hindi words
    for i in range(m):
        adict[all_eng_words[i]]=all_hindi_words[i]
    return adict
    
def translate(lst): #Perform translation
    
    adict=english_hindi_convert()    
    st=""
    for i in range(len(lst)):        
        if lst[i] in adict:
            st+=adict[lst[i]]+" "
    return st

def plot_confusionmatrix(y_train_pred,y_train): #Plotting Confusion Matrix
        
    cf = confusion_matrix(y_train_pred,y_train)        
    sns.heatmap(cf,annot=True,cmap='Blues',fmt='g')
    plt.tight_layout()
    plt.xlabel("Actual")
    plt.ylabel("Predcited")
    plt.title("Fake News Detection")
    plt.show()

def buildMLPerceptron(train_features,test_features,train_targets,test_targets,num_neurons=10): #Classification
    
    classifier=Perceptron() #Initialise classifier
    classifier.fit(train_features,train_targets) #Training model
    predictions=classifier.predict(test_features) #Testing the model
    score=np.round(metrics.accuracy_score(test_targets,predictions),2) #Evaluating Predictions
    print(f'Accuracy of MLPClassifier for Fake News Detection: {round(score*100,2)}%')    
    plot_confusionmatrix(test_targets,predictions)
    return test_features,predictions

def main():
    df=pd.read_csv(r'C:\Users\utsav\Downloads\news\news.csv') #Read dataset using pandas library
    
    x_train,x_test,y_train,y_test=train_test_split(df['text'],df.label,test_size=0.3,random_state=7) 
    #Splitting dataset into test and train datasets    
    s=input("Enter News: ") #Custom Input    
    test_input=pd.Series(s)
    test_output=pd.Series(input("Enter Label (REAL/FAKE): "))    
    tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7) #Finding tf-idf values            
    tfidf_train=tfidf_vectorizer.fit_transform(x_train)     
    tfidf_test=tfidf_vectorizer.transform(x_test)    
    cust_tfidf_test=tfidf_vectorizer.transform(test_input)    
    
    pac=PassiveAggressiveClassifier(max_iter=50) #Using Passive Aggressive Classifier as reference
    pac.fit(tfidf_train,y_train)    
    y_pred=pac.predict(tfidf_test)    
    cust_y_pred=pac.predict(cust_tfidf_test)    
    score=accuracy_score(y_test,y_pred)
    cust_score=accuracy_score(test_output,cust_y_pred)
    print(f'Accuracy of AggresivePassiveClassifier: {round(score*100,2)}%')    
    print(f'Accuracy of AggresivePassiveClassifier for custom input: {round(cust_score*100,2)}%')  #Result for Custom Input 
    print("Actual:",test_output[0],"\nPredicted:",cust_y_pred[0])
    
    print("\nFor Custom Input:")
    test_feat,test_pred=buildMLPerceptron(tfidf_train,cust_tfidf_test,y_train,test_output) #Performing classification using MLP Classifier    
    test_feat=test_feat.toarray()    
    test_input=preprocessing(test_input)
    print("\nResult:")
    unique, counts = np.unique(test_pred, return_counts=True) #Print number of Real and Fake news
    print(np.asarray((unique, counts)).T,"\n\n")
    print("Prediction:\n") #Show News and Prediction    
    print("Sentence -\n",test_input[0],"\n\nPrediction:",test_pred[0],"\n\n") #Result for Custom Input
    
    test_feat,test_pred=buildMLPerceptron(tfidf_train,tfidf_test,y_train,y_test) #Performing classification using MLP Classifier
    x_test.index=[i for i in range(len(x_test))]    
    test_feat=test_feat.toarray()    
    x_test=preprocessing(x_test)
    print("\nResult:")
    unique, counts = np.unique(test_pred, return_counts=True) #Print number of Real and Fake news
    print(np.asarray((unique, counts)).T,"\n\n")
    print("Prediction:\n") #Show News and Prediction
    for i in range(3):
        if(len(x_test[i])!=0):
            print("Sentence ",(i+1)," -\n",x_test[i],"\n\nPrediction:",test_pred[i],"\n\n")
            
    
    print("Translation of Real News:\n") #Translating Real News
    for i in range(3):        
        if(test_pred[i]=="REAL"):
            lst=[j for j in x_test[i].split()]                        
            print("Sentence:\n",x_test[i],"\n\nHindi Translation:\n")
            print(translate(lst),"\n\n")
        
if __name__=="__main__": #Driver Code
    main()

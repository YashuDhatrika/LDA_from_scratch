

#Importing the required libraries
import os
import math
import numpy as np
import random
import time as t
import matplotlib.pyplot as plt
import operator
import pandas as pd


#Importing the txt files and csv files

#Getting the path of the python file
dirpath = os.path.dirname(__file__)
#Importing the csv file:
file_list1=[]
file_name1=[]
for root,dirs,files in os.walk(dirpath):
    for file in files:
       if file.endswith(".csv"):
           len_filename=len(file)
           file_name1.append(file[:len_filename-4])
           x1= os.path.join(dirpath, file)
           y1 = np.genfromtxt(x1, delimiter=",")
           file_list1.append(y1)
        
# Creating a dictionary which has key as a file names and value is the array of the file content
dic_files1=dict(zip(file_name1,file_list1))

label_file=dic_files1.get('index')
label=np.asarray(np.asmatrix(label_file)[:,1])





# Walking text file of the dirpath  and storing them as an array
file_list=[]
file_name=[]
for root,dirs,files in os.walk(dirpath):
    for file in files:
       if file.endswith(".txt"):
           len_filename=len(file)
           file_name.append(file[:len_filename-4])
           x= os.path.join(dirpath, file)
           file1 = open(x, 'r')
           y = file1.read().split()
           file_list.append(y)
dic_files=dict(zip(file_name,file_list))


#Task1: Gibbs Sampling

#Creating list of words and document indices :
doc_ind=[]
word_ind=[]
D=len(dic_files.keys())
K=20

for keys in dic_files.keys():
    for i in range(len(dic_files[keys])):
        doc_ind.append(int(keys))
        word_ind.append(dic_files[keys][i])

#total number of words
Nword=len(word_ind)

#Generating array initial topics indices
rand_top=[]
for i in range (Nword):
    rand_top.append(random.randrange(0, K, 1))

#cd matrix
cd=np.zeros((D,K))
for i in range(len(doc_ind)):
    for j in range(len(rand_top)):
        if i==j:
            cd[doc_ind[i]-1,[rand_top[j]]]+=1
            
#kVmatrix
V=list(set(word_ind))
V_dic=dict(zip(V,range(len(V))))
ct=np.zeros((K,len(V)))
for i in range(len(rand_top)):
    for j in range(len(word_ind)):
        if i==j:
            ct[rand_top[i],V_dic[word_ind[i]]]+=1


#Pmatrix
p=np.zeros((1,K))

#permutation p(n)
n=list(range(Nword))
per_n=np.random.permutation(n)
list_per=per_n.tolist()

#Generating the samples using the collapsed Gibbs sampler
for i in range(500):
    for j in list_per:
        word_i=word_ind[j]
        topic_i = rand_top[j]
        doc_i = doc_ind[j]
        #Updating the cd and ct as a word is removed from a given topic
        cd[doc_i - 1][topic_i] =cd[doc_i - 1][topic_i] - 1
        ct[topic_i][V_dic[word_i]]=ct[topic_i][V_dic[word_i]]-1
        #Calculation of conditional probability for each topic
        for i in range(0,K):
            alpha=5.0/K
            beta=0.01
            exp1=(ct[i][V_dic[word_i]]+beta)/(len(V)*beta+np.sum(ct[i]))
            exp2=(cd[doc_i-1][i]+alpha)/(K*alpha+np.sum(cd[doc_i-1]))
            p[0][i]=exp1*exp2

        sum=np.sum(p,axis=1)
        P_norm=p/sum
        P_list= [item for sublist in P_norm.tolist() for item in sublist]
        #A topic is assigned for the word based on the conditional probabilities
        rand_top[j]=np.random.choice(list(set(rand_top)), 1, p=P_list)[0]
        #Updating the cd nad ct matrix
        cd[doc_i-1][rand_top[j]]=cd[doc_i-1][rand_top[j]]+1
        ct[rand_top[j]][V_dic[word_i]]=ct[rand_top[j]][V_dic[word_i]]+1

# The following step, outputs the 5 most frequent word in each topic and saves in CSV
b=ct.tolist()
final_lst=[]
for i in b:
    maxindex = []
    for j in range(5):
        max_ind=i.index(max(i))
        maxindex.append((max_ind))
        i.remove(max(i))
    final_lst.append(maxindex)


topic_words=[]
for i in final_lst:
    list_words = []
    for j in i:
        for key, values in V_dic.items():
            if values==j:
                list_words.append(key)
    topic_words.append(list_words)

my_df = pd.DataFrame(topic_words)

my_df.to_csv('topicwords.csv', index=False, header=False)

#Task2: Classification

#preparing data for topic representation
topic_rep=np.zeros((D,K))
for i in range(D):
 for j in range(K):
     topic_rep[i][j]=(cd[i][j]+alpha)/((K*alpha)+np.sum(cd[i]))

#preparing data for bag of words:
bag_words=np.zeros((D,len(V)))
for i in range(len(doc_ind)):
 for j in range(len(word_ind)):
     if i==j:
         bag_words[doc_ind[i]-1][V_dic[word_ind[i]]]+=1

#creating a sigmoid function and also prevents math overflow error when mu<<<0
def np_sigmoid(mu):
 if mu<0:
     return 1 - 1 / (1 + math.exp(mu))
 else:
     return (1.0 / (1 + math.exp(-mu)))

#The following steps are performed inside the function
#1: data get  split into trainingset and testingset
#2:Training data of different sizes are generated
#3: Free features is added to both training and testing data
#4:For each training data size, w is using newton method
#5: Based on w , Y is calculated and then error rate
#6: The above steps are performed 30 times and mean error rate and sd error rate is computed
def ds_model(data,tdata):
 N=int(data.shape[0])
 testsize=int(round(N/3,0))
 #Combing the data and targetdata
 combinedata=np.concatenate((data, tdata), axis=1)
 size_lst = []
 err_mean = []
 err_sd = []
 for size in range(int(N / 10), int((2 * N) / 3), int(N / 7)):
     #testdata of size one third is randomly generated
     err_list = []
     for i in range(0, 30):
         # testdata of size one third is randomly generated
         randomindex = np.random.choice(combinedata.shape[0], testsize, replace=False)
         testdata = combinedata[randomindex]
         remaindata = np.delete(combinedata, randomindex, axis=0)
         # train data of different sizes are picked from the remaining data
         traindata = remaindata[np.random.choice(remaindata.shape[0], size, replace=False)]
         col = traindata.shape[1]
         train = traindata[:, 0:col- 1]
         free_para = np.ones((len(train)))
         #free paramenter is added to the trainingdata
         xvalues = np.column_stack((free_para, train))
         tvalues = traindata[:,col - 1]
         test1 = testdata[:, 0:col - 1]
         free_para1 = np.ones((len(test1)))
         #Free parameter is added to testing dataset
         test = np.column_stack((free_para1, test1))
         ttvalue = testdata[:, col - 1]
         w = np.zeros(col)
         flag = 1
         count=0
         while flag == 1:
             Y = []
             R = 0
             #calculating the Hessian Matrix
             for i in range(0, len(xvalues)):
                 x = np.transpose(w).dot(xvalues[i,])
                 y = np_sigmoid(x)

                 r = y * (1 - y) * xvalues[i,][:, None].dot(xvalues[i,][None, :])
                 R += r
                 Y.append(y)

             # calculating the expression in newton method
             # alpha=0.1
             exp1 = np.linalg.inv(0.001 * np.identity(xvalues.shape[1]) + R)
             exp2 = np.transpose(xvalues).dot(np.array(Y) - tvalues) + 0.001 * w
             w_new = w - (exp1.dot(exp2))
             if np.square(np.linalg.norm(w)) != 0 and (np.square(np.linalg.norm(w_new - w)) / np.square(np.linalg.norm(w)) < 0.001 or count>=100):
                 flag = 0
             else:
                 w = w_new
                 count+=1
         # print(w_new)
         list1 = []
         #Calculating the error on the tesdata
         for i in range(0, len(test)):
             x = np.transpose(w_new).dot(test[i,])
             if np_sigmoid(x) >= 0.5:
                 p = 1.0
                 list1.append(p)
             else:
                 p = 0.0
                 list1.append(p)
         error = (len(list1) - list(np.subtract(list1, ttvalue)).count(0)) / len(list1)
         err_list.append(error)
         mean = np.mean(err_list, axis=0)
         sd = np.std(err_list, axis=0)
     size_lst.append(size)
     err_mean.append(mean)
     err_sd.append(sd)
     mean_dic = dict(zip(size_lst, err_mean))
     sd_dic = dict(zip(size_lst, err_sd))
 return mean_dic, sd_dic

# Plotting the graphs for 2 representation
mean,sd=ds_model(topic_rep,label)
mean1,sd1=ds_model(bag_words,label)
fig = plt.figure()
fig.suptitle('Learning curve performance-Topic Representation vs. Bag of words')
ax = fig.add_subplot(111)
ax.set_xlabel('Training size', fontsize = 14)
ax.set_ylabel('Mean Error', fontsize = 14)
linestyle = {"linestyle":"--", "linewidth":4, "markeredgewidth":5, "elinewidth":5, "capsize":10}
ax.errorbar(mean.keys(), mean.values(), sd.values(), marker='^',label='Topic Repres.',color="g", **linestyle)
ax.errorbar(mean1.keys(), mean1.values(), sd1.values(), marker='^',label='Bag of words Repres.',color="b", **linestyle)
plt.show()


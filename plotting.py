import gensim, logging
from nltk import sent_tokenize,word_tokenize
import os
import numpy
from matplotlib.mlab import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
f1 = open('temp_1','r').read()
f2 = open('temp_2','r').read()
x, y = "", ""
for char in f1:
    if(char.isalpha() or char==" "):
        x+=char
    elif(char=="."):
        x+=char

for char in f2:
    if (char.isalpha() or char == " "):
        y += char
    elif (char == "."):
        y += char

# x="My name is Fenil. How are you"
sentences_1=word_tokenize(x)
sentences_2=word_tokenize(y)

lol_1=[]
for i in sentences_1:
    lol_1.append([i])
# print(lol)
model_1 = gensim.models.Word2Vec(min_count=1)
print(model_1.wv.vocab)
model_1.build_vocab(lol_1)
model_1.train(lol_1, total_examples=len(sentences_1), epochs=100)
# print(model.wv.vocab)
# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
# lee_train_file = test_data_dir + 'lee_background.cor'
# sentences = MyText()

# print(sentences)
# model = gensim.models.Word2Vec(lol, min_count=10)
data_1=[]

for i in sentences_1:
    # print(i)
	data_1.append(list(model_1[i]))
data_1=numpy.array(data_1)



lol_2=[]
for i in sentences_2:
    lol_2.append([i])
# print(lol)
model_2 = gensim.models.Word2Vec(min_count=1)
print(model_2.wv.vocab)
model_2.build_vocab(lol_2)
model_2.train(lol_2, total_examples=len(sentences_1), epochs=100)
# print(model.wv.vocab)
# test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data']) + os.sep
# lee_train_file = test_data_dir + 'lee_background.cor'
# sentences = MyText()

# print(sentences)
# model = gensim.models.Word2Vec(lol, min_count=10)
data_2=[]

for i in sentences_2:
    # print(i)
	data_2.append(list(model_2[i]))
data_2=numpy.array(data_2)
print(data_2.shape)
trans_2=numpy.transpose(data_2)
trans_1=numpy.transpose(data_1)

avg_1, avg_2= [], []

for i in trans_2:
    t=sum(i)/float(len(i))
    avg_2.append(t)

for i in trans_1:
    t=sum(i)/float(len(i))
    avg_1.append(t)
dif=0.0
for i in range(len(data_1[0])):
    dif+= (avg_1[i]-avg_2[i])**2
dif=dif**0.5
print ("Difference : ", dif)

#print(len(data_1[0])==len(data_2[0]))
# print(len(data_2[0]))
raw_input("Press enter to generate graph")

ipca = PCA(n_components=3)

ipca.fit(data_1)
x_1=ipca.transform(data_1)

ipca.fit(data_2)
x_2=ipca.transform(data_2)


Xs=[]
Ys=[]
Zs=[]
for i in x_1[0:50]:
    Xs.append(i[0])
    Ys.append(i[1])
    Zs.append(i[2])

ax.scatter(Xs, Ys, Zs, c='r', marker='o')

Xs=[]
Ys=[]
Zs=[]
for i in x_2:
    Xs.append(i[0])
    Ys.append(i[1])
    Zs.append(i[2])

ax.scatter(Xs, Ys, Zs, c='b', marker='o')



ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
plt.close()

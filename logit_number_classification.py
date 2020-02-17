import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
digits=load_digits()

print 'image data shape ',digits.data.shape
print 'image data shape ',digits.target.shape
'''
image data shape  (1797L, 64L)
image data shape  (1797L,)
it means eight by eight images '''

plt.figure(figsize=(20,4))
for ind,(image,label) in enumerate(zip(digits.data[0:5],digits.target[0:5])):
    plt.subplot(1,5,ind+1)
    plt.imshow(np.reshape(image,(8,8)),cmap=plt.cm.gray)
    plt.title('Training : %i\n' % label,fontsize=20)
    plt.savefig('before pred')
    plt.show()

x_train,x_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.23,random_state=2)

print x_train[:1]
'''
[[ 0.  0. 11. 15. 15. 16.  9.  0.  0.  4. 16. 14.  8.  9.  3.  0.  0.  4.
  12.  0.  0.  0.  0.  0.  0.  6. 16. 15.  3.  0.  0.  0.  0.  3. 11. 11.
  12.  0.  0.  0.  0.  0.  0.  2. 16.  0.  0.  0.  0.  2. 12.  9. 16.  0.
   0.  0.  0.  0. 11. 16.  8.  0.  0.  0.]]'''

print x_test[:1]
'''
[[ 0.  0.  0.  3. 16.  3.  0.  0.  0.  0.  0. 10. 16. 11.  0.  0.  0.  0.
   4. 16. 16.  8.  0.  0.  0.  2. 14. 12. 16.  5.  0.  0.  0. 10. 16. 14.
  16. 16. 11.  0.  0.  5. 12. 13. 16.  8.  3.  0.  0.  0.  0.  2. 15.  3.
   0.  0.  0.  0.  0.  4. 12.  0.  0.  0.]]'''
print y_train[:6]    # [5 6 1 6 5 2]
print y_test[:6]     # [4 0 9 1 4 7]


from sklearn.linear_model import LogisticRegression

logreg= LogisticRegression()
logreg.fit(x_train,y_train)

print logreg.predict(x_test[0].reshape(1,-1))
''' [4] '''

print logreg.predict(x_test[0:10])
''' [4 0 9 1 8 7 1 5 1 6]  '''

pred= logreg.predict(x_test)
score=logreg.score(x_test,y_test)
print score   # or score *100  #  94.20289855072464
''' 0.9420289855072463 '''

from sklearn import metrics

cm=metrics.confusion_matrix(y_test,pred)
print cm   # confussion matrix
'''
[[38  0  0  0  0  0  0  0  0  0]
 [ 0 44  0  1  0  0  0  0  2  1]
 [ 0  0 43  0  0  0  0  0  0  0]
 [ 0  0  0 37  0  0  0  3  2  0]
 [ 0  2  0  0 34  0  0  0  1  1]
 [ 0  0  0  0  0 46  0  0  0  0]
 [ 0  0  0  0  0  0 40  0  1  0]
 [ 0  0  0  0  0  0  0 45  1  0]
 [ 0  2  0  0  0  0  0  0 35  1]
 [ 0  0  0  1  0  1  0  1  3 28]]  '''

# to create heat map

plt.figure(figsize=(9,9))
sns.heatmap(cm,annot=True, fmt =".3f",linewidths=.5,square=True,cmap='Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Accuracy score : {0}'.format(score)
plt.title(all_sample_title,size=15);
plt.savefig('confussion _matrix')
plt.show()

index=0
classifiedIndex=[]
for predict,actual in zip(pred,y_test):
    if predict==actual:
        classifiedIndex.append(index)

    index+=1

plt.figure(figsize=(20,4))
for plotIndex,wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4,plotIndex+1)
    plt.imshow(np.reshape(x_test[wrong],(8,8)),cmap=plt.cm.gray)
    plt.title('Predicted : {} , Actual : {} '.format(pred[wrong],y_test[wrong]),fontsize=20)
    plt.savefig('After predictions')
    plt.show()
    
        

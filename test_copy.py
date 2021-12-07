from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pickle
import findspark
findspark.init()
import pyspark
import json
from pyspark.sql.session import SparkSession
from pyspark.sql import functions as sf
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import numpy
import string
from nltk.corpus import stopwords

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext(appName="test")
ssc = StreamingContext(sc, 10)
spark=SparkSession(sc)
#spark.sparkContext.setLogLevel('WARN')

# Create a DStream that will connect to hostname:port, like localhost:9999
lines = ssc.socketTextStream("localhost", 6100)

lines.foreachRDD(lambda rdd: collect(rdd))

def collect(inp):
	if not inp.isEmpty():
		df = spark.read.json(inp).withColumn('result', sf.explode(sf.arrays_zip('feature0', 'feature1', 'feature2'))).select('result.feature0', 'result.feature1', 'result.feature2')
		#df.show()
		f0 = np.array(df.select('feature0').collect())
		f1 = np.array(df.select('feature1').collect())
		res = np.array(df.select('feature2').collect()).flatten()
		gaussian_model(f1, res)
		perceptron_model(f0, res)
		sgdclassifier_model(f1, res)

#Preprocessing
def text_cleaning(a):
	rp = []
	if a == None:
		return
	for char in a:
		if char in string.punctuation:
			rp.append(' ')
		else:
			rp.append(char)
	#rp = [char for char in a if char not in string.punctuation]
	rp = ''.join(rp)
	return [word for word in rp.split() if word.lower not in stopwords.words('english')]


def gaussian_model(f1, res):
	trans = HashingVectorizer(analyzer=text_cleaning).fit(f1)
	bow = trans.transform(f1).toarray()
	
	#Saving in pickle
	with open('g_model.sav', 'rb') as f:
		model = pickle.load(f)
		
	#Calculating accuracy-Change
	'''
	cor = (res == model.predict(bow)).sum()
	accuracy = cor/(res.size)
	print('Gausian_Model Accuracy = %s' %(accuracy))
	'''
	cor = model.predict(bow)
	#print(cor)

	with open('graph_g.txt', 'a') as f:
		f.write('%s %s %s %s\n' %(precision_score(cor, res, pos_label = 'ham'), recall_score(cor, res,  pos_label = 'ham'), f1_score(cor, res, pos_label = 'ham'), accuracy_score(cor, res)))
	
	print("Confusion matrix: \n", confusion_matrix(cor, res))		
	print("Precision:", precision_score(cor, res, pos_label = 'ham'))
	print("Recall: ", recall_score(cor, res,  pos_label = 'ham'))
	print("F1 score: ", f1_score(cor, res, pos_label = 'ham'))
	print("Accuracy: ", accuracy_score(cor, res))
	
	print('----------------------------------------------------------------------')
	


def perceptron_model(f0, res):
	trans = HashingVectorizer(analyzer=text_cleaning).fit(f0)
	bow = trans.transform(f0).toarray()
	
	#Saving in pickle
	with open('p_model.sav', 'rb') as f:
		model = pickle.load(f)
		
	'''cor = (res == model.predict(bow)).sum()
	accuracy = cor/(res.size)
	print('Perceptron_Model Accuracy = %s' %(accuracy))'''
		
	cor = model.predict(bow)
	
	with open('graph_p.txt', 'a') as f:
		f.write('%s %s %s %s\n' %(precision_score(cor, res, pos_label = 'ham'), recall_score(cor, res,  pos_label = 'ham'), f1_score(cor, res, pos_label = 'ham'), accuracy_score(cor, res)))
		
	print("Confusion matrix: \n", confusion_matrix( cor, res))
	print("Precision:", precision_score(cor, res, pos_label = 'ham'))
	print("Recall: ", recall_score(cor, res,  pos_label = 'ham'))
	print("F1 score: ", f1_score(cor, res, pos_label = 'ham'))
	print("Accuracy: ", accuracy_score(cor, res))
	
	print('----------------------------------------------------------------------')
	
def sgdclassifier_model(f01, res):
	trans = HashingVectorizer(analyzer=text_cleaning).fit(f01)
	bow = trans.transform(f01).toarray()
	
	#Saving in pickle
	with open('sgd_model.sav', 'rb') as f:
		model = pickle.load(f) 
		
	#Calculating accuracy-Change
	'''cor = (res == model.predict(bow)).sum()
	accuracy = cor/(res.size)
	print('SGDClassifier_Model Accuracy = %s' %(accuracy))'''
		
	cor = model.predict(bow)
	
	with open('graph_sgd.txt', 'a') as f:
		f.write('%s %s %s %s\n' %(precision_score(cor, res, pos_label = 'ham'), recall_score(cor, res,  pos_label = 'ham'), f1_score(cor, res, pos_label = 'ham'), accuracy_score(cor, res)))
		
	print("Confusion matrix: \n", confusion_matrix( cor, res))
	print("Precision:", precision_score(cor, res, pos_label = 'ham'))
	print("Recall: ", recall_score(cor, res,  pos_label = 'ham'))
	print("F1 score: ", f1_score(cor, res, pos_label = 'ham'))
	print("Accuracy: ", accuracy_score(cor, res))
	print('----------------------------------------------------------------------')
	print('----------------------------------------------------------------------')




ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

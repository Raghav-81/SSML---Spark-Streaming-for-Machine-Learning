from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
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
		Kmeans(f1, res)


def convert(obj):
	if(obj == 'spam' or obj == '0' or obj == 0):
		return 0
	return 1

def Kmeans(f0, res):
	
	vectorizer = TfidfVectorizer(stop_words='english')
	features = vectorizer.fit_transform(f0.flatten())
	#print(features)
	#print("----------------")
	
	
	#Saving in pickle
	with open('km_model.sav', 'rb') as f:
		model = pickle.load(f)
	cor = model.labels_
	cor1 = np.array(list(map(convert, cor)))
	print(cor1)
	print("----------------")
	res1 = np.array(list(map(convert, res)))
	print(res1)
	print("----------------")
	
	
	
	
	print("Confusion matrix: \n", confusion_matrix(cor1, res1))
			
	print("Precision:", precision_score(cor1, res1, pos_label = 1))
	print("Recall: ", recall_score(cor1, res1,  pos_label = 1))
	print("F1 score: ", f1_score(cor1, res1, pos_label = 1))
	print("Accuracy: ", accuracy_score(cor1, res1))








ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron, SGDClassifier
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
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
sc = SparkContext(appName="main")
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
		
		#print(np.array(df.select('feature0').collect()))
		f0 = np.array(df.select('feature0').collect())
		f1 = np.array(df.select('feature1').collect())
		#f01 = np.array(df.select('feature0','feature1').collect())
		res = np.array(df.select('feature2').collect()).flatten()
		gaussian_model(f1, res)
		perceptron_model(f0, res)
		sgdclassifier_model(f1, res)
		print('DONE')

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
	model = GaussianNB().partial_fit(bow, res, classes=np.unique(res))
	
	#Saving in pickle
	with open('g_model.sav', 'wb') as f:
		pickle.dump(model, f)
		
def perceptron_model(f0, res):
	trans = HashingVectorizer(analyzer=text_cleaning).fit(f0)
	bow = trans.transform(f0).toarray()
	model = Perceptron().partial_fit(bow, res, classes=np.unique(res))
	
	#Saving in pickle
	with open('p_model.sav', 'wb') as f:
		pickle.dump(model, f)
	
def sgdclassifier_model(f01, res):
	trans = HashingVectorizer(analyzer=text_cleaning).fit(f01)
	bow = trans.transform(f01).toarray()
	model = SGDClassifier().partial_fit(bow, res, classes=np.unique(res))
	
	#Saving in pickle
	with open('sgd_model.sav', 'wb') as f:
		pickle.dump(model, f)	


#lines.foreachRDD(collect)

#lines.pprint()

ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

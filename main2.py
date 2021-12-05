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
lines = ssc.socketTextStream("localhost", 6102)
lines.foreachRDD(lambda rdd: collect(rdd))

def collect(inp):
	if not inp.isEmpty():
		df = spark.read.json(inp)
		df.show()

#lines.pprint()




ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

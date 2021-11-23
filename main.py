import findspark
findspark.init()
import pyspark
import json
from pyspark.sql.session import SparkSession

from pyspark import SparkContext
from pyspark.streaming import StreamingContext

# Create a local StreamingContext with two working thread and batch interval of 1 second
sc = SparkContext(appName="main")
ssc = StreamingContext(sc, 1)
spark=SparkSession(sc)
# Create a DStream that will connect to hostname:port, like localhost:9999
lines = ssc.socketTextStream("localhost", 6100)

lines.foreachRDD(lambda rdd: print(spark.read.json(rdd,multiLine=True)))



ssc.start()             # Start the computation
ssc.awaitTermination()  # Wait for the computation to terminate

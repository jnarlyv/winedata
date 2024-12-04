#importing necesarry libraries
import boto3
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import DoubleType, StructField, StructType
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.stat import Correlation

spark = SparkSession.builder \
    .appName("MLP Wine Training") \
    .getOrCreate()


# Creating a schema for our training and validation file
labels = [
     ('FixedAcidity',DoubleType()),
     ('VolatileAcidity',DoubleType()),
     ('CitricAcid',DoubleType()),
     ('ResidualSugar',DoubleType()),
     ('Chlorides',DoubleType()),
     ('FreeSulfurDioxide',DoubleType()),
     ('TotalSulfurDioxide',DoubleType()),
     ('Density',DoubleType()),
     ('pH',DoubleType()),
     ('Sulphates',DoubleType()),
     ('Alcohol',DoubleType()),
     ('Quality',DoubleType())
]

schema = StructType([StructField (x[0], x[1], True) for x in labels])

#Read the training data using Spark
trainData = spark.read.csv('s3a://mlptrain/TrainingDataset.csv', header=True, sep=";", schema=schema)

#Extracting all column
features = list(trainData.columns)

#Extracting the feature column
features.remove('Quality')


#Using vector assember to transform our data
assembler = VectorAssembler(inputCols=features, outputCol="features")
trainData = assembler.transform(trainData)

#Renaming the quality column as label
trainData = trainData.withColumnRenamed("Quality", "label")

#Using the standard scaler to scale our features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scaler_model = scaler.fit(trainData)

scaler_model.save("s3a://mlptrain/scaler_model")
trainData = scaler_model.transform(trainData)

#Using a multi layer perceptron classifier to make the predictions
layers = [11, 4, 2,10]

trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=32, stepSize= 0.01, labelCol="label",
                                     featuresCol="scaledFeatures",seed=1234)

model = trainer.fit(trainData)

#Save model for future use
model.save("s3a://mlptrain/mlp_model")

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

f1_score = evaluator.evaluate(model.transform(trainData))

#Printing F1 Score for train data
print(f"F1 Score on train data: {f1_score}\n")

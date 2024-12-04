#!/usr/bin/python3

import sys
import urllib.request
import tarfile
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScalerModel
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import DoubleType, StructField, StructType
import logging

# Constants
SCALER_MODEL_URL = "https://raw.githubusercontent.com/jnarlyv/winedata/main/model/scaler_model.tar.gz"
MODEL_URL = "https://raw.githubusercontent.com/jnarlyv/winedata/main/model/model.tar.gz"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_extract(url, output_dir):
    """Downloads and extracts a tar.gz file."""
    try:
        filename = url.split("/")[-1]
        logger.info(f"Downloading {url}...")
        urllib.request.urlretrieve(url, filename)
        
        logger.info(f"Extracting {filename}...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(path=output_dir)
    except Exception as e:
        logger.error(f"Error downloading or extracting {url}: {e}")
        sys.exit(1)

def main(input_csv_path):
    if not input_csv_path:
        print("Please provide a CSV file path as an argument.")
        sys.exit(1)

    # Initialize Spark session
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    logger.info("Spark session initialized.")

    # Define schema
    labels = [
        ('FixedAcidity', DoubleType()), ('VolatileAcidity', DoubleType()), 
        ('CitricAcid', DoubleType()), ('ResidualSugar', DoubleType()), 
        ('Chlorides', DoubleType()), ('FreeSulfurDioxide', DoubleType()), 
        ('TotalSulfurDioxide', DoubleType()), ('Density', DoubleType()), 
        ('pH', DoubleType()), ('Sulphates', DoubleType()), 
        ('Alcohol', DoubleType()), ('Quality', DoubleType())
    ]
    schema = StructType([StructField(name, dtype, True) for name, dtype in labels])

    # Load CSV data
    test_data = spark.read.csv(input_csv_path, header=True, sep=";", schema=schema)
    features = [col for col in test_data.columns if col != "Quality"]
    test_data = test_data.withColumnRenamed("Quality", "label")

    # Feature engineering
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    test_data = assembler.transform(test_data)

    # Download and load scaler model
    download_and_extract(SCALER_MODEL_URL, ".")
    scaler_model = StandardScalerModel.load("scaler_model")
    test_data = scaler_model.transform(test_data)

    # Download and load MLP model
    download_and_extract(MODEL_URL, ".")
    model = MultilayerPerceptronClassificationModel.load("mlp_model")

    # Predictions and evaluation
    result = model.transform(test_data)
    prediction_and_labels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="f1")
    f1_score = evaluator.evaluate(prediction_and_labels)
    logger.info(f"F1 Accuracy: {f1_score}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: script.py <csv_file_path>")
        sys.exit(1)
    
    input_csv_path = sys.argv[1]
    main(input_csv_path)


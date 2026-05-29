# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random

# Create Spark session
spark = SparkSession.builder.appName("RandomForestRealistic").getOrCreate()

# --------------------------------------
# STEP 1: Generate Random Dataset (with noise)
# --------------------------------------
data = []

for i in range(200):  # 200 records
    age = random.randint(18, 60)
    marks = random.randint(30, 100)

    # Add noise (not perfect rule)
    if marks + random.randint(-20, 20) > 60:
        label = 1
    else:
        label = 0

    data.append((age, marks, label))

columns = ["age", "marks", "label"]
df = spark.createDataFrame(data, columns)

print("Sample Data:")
df.show(10)

# --------------------------------------
# STEP 2: Feature Engineering
# --------------------------------------
assembler = VectorAssembler(
    inputCols=["age", "marks"],
    outputCol="features"
)

df = assembler.transform(df)
df = df.select("features", "label")

# --------------------------------------
# STEP 3: Train-Test Split
# --------------------------------------
train_data, test_data = df.randomSplit([0.7, 0.3])

# --------------------------------------
# STEP 4: Train Random Forest Model
# --------------------------------------
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=20,
    maxDepth=5
)

model = rf.fit(train_data)

# --------------------------------------
# STEP 5: Predictions
# --------------------------------------
predictions = model.transform(test_data)

print("Predictions:")
predictions.select("features", "label", "prediction").show(10)

# --------------------------------------
# STEP 6: Evaluation
# --------------------------------------
evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = evaluator.evaluate(predictions)

print("Model Accuracy:", accuracy)

# Stop Spark
spark.stop()

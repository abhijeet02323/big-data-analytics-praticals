# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random

# Create Spark session
spark = SparkSession.builder.appName("NaiveBayesRealistic").getOrCreate()

# --------------------------------------
# STEP 1: Create Realistic Dataset
# --------------------------------------
data = []

for i in range(300):
    age = random.randint(18, 60)
    income = random.randint(20000, 100000)
    experience = random.randint(0, 20)

    # Add noise + imperfect logic
    score = income/10000 + experience + random.uniform(-2, 2)

    if score > 10:
        label = 1
    else:
        label = 0

    data.append((age, income, experience, label))

columns = ["age", "income", "experience", "label"]
df = spark.createDataFrame(data, columns)

print("Sample Data:")
df.show(10)

# --------------------------------------
# STEP 2: Feature Engineering
# --------------------------------------
assembler = VectorAssembler(
    inputCols=["age", "income", "experience"],
    outputCol="features"
)

df = assembler.transform(df)
df = df.select("features", "label")

# --------------------------------------
# STEP 3: Train-Test Split
# --------------------------------------
train_data, test_data = df.randomSplit([0.7, 0.3])

# --------------------------------------
# STEP 4: Train Model
# --------------------------------------
nb = NaiveBayes(featuresCol="features", labelCol="label")
model = nb.fit(train_data)

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

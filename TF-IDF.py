from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF

#Create Spark Session
spark = SparkSession.builder.appName("TextPreprocessingTFIDF").getOrCreate()

#Create Dataset
data = [
    (0, "Spark is great for big data processing"),
    (1, "Machine learning with Spark is powerful"),
    (2, "Big data analytics is important"),
    (3, "Spark makes data processing fast")
]

columns = ["id", "text"]
df = spark.createDataFrame(data, columns)

print("Original Data:")
df.show(truncate=False)

#Tokenization
tokenizer = Tokenizer(inputCol="text", outputCol="words")
words_data = tokenizer.transform(df)

#Remove Stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
filtered_data = remover.transform(words_data)

#Term Frequency (TF)
hashingTF = HashingTF(
    inputCol="filtered_words",
    outputCol="rawFeatures",
    numFeatures=20
)

featurized_data = hashingTF.transform(filtered_data)


#Inverse Document Frequency (IDF)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)

tfidf_data = idf_model.transform(featurized_data)

#Show Result
print("Final TF-IDF Features:")
tfidf_data.select("text", "filtered_words", "features").show(truncate=False)

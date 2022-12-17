from pyspark.sql.functions import *  
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler  
from pyspark.ml import Pipeline  
from pyspark.ml.classification import LogisticRegression  
from pyspark.ml.evaluation import BinaryClassificationEvaluator  
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder  
from pyspark.sql.types import DoubleType  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt 

aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Gift_Card_v1_00.tsv.gz  . 
aws s3 cp s3://amazon-reviews-pds/tsv/amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv.gz  . 

hdfs dfs -put amazon_reviews_us_Gift_Card_v1_00.tsv.gz hdfs:/// 
hdfs dfs -put amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv.gz hdfs:/// 
hdfs dfs -ls hdfs:/// 

df1 = spark.read.csv("hdfs:///amazon_reviews_us_Gift_Card_v1_00.tsv.gz", sep='\t', header=True, inferSchema=True)   
df2 = spark.read.csv("hdfs:///amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv.gz", sep='\t', header=True, inferSchema=True)   
df = df1.union(df2)  

df = df.sample(.25) 
df = df.na.drop(subset=["star_rating", "review_body", "review_date", "vine", "product_category", "verified_purchase"])  
df = df.filter(df.product_category.isin(['Gift Card', 'Personal Care Appliances'])) 
Df.summary().show() 

df = df.withColumn('review_body_wordcount', size(split(col('review_body'), ' '))) 
df = df.withColumn("label", when(col("star_rating") > 3, 1.0).otherwise(0.0))  
df = df.withColumn("total_votes",df.total_votes.cast(DoubleType()))  
df = df.withColumn("review_body_wordcount",df.review_body_wordcount.cast(DoubleType()))  
df = df.drop("review_body", "review_headline", "marketplace", "customer_id", "review_id", "product_id", "product_parent", "product_title") 

trainingData, testData = df.randomSplit([0.7, 0.3], seed=3456) 

indexer = StringIndexer(inputCols=["product_category", "vine", "verified_purchase"], outputCols=["product_categoryIndex", "vineIndex", "verified_purchaseIndex"], handleInvalid="keep") 
encoder = OneHotEncoder(inputCols=["product_categoryIndex", "vineIndex", "verified_purchaseIndex" ], outputCols=["product_categoryVector", "vineVector", "verified_purchaseVector" ], dropLast=True, handleInvalid="keep") 
assembler = VectorAssembler(inputCols=["product_categoryVector", "vineVector", "verified_purchaseVector", "total_votes", "review_body_wordcount"], outputCol="features")  

lr = LogisticRegression(maxIter=10) 

reviewsPipe = Pipeline(stages=[indexer, encoder, assembler, lr])  

grid = ParamGridBuilder()  
grid = grid.addGrid(lr.regParam, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])  
grid = grid.addGrid(lr.elasticNetParam, [0, 0.5, 1])  
print('Number of models to be tested: ', len(grid)) 

evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC")  

cv = CrossValidator(estimator=reviewsPipe, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3, seed=789 )  
cv = cv.fit(trainingData) 

predictions = cv.transform(testData)  
auc = evaluator.evaluate(predictions)  
print('AUC:', auc) 
predictions.groupby('label').pivot('prediction').count().fillna(0).show()  

cm = predictions.groupby('label').pivot('prediction').count().fillna(0).collect()  
def calculate_precision_recall(cm): 
  tn = cm[0][1]  
  fp = cm[0][2]  
  fn = cm[1][1]  
  tp = cm[1][2]  
  precision = tp / ( tp + fp )  
  recall = tp / ( tp + fn )  
  accuracy = ( tp + tn ) / ( tp + tn + fp + fn )  
  f1_score = 2 * ( ( precision * recall ) / ( precision + recall ) )  
  return accuracy, precision, recall, f1_score 
print( calculate_precision_recall(cm) ) 

parammap = cv.bestModel.stages[3].extractParamMap() 
for p, v in parammap.items():  
  print(p, v)
  
import matplotlib.pyplot as plt 
plt.figure(figsize=(6,6))  
plt.plot([0, 1], [0, 1], 'r--')  
x = mymodel.summary.roc.select('FPR').collect()  
y = mymodel.summary.roc.select('TPR').collect()  
plt.scatter(x, y)  
plt.xlabel('False Positive Rate')  
plt.ylabel('True Positive Rate')  
plt.title("ROC Curve")  
plt.savefig("reviews_roc.png") 

coeff = mymodel.coefficients.toArray().tolist() 
var_index = dict() 
for variable_type in ['numeric', 'binary']:
  for variable in predictions.schema["features"].metadata["ml_attr"]["attrs"][variable_type]: 
    print("Found variable:", variable) 
    idx = variable['idx']  
    name = variable['name'] 
    var_index[idx] = name 
    
for i in range(len(var_index)):  
  print(i, var_index[i], coeff[i]) 

hdfs df -get hdfs:///reviews_roc.png 
aws s3 cp reviews_roc.png s3://tc-data 

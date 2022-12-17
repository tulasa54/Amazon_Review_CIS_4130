import io  
import pandas as pd  
import s3fs  
import boto3  
import matplotlib.pyplot as plt  
import seaborn as sns

from pyspark.sql.functions import col, isnan, when, count, udf, to_date, year, month, date_format, size, split 
from pyspark.ml.stat import Correlation 
from pyspark.ml.feature import VectorAssembler 
sc.setLogLevel("ERROR") 

df1 = spark.read.csv("hdfs:///amazon_reviews_us_Gift_Card_v1_00.tsv.gz", sep='\t', header=True, inferSchema=True)   
df2 = spark.read.csv("hdfs:///amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv.gz", sep='\t', header=True, inferSchema=True)   
df = df1.union(df2) 
df.printSchema() 

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["star_rating", "review_body"]]).show() 
df.select([count(when(col(c).isNull(), c)).alias(c) for c in ["review_date"]] ).show() 
df =df.na.drop(subset=["star_rating", "review_body", "review_date"])  

df = df.withColumn("review_year", year(col("review_date")))
df = df.withColumn("review_month", month(col("review_date")))
df = df.withColumn("review_yearmonth", date_format(col("review_date"), "yyyy-MM"))
df = df.withColumn('review_body_wordcount', size(split(col('review_body'), ' ')))
df.count() 

df.select("star_rating","helpful_votes","total_votes").summary("count", "min", "max", "mean").show() 

pdf=df.where(col("review_year")>2012).groupby("review_yearmonth").count().sort("review_yearmonth").toPandas()
pandaPlot = pdf.plot.bar('review_yearmonth','count')
pandaPlot.set(xlabel='Year-Month', ylabel='Number of Reviews')
pandaPlot.set(title='Number of Reviews by Year and Month')
pandaPlot.figure.set_tight_layout('tight')
pandaPlot.get_figure().savefig("review_count_byYearMonth.png") 

mdf = df.where(col("review_year") > 2012).groupby("review_yearmonth").count().sort("review_yearmonth").toPandas()
fig = plt.figure()
plt.bar(mdf['review_yearmonth'], df['count'])
plt.xlabel("Year-Month")
plt.ylabel("Number of Reviews")
plt.title("Number of Reviews by Year and Month")
plt.xticks(rotation=90, ha='right')
fig.tight_layout()
plt.savefig("review_count_byYearMonth_matplotlib.png")
plt.show() 

star_counts_df = df.groupby('star_rating').count().sort('star_rating').toPandas()
fig = plt.figure()
plt.bar(star_counts_df['star_rating'],star_counts_df['count'] )
plt.title("Review Count by Star Rating")
plt.savefig("frequency_star_rating.png") 


hdfs df -get hdfs:///frequency_star_rating.png 
aws s3 cp frequency_star_rating.png s3://tc-data 

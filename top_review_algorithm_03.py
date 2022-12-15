from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, IntegerType, DateType, FloatType
import pyspark.sql.functions as F
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Create Spark session
spark = SparkSession \
    .builder \
    .appName("top_reviews_by_category") \
    .getOrCreate()

# Specify input and output paths
directory = 'hdfs://localhost:9000/user/ds503/finalproject/data/'
output_path_top_products = 'hdfs://localhost:9000/user/ds503/finalproject/output/top_products_by_cat_algorithm03.csv'

# Define schema of input dataframe
schema = StructType([
                    StructField('marketplace', StringType(), True,  metadata={'marketplace': '2 letter country code of the marketplace where the review was written'}),
                    StructField('customer_id', StringType(), True, metadata={'customer_id': 'Random identifier that can be used to aggregate reviews written by a single author'}),
                    StructField('review_id', StringType(), True, metadata={'review_id': 'The unique ID of the review'}),
                    StructField('product_id', StringType(), True, metadata={'product_id': 'The unique Product ID the review pertains to'}),
                    StructField('product_parent', StringType(), True, metadata={'parent_id': 'Random identifier that can be used to aggregate reviews for the same product'}),
                    StructField('product_title', StringType(), True, metadata={'product_title': 'Title of the product'}),
                    StructField('product_category', StringType(), True, metadata={'product_category': 'Broad product category that can be used to group reviews (also used to group the dataset into coherent parts)'}),
                    StructField('star_rating', IntegerType(), True, metadata={'star_rating': 'The 1-5 star rating of the review'}),
                    StructField('helpful_votes', IntegerType(), True, metadata={'helpful_votes': 'Number of helpful votes'}),
                    StructField('total_votes', IntegerType(), True, metadata={'total_votes': 'Number of total votes the review received'}),
                    StructField('vine', StringType(), True, metadata={'vine': 'Review was written as part of the Vine program'}),
                    StructField('verified_purchase', StringType(), True, metadata={'verified_purchase': 'The review is on a verified purchase'}),
                    StructField('review_headline', StringType(), True, metadata={'review_headline': 'The title of the review'}),
                    StructField('review_body', StringType(), True, metadata={'review_body': 'The review text'}),
                    StructField('review_date', DateType(), True, metadata={'review_date': 'The date the review was written'})
                    ])


# Define schema of output dataframe
output_schema = StructType([
                            StructField('product_category', StringType(), True),
                            StructField('product_parent', StringType(), True),
                            StructField('product_title', StringType(), True),
                            StructField('bayes_rating', FloatType(), True),
                            StructField('avg_rating', FloatType(), True),
                            StructField('n_ratings', IntegerType(), True)
                            ])

# Create output dataframe
top_ratings = spark.createDataFrame([], output_schema)

# For each file in the input directory, perform the aggregation and output the results
for file in os.listdir(directory):
    if 'amazon' in file:
        # Get the product category
        category = os.path.basename(file)
        category = re.findall(r'_us_(.*)_v1_', category)[0]

        path = 'user/datasets/' + file

        # Load the data
        data = spark.read.option('header', 'true') \
                    .option('delimiter', '\t') \
                    .schema(schema) \
                    .csv(path) \
                    .select(F.col('product_parent'),
                            F.col('product_title'),
                            F.col('star_rating')) \
                    .dropna()

        # Define confidence parameter and get the average of all product reviews
        c_bayes = 200
        avg_all = data.agg(F.avg('star_rating')).collect()[0][0]

        # Aggregate data and find the Bayes average
        output = data.select(F.col('product_parent'), F.col('product_title'), F.col('star_rating')) \
                     .groupBy('product_parent', 'product_title') \
                     .agg(F.count('star_rating').alias('n_ratings'),
                          F.avg('star_rating').alias('avg_rating')) \
                     .filter(F.col('avg_rating') >= 4.5) \
                     .withColumn('bayes_rating',
                                 (F.col('avg_rating') * F.col('n_ratings') + c_bayes * avg_all) / (F.col('n_ratings') + c_bayes))

        # From the aggregated data, find the max Bayes average and find the entry that matches this
        max_bayes = output.agg(F.max('bayes_rating')).collect()[0][0]
        output = output.filter(F.col('bayes_rating') == max_bayes) \
                       .select(F.col('product_parent'),
                               F.col('product_title'),
                               F.col('bayes_rating'),
                               F.col('avg_rating'),
                               F.col('n_ratings')) \
                       .withColumn('product_category', F.lit(category))

        # Append the entry to the output dataframe
        top_ratings = top_ratings.unionByName(output)


# Write the results to disk
top_ratings.coalesce(1).write \
                       .option('header', 'true') \
                       .csv(output_path_top_products)
import nltk
nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# import packages
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pyspark.sql import SparkSession
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import sys
import time
# initilize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def collect_tokens(review_rdd, MIN_TOKEN_FREQUENCY=500, MIN_DOCUMENT_FREQUENCY=100):   
    # split each row with tab and get review column 
    def process(x):    
        value = x.value
        col = value.split("\t")
        review = col[-2]
        
        # casefold reviews to all lower cases
        words = word_tokenize(review.casefold())
        # lemmatize words and leave with only alphabets
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if  word.isalpha()]
        # filter with stopwords and length>2
        filtered_list = [word for word in lemmatized_words if word not in stop_words and len(word) > 2]    
    
        # dictionary of {word: word count}
        dic = {}
        for word in filtered_list:
            if word not in dic:
                dic[word] = 0        
            dic[word] += 1
        # (word, (word count, review count(=1))
        return [(word, (dic[word], 1)) for word in dic]

    print("============ PreCollecting Tokens ===========")
    print("MAX_TOKEN_FREQUENCY:", MIN_TOKEN_FREQUENCY)
    print("MIN_DOCUMENT_FREQUENCY:", MIN_DOCUMENT_FREQUENCY)
    token_file = f"{path_prefix}/tmp/{CATEGORY}/{RATING}/all_tokens"
    print("Saving tokens to", token_file)
    
    # reduceByKey:  (key(word), (sum word count, sum review count))
    # Filter with MIN_TOKEN_FREQUENCY/MIN_DOCUMENT_FREQUENCY/MAX_TOKEN_FREQUENCY
    # map with index and return csv file path 
    review_rdd.flatMap(lambda x: process(x))\
            .reduceByKey(lambda x, y : (x[0] + y[0], x[1] + y[1]))\
            .filter(lambda x: x[1][0] >= MIN_TOKEN_FREQUENCY 
                                and x[1][1] >= MIN_DOCUMENT_FREQUENCY)\
            .map(lambda x: x[0]).zipWithIndex()\
            .toDF().write.csv(token_file)    
    return token_file

def encoding(review_rdd, token_file):
    print("="*20 + "encoding" + "="*20)
    # word_to_idx: {word: index}; idx_to_word: {index: word}
    word_to_idx = {}
    idx_to_word = {}

    all_tokens = spark.read.csv(token_file).rdd.collect()
    
    for token in all_tokens:    
        word,idx = token
        # get word by indx, get index by word    
        word_to_idx[word] = idx
        idx_to_word[int(idx)] = word

    def hash_token(x):
        value = x.value
        col = value.split("\t")
        # casefold reviews to all lower cases
        r_id, review = col[2], col[-2].casefold()

        # tokenize reviews    
        words = word_tokenize(review)
        # lemmatize words and leave with only alphabets
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words if  word.isalpha()]
        # filter with lemmatized words and word_to_inx list
        filtered_tokens = [word_to_idx[word] for word in lemmatized_words if word in word_to_idx]    
        
        return (r_id , " ".join(filtered_tokens))

    file = f"{path_prefix}/tmp/{CATEGORY}/{RATING}/encode"
    print(f"Hashing Beauty review with rating {RATING} to ", file)
    review_rdd\
        .map(lambda x: hash_token(x))\
        .toDF().write.csv(file)
    
    return file, idx_to_word

# compute tf_idf
def tf_idf(encoding_file, N):
    print("="*20 + "calcualting tf_idf" + "="*20)
    # read files from encoded file path generated 
    encoding_rdd = spark.read.csv(encoding_file).rdd
    # remove last empty row and convert index(strings) to list of integers
    encoding_rdd = encoding_rdd.filter(lambda x: x[1])\
        .map(lambda x: [int(n) for n in x[1].split(" ")])

    # calculate idf 
    # sum number of reviews an index appears then log(N/sum)
    idf = encoding_rdd.flatMap(lambda x: [(index, 1) for index in set(x)])\
            .reduceByKey(lambda x, y: x+y)\
            .map(lambda x:  (x[0],  math.log(N/x[1])))
    file = f"{path_prefix}/tmp/{CATEGORY}/{RATING}/idf"

    print("="*20 +"saving idf"+"="*20)
    idf.toDF().write.csv(file)

    idf = idf.collect()
    # (reviewid, indices) 
    #   => (index, 1) 
    #   => (index, DF) 
    #   => (index, idf_val)

    index_idf = [0]*len(idf)
    for entry in idf:
        index, idf_val = entry
        index_idf[index] = idf_val
    
    print("size", len(index_idf))

    # calculate tf-idf
    def preprocessTFIDF(indices):          
        length = len(indices)
            
        dic = {}
        for index in indices:
            if index not in dic:
                dic[index] = 0        
            dic[index] += 1
        
        # tf =  [(token, dic[token]/length)) for token in dic]
        tf_idf =  [(index, dic[index]/length * index_idf[index]) for index in dic]
        # leave with top 10 tf-idf values
        top_10_tf_idf = sorted(tf_idf, key=lambda x: -x[1])[:10]
        return [(index[0], 1) for index in top_10_tf_idf]


    tf_idf_rdd = encoding_rdd.flatMap(lambda x, : preprocessTFIDF(x))\
                            .reduceByKey(lambda x, y : x+y)\
                            .sortBy(lambda x: -x[1])

    # print(all_tf_idf.first())
    file = f"{path_prefix}/tmp/{CATEGORY}/{RATING}/tf_idf"
    print("saving to ", file)
    tf_idf_rdd.toDF().write.csv(file)
    return tf_idf_rdd

# draw wordcloud
def draw_word_cloud(tf_idf_rdd, idx_to_word, top=1000):
    print("="*20 + "draw_word_cloud" + "="*20)
    tf_idf_top = tf_idf_rdd.take(top)

    frequencies = {}
    for index, index_occurrences in tf_idf_top:
        frequencies[idx_to_word[index]] = index_occurrences

    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                min_font_size = 10).generate_from_frequencies(frequencies)

    fig = plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    
    file = f"{path_prefix}/result/{CATEGORY}_{RATING}_word_cloud.jpg"
    fig.savefig(file, bbox_inches='tight', dpi=300)
    plt.show()


def main(review_file = "./data/amazon_reviews_us_Books_v1_02.tsv",
            r = 5, 
            MIN_TOKEN_FREQUENCY = 500,
            MIN_DOCUMENT_FREQUENCY = 50,            
            ):
    global CATEGORY, RATING, spark
    spark = SparkSession.builder.appName("wordcount").getOrCreate()
    
    RATING = r    
    CATEGORY = review_file.split('_')[3] if "_" in review_file else "ALL"

    def clean(x, r):
        value = x.value
        # get dataset
        if value.startswith("marketplace"):
            return False
        if r == -1:
            return True
        # split with tab
        rating = value.split("\t")[7]            
        return  int(rating) == r

    review_rdd = spark.read.text(review_file).rdd.filter(lambda x: clean(x, RATING))
    
    # total number of reivews
    N = review_rdd.count()        
    print("N:", N)

    # record time for each step
    t_0 = time.time()
    word_index_path = collect_tokens(review_rdd, MIN_TOKEN_FREQUENCY, MIN_DOCUMENT_FREQUENCY)   
    t_1 = time.time()

    print("Time used for collecting token is", t_1 - t_0)

    encoded_path, idx_to_word = encoding(review_rdd, word_index_path)

    t_2 = time.time()
    print("Time used for encoding is", t_2 - t_1)

    del review_rdd

    tf_idf_rdd = tf_idf(encoded_path, N)
    t_3 = time.time()
    print("Time used for calculate tf_idf is", t_3 - t_2)

    draw_word_cloud(tf_idf_rdd, idx_to_word, 500)

if __name__=="__main__":
    
    env = {
        "local": ".",
        "docker": "/user/ds503/finalproject"
    }
    global path_prefix
    path_prefix = env[sys.argv[1]]
    main(path_prefix+"/data", -1)

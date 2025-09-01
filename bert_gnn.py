# Databricks notebook source
pip install nltk

# COMMAND ----------

pip install tensorflow


# COMMAND ----------

# # file locations 
# file_location_all = "/FileStore/final_project_dataset/all.tsv"
# file_location_entities = "/FileStore/final_project_dataset/entities.tsv"
# file_location_relations = "/FileStore/final_project_dataset/relations.tsv"
# file_type = "csv"

# # CSV options
# infer_schema = "false"
# first_row_is_header = "true"
# delimiter = "\t"

# # CSV options are applied. For other types, these options will be ignored.
# df_all = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_all)
# df_entities = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_entities)
# df_relations = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_relations)
# df_all1 = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_all)

# COMMAND ----------

# file locations 
file_location_text = "/FileStore/final_project_dataset/entity2text.tsv" 
# "FileStore/final_project_dataset/entity2textlong.tsv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = "\t"

# CSV options are applied. For other types, these options will be ignored.
df_text = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)

# show the created dataframes in record details
display(df_text)

# COMMAND ----------

#参考草稿# 
# # file locations 
file_location_all = "/FileStore/final_project_dataset/all.tsv"
file_location_entities = "/FileStore/final_project_dataset/entities.tsv"
file_location_relations = "/FileStore/final_project_dataset/relations.tsv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = "\t"

# CSV options are applied. For other types, these options will be ignored.
df_long = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_all)
df_entities = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_entities)
df_relations = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_relations)

df_all1 = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_all)

# file locations - long text
file_location_text = "/FileStore/final_project_dataset/entity2textlong.tsv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = "\t"

# CSV options are applied. For other types, these options will be ignored.
df_text = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)

# show the created dataframes in record details
# display(df_text)
df_long = df_long.join(df_text, df_long["head"] == df_text["entity"], "left")
df_long = df_long.select("head", "relation", "tail", "text").withColumnRenamed("text", "head_text")
df_long = df_long.join(df_text, df_long["tail"] == df_text["entity"], "left")
df_long = df_long.select("head", "relation", "tail", "head_text", "text").withColumnRenamed("text", "tail_text")
# from pyspark.sql.functions import *
df_long = df_long.withColumn('joined_text2', concat(col('head_text'), lit(' '), col('tail_text')))

# COMMAND ----------

# df_all = df_all.join(df_text, df_all["head"] == df_text["entity"], "left")
# df_all = df_all.select("head", "relation", "tail", "text").withColumnRenamed("text", "head_text")
# df_all = df_all.join(df_text, df_all["tail"] == df_text["entity"], "left")
# df_all = df_all.select("head", "relation", "tail", "head_text", "text").withColumnRenamed("text", "tail_text")

# COMMAND ----------

# from pyspark.ml.feature import *
# string_indexer = StringIndexer(inputCol="relation", outputCol="indexed_relation")
# string_indexer = string_indexer.fit(df_all)
# df_all = string_indexer.transform(df_all)

# COMMAND ----------

# from pyspark.sql.functions import *
# df_all = df_all.withColumn('joined_text', concat(col('head_text'), lit(' '), col('tail_text')))

# COMMAND ----------

# outputPath = "/FileStore/final_project_dataset/cleaned_all_short.parquet"
# df_all.write.mode("overwrite").parquet(outputPath)

# COMMAND ----------

filePath = "/FileStore/final_project_dataset/cleaned_all_short.parquet"
# filePath = "/FileStore/final_project_dataset/.parquet"
df_short = spark.read.parquet(filePath)

display(df_short)
df_short.count()

# COMMAND ----------

#提取30大类并index
from pyspark.sql.functions import split
from pyspark.ml.feature import *

df_short = df_short.withColumn("upper", split(df_short["relation"], "/").getItem(1))
string_indexer = StringIndexer(inputCol="upper", outputCol="indexed_upper")
string_indexer = string_indexer.fit(df_short)
df_short = string_indexer.transform(df_short)
display(df_short)

# COMMAND ----------

df_short.groupby("upper").count().orderBy("count").show() 

# COMMAND ----------

# 统计df_short下，每个upper列的值有多少unique的relation列的值
# from pyspark.sql.functions import *

# df_short.groupBy("upper").agg(count.Distinct("relation").alias("unique_relations"))
from pyspark.sql.functions import count, col

df_short.groupBy("upper").agg(count(col("relation")).alias("unique_relations"))


# COMMAND ----------

df_short = df_short.select("*").toPandas()

# COMMAND ----------


from tensorflow.keras.preprocessing.text import Tokenizer

max_words = 100000
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@', lower=True)
tokenizer.fit_on_texts(df_short['joined_text'].values)
word_index = tokenizer.word_index
print(f"length of dictionary {len(word_index)}")

# COMMAND ----------

from tensorflow.keras.utils import pad_sequences
import pandas as pd
import numpy as np


max_length = 128
all_texts = tokenizer.texts_to_sequences(df_short['joined_text'].values)
all_texts = pad_sequences(all_texts, maxlen=max_length)
all_labels = pd.get_dummies(df_short['indexed_upper']).values


# COMMAND ----------

from sklearn.model_selection import train_test_split
seed=42
train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=0.2, random_state=seed)

# COMMAND ----------

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import SpatialDropout1D

embedding_dim = 100
model = Sequential()
model.add(Embedding(max_words, embedding_dim)) 
model.add(LSTM(100)) 
model.add(Dense(30, activation='softmax'))#这里的237我改30应该没错啊。这是最后一层，是标签种类个数
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])

# COMMAND ----------

df_short["upper"].value_counts()#这个

# COMMAND ----------

trainHistory = model.fit(train_texts, train_labels, epochs=4, batch_size=128, validation_split=0.2)

# COMMAND ----------

# trainHistory.history

# COMMAND ----------

print("evaluate on test data")  
results = model.evaluate(test_texts, test_labels, batch_size=128)
print(f"Testing results: cross-entropy loss = {results[0]}, accuracy = {results[1]}, precision = {results[2]}, recall = {results[3]}")

# COMMAND ----------

# file locations 
file_location_text = "/FileStore/final_project_dataset/hold_wo_label.tsv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = "\t"

# CSV options are applied. For other types, these options will be ignored.
infer_df = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)

# show the created dataframes in record details
display(infer_df)

# COMMAND ----------

infer_df = infer_df.join(df_text, infer_df["head"] == df_text["entity"], "left")
infer_df = infer_df.select("head", "relation", "tail", "text").withColumnRenamed("text", "head_text")
infer_df = infer_df.join(df_text, infer_df["tail"] == df_text["entity"], "left")
infer_df = infer_df.select("head", "relation", "tail", "head_text", "text").withColumnRenamed("text", "tail_text")

# COMMAND ----------

from pyspark.sql.functions import *

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pyspark.sql.types import DoubleType
import numpy as np

infer_df = infer_df.withColumn('joined_text', concat(col('head_text'), lit(' '), col('tail_text')))


# 将 DataFrame 转换为 Pandas DataFrame
infer_df_pd = infer_df.toPandas()

#tokenizer = Tokenizer()
#tokenizer.fit_on_texts(infer_df_pd['joined_text'])

max_length = 128

test_texts = tokenizer.texts_to_sequences(infer_df_pd['joined_text'])
test_texts = pad_sequences(test_texts, maxlen=max_length)

# 使用模型进行预测
predicted_labels = model.predict(test_texts)

# 将预测结果转换为标签值
predicted_labels = np.argmax(predicted_labels, axis=1)

# 将预测结果添加到数据集中
infer_df_pd['predicted_relation'] = predicted_labels

# 将 Pandas DataFrame 转换为 Spark DataFrame
infer_df = spark.createDataFrame(infer_df_pd)

# 输出预测结果
infer_df.show()


# COMMAND ----------

#别select
labelConverter = IndexToString(inputCol="predicted_relation", outputCol="final_relation",labels=string_indexer.labels)
infer_df = labelConverter.transform(infer_df)
display(infer_df)
# infer_df = infer_df.select("head", "final_relation", "tail")


# COMMAND ----------

# display(infer_df)

# COMMAND ----------

# infer_df_pd2 = infer_df.toPandas()
# # infer_df_pd2.to_csv('output_infer_df.csv', index=False) #这个，然后检查下csv在不在你目录
# display(infer_df_pd2)
# # infer_df_pd2.to_csv('/FileStore/final_project_dataset/output_infer_df.csv', index=False) 



# COMMAND ----------

# # pd.read_csv('output_infer_df.csv', sep='\t')
# pd.read_csv('/FileStore/final_project_dataset/output_infer_df.csv', sep='\t')

# COMMAND ----------

# import pandas as pd
# infer_df_pd3 = pd.read_csv('/FileStore/final_project_dataset/output_infer_df.csv')
# infer_df_pd3.head() #检查下是否正常 

# COMMAND ----------

# infer_df.write.csv('/FileStore/final_project_dataset/output322.tsv', header=True)

from pyspark.sql.types import StringType

infer_df = infer_df.withColumn('relation', infer_df['relation'].cast(StringType()))
# infer_df.write.csv('/FileStore/final_project_dataset/outputq.tsv', header=True)

# COMMAND ----------

display(infer_df)

# COMMAND ----------

import pandas as pd#别管文件读取了，直接跑吧
#目标文档中的大类
# Read the output.tsv file
# df2 = pd.read_csv('/FileStore/final_project_dataset/outputq.tsv', sep='\t')
df2 = infer_df.toPandas()#OK
# Define the specified list
# specified_list = ["dataworld", "broadcast", "ice_hockey", "baseball", "language", "celebrities", "medicine", "time", "american_football", "travel", "food", "military", "user", "business", "influence"]#, "media_common"] #15个样本量在3000以下的大关系
# Create a new column and mark rows based on the specified list
df2['new_column'] = df2['final_relation'].apply(lambda x: 1 if x in specified_list else 0)

# Count the number of rows marked as 1
count = df2['new_column'].sum()

count
# display(df2)#run run
#pao

# COMMAND ----------

# Rename the 'final_relation' column to 'upper_relation' in df2
df2 = df2.rename(columns={'final_relation': 'upper_relation'})
display(df2)

# COMMAND ----------

# df2 = pd.read_csv('/output2.tsv', encoding='utf-8')

# COMMAND ----------

# Initialize an empty list to store unique relation values
unique_relations = []

# For loop to iterate through each specified relation in specified list
for i in specified_list:
    # Filter df_short DataFrame for rows where upper column is equal to i
    sub_df = df_short[df_short['upper'] == i]
    # Get unique values of relation column in sub_df and add them to unique_relations list
    
    unique_relations.append(sub_df['relation'].unique())

# Display the unique relation values
unique_relations

# COMMAND ----------



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.utils import pad_sequences
import pandas as pd
import numpy as np

# COMMAND ----------

#读取之前的结果，把我们预测是小众关系的
# file locations 
file_location_text = "/FileStore/final_project_dataset/A_B_attempt1.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# CSV options are applied. For other types, these options will be ignored.
infer_df322 = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)

# show the created dataframes in record details
display(infer_df322)
# infer_df_pd3 = pd.read_csv('output_infer_df.csv')

# COMMAND ----------

infer_df322.count()

# COMMAND ----------

infer_df_pd3 = df2
infer_df322 = infer_df322.toPandas()#这是之前的结果
infer_df_pd3 = infer_df_pd3.merge(infer_df322[['head', 'tail', 'relation']], on=['head', 'tail'], how='left')
infer_df_pd3['relation'] = infer_df322['relation']
display(infer_df_pd3)#在我们的大类文档上显示了之前预测的小众关系


# COMMAND ----------

infer_df_pd3.count()

# COMMAND ----------

infer_df_pd3.drop(columns=['relation_x', 'relation_y'], inplace=True) #补救一下 #已经drop了 好44、45！

# COMMAND ----------

#定义
#防止过拟合早停
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

# model.fit(train_texts, train_labels, epochs=50, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

from keras.callbacks import ModelCheckpoint

# 保存模型权重的路径
# checkpoint_path = '/FileStore/huggingface_models/model_weights_'+i+'.h5'
# model_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path, save_best_only=True, mode='min')

# 训练模型并使用ModelCheckpoint回调
# model.fit(train_texts, train_labels, epochs=50, batch_size=128, validation_split=0.2, callbacks=[model_checkpoint, early_stopping])

# model.load_weights(checkpoint_path) #ok！

# COMMAND ----------

# filePath = "/FileStore/final_project_dataset/cleaned_all_long.parquet"
# # filePath = "/FileStore/final_project_dataset/.parquet"
# df_long = spark.read.parquet(filePath)
# df_long.rename(columns={'joined_text': 'join_text2'}, inplace=True)
# # display(df_long)

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.getOrCreate()

# Define the file path
filePath = "/FileStore/final_project_dataset/cleaned_all_long.parquet"

# Read the parquet file into a DataFrame
df_long = spark.read.parquet(filePath)

# Rename the 'joined_text' column to 'join_text2'
df_long = df_long.withColumnRenamed('joined_text', 'joined_text2')

# Display the DataFrame
df_long.show()

# COMMAND ----------

# Convert df_short from Pandas DataFrame to Spark DataFrame
df_short_spark = spark.createDataFrame(df_short)

# Join df_short_spark with df_long on 'head' and 'tail'
df_short2 = df_short_spark.join(df_long.select('head', 'tail', 'joined_text2'), ['head', 'tail'], 'left')

# Rename the 'joined_text2' column to 'joined_text' in df_short2
# df_short2 = df_short2.withColumnRenamed('joined_text2', 'joined_text')
display(df_short2)

# COMMAND ----------

df_long = df_long.toPandas() #ok
# df_short.rename(columns={'joined_text': 'join_text1'}, inplace=True)
df_short2 = df_short.merge(df_long[['head', 'tail', 'joined_text2']], on=['head', 'tail'], how='left')
# df_short2.rename(columns={'joined_text': 'joined_text2'}, inplace=True)

# COMMAND ----------

display(df_short) 
df_short.count()

# COMMAND ----------

sub_to_predict.head()
from pyspark.sql import SparkSession

# 创建 SparkSession
spark = SparkSession.builder.getOrCreate()

# 重命名 sub_to_predict 的列
# sub_to_predict = spark.createDataFrame(sub_to_predict)
sub_to_predict = sub_to_predict.withColumnRenamed("head", "head_key").withColumnRenamed("tail", "tail_key")

# 执行双键连接
joined_df = sub_to_predict.join(df_text, (sub_to_predict.head_key == df_text.tail) & (sub_to_predict.tail_key == df_text.head), "left")

# 显示连接后的结果
joined_df.show()

# COMMAND ----------

update_result_df = pd.DataFrame(columns=['head', 'tail', 'new_predict'])

# COMMAND ----------

infer_df_pd4 = infer_df_pd3.copy()
infer_df_pd4 = spark.createDataFrame(infer_df_pd4)

# file locations - long text
file_location_text = "/FileStore/final_project_dataset/entity2textlong.tsv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = "\t"

# CSV options are applied. For other types, these options will be ignored.
df_text = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)



# COMMAND ----------

# show the created dataframes in record details
# display(df_text)
infer_df_pd4 = infer_df_pd4.join(df_text, infer_df_pd4["head"] == df_text["entity"], "left")
infer_df_pd4 = infer_df_pd4.withColumnRenamed("text", "head_text2")
infer_df_pd4 = infer_df_pd4.drop("entity")
infer_df_pd4 = infer_df_pd4.join(df_text, infer_df_pd4["tail"] == df_text["entity"], "left")
infer_df_pd4 = infer_df_pd4.withColumnRenamed("text", "tail_text2")
infer_df_pd4 = infer_df_pd4.drop("entity")
# from pyspark.sql.functions import *
infer_df_pd4 = infer_df_pd4.withColumn('joined_text2', concat(col('head_text2'), lit(' '), col('tail_text2')))
display(infer_df_pd4) 
infer_df_pd4 = infer_df_pd4.toPandas()

# COMMAND ----------

infer_df_pd4['new_predict'] = None
# infer_df_pd4.update(sub_to_predict[['head', 'tail', 'new_predict']].set_index(['head', 'tail']))

# COMMAND ----------

#top22
from sklearn.preprocessing import LabelEncoder

result_list = []
result_list2 = []
#18 个样本量小于4000的大类标签
specified_list_for_test = ["celebrities", "dataworld", "military"]
specified_list = ["media_common","dataworld", "broadcast", "ice_hockey", "baseball", "language", "celebrities", "medicine", "time", "american_football", "travel", "food", "military", "user", "business", "influence","government","common","olympics", "tv", "soccer", "organization"]#top16+6=top22
# infer_df_pd4 = infer_df_pd3.copy()

#long text
infer_df_pd4['new_predict'] = None


for i in specified_list: #specified_list:#对于每个小众大类进行模型训练，在目标文档上预测
    # Filter df_short DataFrame for rows where upper column is equal to i
    sub_df = df_short2[df_short2['upper'] == i]
    label_encoder = LabelEncoder()
    sub_df['indexed_subrelation'] = label_encoder.fit_transform(sub_df['relation'])
    num_labels = len(sub_df['indexed_subrelation'].unique())
    # display(sub_df)
    # break #以上测试indexer
    tokenizer = Tokenizer(num_words=100000, filters='!"#$%&()*+,-./:;<=>?@', lower=True)
    tokenizer.fit_on_texts(df_short2['joined_text2'].values)
    all_texts = tokenizer.texts_to_sequences(sub_df['joined_text2'].values)
    all_texts = pad_sequences(all_texts, maxlen=128)
    all_labels = pd.get_dummies(sub_df['indexed_subrelation']).values
    train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=0.2, random_state=seed) #目前sub_df为train
    # break #测试tokenizer
    embedding_dim = 100
    model = Sequential()#
    model.add(Embedding(max_words, embedding_dim)) 
    model.add(LSTM(100)) 
    model.add(Dense(num_labels, activation='softmax'))#这里的数字是该大类下的小类数量
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
    # break #测试模型定义
    checkpoint_path = '/FileStore/huggingface_models/model_weights_'+i+'.keras'#'.h5'
    model_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path, save_best_only=True, mode='min')
    trainHistory = model.fit(train_texts, train_labels, epochs=10, batch_size=128, validation_split=0.2, callbacks=[model_checkpoint])#, early_stopping
    result_list2.append(trainHistory.history)
    model.load_weights(checkpoint_path)
    results = model.evaluate(test_texts, test_labels, batch_size=256)
    msg = f"{i}: cross-entropy loss = {results[0]}, accuracy = {results[1]}, precision = {results[2]}, recall = {results[3]}"
    print(msg)
    result_list.append(msg)
    # break #测试单个大类可以跑通
    #还差一步用模型predict
    sub_to_predict = infer_df_pd4[infer_df_pd4['upper_relation'] == i]

    # Preprocess text data
    test_texts = tokenizer.texts_to_sequences(sub_to_predict['joined_text2'])
    test_texts = pad_sequences(test_texts, maxlen=max_length)

    # Use the model to make predictions
    predicted_labels = model.predict(test_texts)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    # sub_df['text_labels'] = label_encoder.inverse_transform(sub_df['indexed_subrelation'])
    predicted_labels = label_encoder.inverse_transform(predicted_labels)

    # Add the predicted labels as a new column 'new_predict'
    sub_to_predict['new_predict'] = pd.Series(predicted_labels, index=sub_to_predict.index)#predicted_labels
    update_result_df = pd.concat([update_result_df , sub_to_predict], ignore_index=True)
    # break #测试单个大类可以跑通，预测格式正确
    # if i == specified_list_for_test[1]:
    #     break #测试前2个大类可以跑通 

print(result_list)
print(result_list2)
#最后保存/导出infer_df_pd4
display(update_result_df)
# display(infer_df_pd4)


# COMMAND ----------

#     # Use the model to make predictions
#     predicted_labels = model.predict(test_texts)
#     predicted_labels = np.argmax(predicted_labels, axis=1)
#     # sub_df['text_labels'] = label_encoder.inverse_transform(sub_df['indexed_subrelation'])
#     predicted_labels = label_encoder.inverse_transform(predicted_labels)

#     # Add the predicted labels as a new column 'new_predict'
#     sub_to_predict['new_predict'] = pd.Series(predicted_labels, index=sub_to_predict.index)#predicted_labels
#     update_result_df = pd.concat([update_result_df , sub_to_predict], ignore_index=True)
#     # break #测试单个大类可以跑通，预测格式正确
#     # if i == specified_list_for_test[1]:
#     #     break #测试前2个大类可以跑通 

# print(result_list)
# print(result_list2)
# #最后保存/导出infer_df_pd4
# display(update_result_df)
# # display(infer_df_pd4)

# COMMAND ----------


#     # Use the model to make predictions
#     predicted_labels = model.predict(test_texts)
#     predicted_labels = np.argmax(predicted_labels, axis=1)
#     # sub_df['text_labels'] = label_encoder.inverse_transform(sub_df['indexed_subrelation'])
#     predicted_labels = label_encoder.inverse_transform(predicted_labels)

#     # Add the predicted labels as a new column 'new_predict'
#     sub_to_predict['new_predict'] = pd.Series(predicted_labels, index=sub_to_predict.index)#predicted_labels
#     update_result_df = pd.concat([update_result_df , sub_to_predict], ignore_index=True)
#     # break #测试单个大类可以跑通，预测格式正确
#     # if i == specified_list_for_test[1]:
#     #     break #测试前2个大类可以跑通 

# print(result_list)
# print(result_list2)
# #最后保存/导出infer_df_pd4
# display(update_result_df)
# # display(infer_df_pd4)

# COMMAND ----------


# #历史准确率留底，别跑这个
# from sklearn.preprocessing import LabelEncoder
# result_list = []
# result_list2 = []
# #18 个样本量小于4000的大类标签
# specified_list_for_test = ["celebrities", "dataworld", "military"]
# specified_list = ["media_common","dataworld", "broadcast", "ice_hockey", "baseball", "language", "celebrities", "medicine", "time", "american_football", "travel", "food", "military", "user", "business", "influence"]#,"government","common"
# # infer_df_pd4 = infer_df_pd3.copy()

# #long text
# infer_df_pd4['new_predict'] = None


# for i in specified_list: #specified_list:#对于每个小众大类进行模型训练，在目标文档上预测
#     # Filter df_short DataFrame for rows where upper column is equal to i
#     sub_df = df_short2[df_short2['upper'] == i]
#     label_encoder = LabelEncoder()
#     sub_df['indexed_subrelation'] = label_encoder.fit_transform(sub_df['relation'])
#     num_labels = len(sub_df['indexed_subrelation'].unique())
#     # display(sub_df)
#     # break #以上测试indexer
#     tokenizer = Tokenizer(num_words=100000, filters='!"#$%&()*+,-./:;<=>?@', lower=True)
#     tokenizer.fit_on_texts(df_short2['joined_text2'].values)
#     all_texts = tokenizer.texts_to_sequences(sub_df['joined_text2'].values)
#     all_texts = pad_sequences(all_texts, maxlen=128)
#     all_labels = pd.get_dummies(sub_df['indexed_subrelation']).values
#     train_texts, test_texts, train_labels, test_labels = train_test_split(all_texts, all_labels, test_size=0.2, random_state=seed) #目前sub_df为train
#     # break #测试tokenizer
#     embedding_dim = 100
#     model = Sequential()#
#     model.add(Embedding(max_words, embedding_dim)) 
#     model.add(LSTM(100)) 
#     model.add(Dense(num_labels, activation='softmax'))#这里的数字是该大类下的小类数量
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])
#     # break #测试模型定义
#     checkpoint_path = '/FileStore/huggingface_models/model_weights_'+i+'.keras'#'.h5'
#     model_checkpoint = ModelCheckpoint(monitor='val_loss', filepath=checkpoint_path, save_best_only=True, mode='min')
#     trainHistory = model.fit(train_texts, train_labels, epochs=10, batch_size=128, validation_split=0.2, callbacks=[model_checkpoint])#, early_stopping
#     result_list2.append(trainHistory.history)
#     model.load_weights(checkpoint_path)
#     results = model.evaluate(test_texts, test_labels, batch_size=256)
#     msg = f"{i}: cross-entropy loss = {results[0]}, accuracy = {results[1]}, precision = {results[2]}, recall = {results[3]}"
#     print(msg)
#     result_list.append(msg)
#     # break #测试单个大类可以跑通
#     #还差一步用模型predict
#     sub_to_predict = infer_df_pd4[infer_df_pd4['upper_relation'] == i]

#     # Preprocess text data
#     test_texts = tokenizer.texts_to_sequences(sub_to_predict['joined_text2'])
#     test_texts = pad_sequences(test_texts, maxlen=max_length)

#     # Use the model to make predictions
#     predicted_labels = model.predict(test_texts)
#     predicted_labels = np.argmax(predicted_labels, axis=1)
#     # sub_df['text_labels'] = label_encoder.inverse_transform(sub_df['indexed_subrelation'])
#     predicted_labels = label_encoder.inverse_transform(predicted_labels)

#     # Add the predicted labels as a new column 'new_predict'
#     sub_to_predict['new_predict'] = pd.Series(predicted_labels, index=sub_to_predict.index)#predicted_labels
#     update_result_df = pd.concat([update_result_df , sub_to_predict], ignore_index=True)
#     # break #测试单个大类可以跑通，预测格式正确
#     # if i == specified_list_for_test[1]:
#     #     break #测试前2个大类可以跑通 

# print(result_list)
# print(result_list2)
# #最后保存/导出infer_df_pd4
# display(update_result_df)
# # display(infer_df_pd4)




# COMMAND ----------

update_result_df.count()

# COMMAND ----------

display(update_result_df)

# COMMAND ----------

update_result_df["upper_relation"].value_counts()

# COMMAND ----------

# display(sub_to_predict)

# COMMAND ----------

# # Iterate through each row in sub_df
# sub_to_predict = sub_to_predict.reset_index()
# for index, row in sub_df.iterrows():
#     # Find the corresponding row in infer_df_pd4 where head and tail values match
#     matching_row = infer_df_pd4[(infer_df_pd4['head'] == row['head']) & (infer_df_pd4['tail'] == row['tail'])].index[0]
    
#     # Update the new_predict value in infer_df_pd4 at the matching row
#     infer_df_pd4.loc[matching_row, 'new_predict'] = row['new_predict']

# # Iterate through each row in sub_df （gpt结果 可以参考看看）
# sub_to_predict = sub_to_predict.reset_index()
# for index, row in sub_df.iterrows():
#     # Find the corresponding row in infer_df_pd4 where head and tail values match
#     matching_rows = infer_df_pd4[(infer_df_pd4['head'] == row['head']) & (infer_df_pd4['tail'] == row['tail'])]
    
#     # Check if there is a matching row
#     if not matching_rows.empty:
#         matching_row = matching_rows.index[0]
        
#         # Update the new_predict value in infer_df_pd4 at the matching row
#         infer_df_pd4.loc[matching_row, 'new_predict'] = row['new_predict']

# COMMAND ----------

# from pyspark.sql import SparkSession

# # 创建 SparkSession
# spark = SparkSession.builder.getOrCreate()

# # 将 sub_to_predict 转换为 PySpark DataFrame
# sub_to_predict_spark = spark.createDataFrame(sub_to_predict.reset_index())

# # 将 infer_df_pd4 转换为 Pandas DataFrame
# infer_df_pd4 = infer_df_pd4.toPandas()

# # 在 infer_df_pd4 中查找匹配的行并更新 new_predict 值
# for index, row in sub_to_predict.iterrows():
#     matching_row = (infer_df_pd4['head'] == row['head']) & (infer_df_pd4['tail'] == row['tail'])
#     infer_df_pd4.loc[matching_row, 'new_predict'] = row['new_predict']

# # 将更新后的 infer_df_pd4 转换回 PySpark DataFrame
# infer_df_spark = spark.createDataFrame(infer_df_pd4)

# # 显示更新后的 infer_df_spark
# infer_df_spark.show()

# COMMAND ----------

infer_df_pd4['new_relation'] = infer_df_pd4.apply(lambda x: x['new_predict'] if x['new_column'] == 1 else x['relation'], axis=1)
display(infer_df_pd4)

# COMMAND ----------

# infer_df_pd4['new_predict'] = None
# infer_df_pd4.update(sub_to_predict[['head', 'tail', 'new_predict']].set_index(['head', 'tail']))


# COMMAND ----------

display(infer_df_pd4)

# COMMAND ----------

trainHistory.history #用这个，对比结果 #不用load_weights大概是77

# COMMAND ----------

# Assuming infer_df_pd4 is a pandas DataFrame
# infer_df_pd4 = infer_df_pd3.copy()
# infer_df_pd4 = spark.createDataFrame(infer_df_pd4)

# # file locations - long text
# file_location_text = "/FileStore/final_project_dataset/entity2textlong.tsv"
# file_type = "csv"

# # CSV options
# infer_schema = "false"
# first_row_is_header = "true"
# delimiter = "\t"

# # CSV options are applied. For other types, these options will be ignored.
# df_text = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)

# # show the created dataframes in record details
# # display(df_text)
# infer_df_pd4 = infer_df_pd4.join(df_text, infer_df_pd4["head"] == df_text["entity"], "left")
# infer_df_pd4 = infer_df_pd4.withColumnRenamed("text", "head_text2")
# infer_df_pd4 = infer_df_pd4.drop("entity")
# infer_df_pd4 = infer_df_pd4.join(df_text, infer_df_pd4["tail"] == df_text["entity"], "left")
# infer_df_pd4 = infer_df_pd4.withColumnRenamed("text", "tail_text2")
# infer_df_pd4 = infer_df_pd4.drop("entity")
# # from pyspark.sql.functions import *
# infer_df_pd4 = infer_df_pd4.withColumn('joined_text2', concat(col('head_text2'), lit(' '), col('tail_text2')))
# display(infer_df_pd4) 
# infer_df_pd4 = infer_df_pd4.toPandas()

# If infer_df_pd4 is already a Spark DataFrame, skip the above step
# infer_df_spark = infer_df_pd4

# COMMAND ----------


# # file locations - long text
# file_location_text = "/FileStore/final_project_dataset/entity2textlong.tsv"
# file_type = "csv"

# # CSV options
# infer_schema = "false"
# first_row_is_header = "true"
# delimiter = "\t"

# # CSV options are applied. For other types, these options will be ignored.
# df_text = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)

# # show the created dataframes in record details
# # display(df_text)
# infer_df_pd4 = infer_df_pd4.join(df_text, infer_df_pd4["head"] == df_text["entity"], "left")
# infer_df_pd4 = infer_df_pd4.withColumnRenamed("text", "head_text2")
# infer_df_pd4 = infer_df_pd4.drop("entity")
# infer_df_pd4 = infer_df_pd4.join(df_text, infer_df_pd4["tail"] == df_text["entity"], "left")
# infer_df_pd4 = infer_df_pd4.withColumnRenamed("text", "tail_text2")
# infer_df_pd4 = infer_df_pd4.drop("entity")
# # from pyspark.sql.functions import *
# infer_df_pd4 = infer_df_pd4.withColumn('joined_text2', concat(col('head_text2'), lit(' '), col('tail_text2')))
# display(infer_df_pd4) 
# infer_df_pd4 = infer_df_pd4.toPandas()


# COMMAND ----------

#结果更新，要用新的名称！！别把之前的覆盖了
#读取之前的结果，把我们预测是小众关系的 收到

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# 创建 SparkSession
spark = SparkSession.builder.getOrCreate()

file_location_text = "/FileStore/final_project_dataset/A_B_attempt1.csv"
file_type = "csv"

# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

# CSV options are applied. For other types, these options will be ignored.
infer_df323 = spark.read.format(file_type).option("inferSchema", infer_schema).option("header", first_row_is_header).option("sep", delimiter).load(file_location_text)

# show the created dataframes in record details
display(infer_df323)

# COMMAND ----------

display(update_result_df)
update_result_df.count()
update_result_df["upper_relation"].value_counts()

# COMMAND ----------

# 将 Pandas DataFrame 列重命名为 "update_head" 和 "update_tail"
update_result_df.rename(columns={"head": "update_head", "tail": "update_tail","relation": "update_relation"}, inplace=True)


# 将 Pandas DataFrame 转换为 PySpark DataFrame
update_result_spark = spark.createDataFrame(update_result_df)

# 执行左连接（left join）
joined_df = infer_df323.join(update_result_spark, (infer_df323['head'] == update_result_spark['update_head']) & (infer_df323['tail'] == update_result_spark['update_tail'])  , "left")

joined_df = joined_df.select("head", "relation", "tail", "new_predict","upper_relation")
# joined_df2 = joined_df.select("head", "tail", "new_predict")
# joined_df2 = joined_df2.withColumnRenamed("new_predict", "relation")

# 显示结果
display(joined_df)

# COMMAND ----------

joined_df.count()

# COMMAND ----------

# top 15 输出结果

from pyspark.sql import functions as F

# 使用 fill() 方法将 null 值替换为 relation 列的值
filled_df = joined_df.withColumn("new_predict", F.when(F.col("new_predict").isNull(), F.col("relation")).otherwise(F.col("new_predict")))

# 显示结果
display(filled_df)

# COMMAND ----------

from pyspark.sql.functions import col

# 筛选出 relation 和 new_predict 列不同的行
different_rows = joined_df.filter(col("relation") != col("new_predict"))

# 计数不同的行
count = different_rows.count()

# 显示结果
different_rows.show()
print("Count:", count)

# COMMAND ----------

# #draft供参考和复制粘贴
# #1
# from tensorflow.keras.preprocessing.text import Tokenizer

# max_words = 100000
# tokenizer = Tokenizer(num_words=100000, filters='!"#$%&()*+,-./:;<=>?@', lower=True)
# tokenizer.fit_on_texts(df_short['joined_text'].values)
# word_index = tokenizer.word_index
# print(f"length of dictionary {len(word_index)}")

# #2
# from tensorflow.keras.utils import pad_sequences
# import pandas as pd
# import numpy as np

# #3
# max_length = 128
# all_texts = tokenizer.texts_to_sequences(df_short['joined_text'].values)
# all_texts = pad_sequences(all_texts, maxlen=128)
# all_labels = pd.get_dummies(df_short['indexed_upper']).values

# #4
# embedding_dim = 100
# model = Sequential()
# model.add(Embedding(max_words, embedding_dim)) 
# model.add(LSTM(100)) 
# model.add(Dense(30, activation='softmax'))#这里的237我改30应该没错啊。这是最后一层，是标签种类个数
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'Precision', 'Recall'])

# #fit
# trainHistory = model.fit(train_texts, train_labels, epochs=5, batch_size=128, validation_split=0.2)

# #add to list score
# results = model.evaluate(test_texts, test_labels, batch_size=128)
# print(f"{i}: cross-entropy loss = {results[0]}, accuracy = {results[1]}, precision = {results[2]}, recall = {results[3]}")

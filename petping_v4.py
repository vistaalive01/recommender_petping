#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Reference: https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Collaborative%20Filtering%20Model%20with%20TensorFlow.ipynb
# https://towardsdatascience.com/make-your-own-recommendation-system-b596d847296d

#The collaborative filter approach focuses on finding users 
#who have given similar ratings to the same pets, 
#thus creating a link between users, to whom will be suggested pets 
#that were reviewed in a positive way. 
#In this way, we look for associations between users, not between pets.

import numpy as np
import pandas as pd
from scipy import sparse
import tensorflow as tf
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from tensorflow import keras
"""
from tensorflow.contrib.lite.python import convert_saved_model
from tensorflow.python.keras.saving.saved_model import export_saved_model
"""



# In[2]:


data_ratings_matrix = pd.read_csv('db_test_2803.csv')
data_ratings_matrix = data_ratings_matrix.drop(columns=['user'])
data_ratings_matrix = data_ratings_matrix.astype(int)
data_ratings_matrix.fillna(0)

data_user_matrix = pd.read_csv('db_test_2803.csv')
data_user_matrix = data_user_matrix.astype(int)
data_user_matrix.fillna(0)


rating = pd.read_csv('db_ratings.csv')
user = pd.read_csv('db_users.csv')
pet = pd.read_csv('db_pets.csv')
#, sep=';', error_bad_lines=False, encoding="latin-1"

pet_rating = pd.merge(rating, pet, on="pet_id")
cols = ['pet_type','pet_name','pet_sex','pet_breed','pet_color','pet_age','pet_marking','pet_weight','pet_health','pet_loc',
        'pet_char','pet_story','pet_status','pet_firebase_id']
pet_rating.drop(cols,axis=1,inplace=True)
#pet_rating.head(100)


# In[3]:


pet_rating.head(5)


# In[4]:


data_ratings_matrix.head(5)


# In[5]:


data_user_matrix.head(5)


# In[6]:


rating_v2 = rating.groupby('user_id').agg({'pet_id': ', '.join}).reset_index() 
rating_v2


# In[7]:


data_ratings_matrix.head(5)


# In[8]:



pet_ratings_matrix_v2 = pd.DataFrame(index=data_ratings_matrix.columns,columns=data_ratings_matrix.columns)
pet_ratings_matrix_v2.head(5)


# In[9]:


for i in range(0,len(pet_ratings_matrix_v2.columns)) :
    # Loop through the columns for each column
    for j in range(0,len(pet_ratings_matrix_v2.columns)) :
      # Fill in placeholder with cosine similarities
      pet_ratings_matrix_v2.iloc[i,j] = 1-cosine(data_ratings_matrix.iloc[:,i],data_ratings_matrix.iloc[:,j])
      
pet_ratings_matrix_v2


# In[10]:


cosine_value = cosine_similarity(data_user_matrix)
np.fill_diagonal(cosine_value, 0)
user_similarity_matrix =pd.DataFrame(cosine_value,index=data_user_matrix.index+1)
user_similarity_matrix.columns=data_user_matrix.index+1
user_similarity_matrix.head(5)


# In[11]:


# user similarity on replacing NAN by item(movie) avg
cosine = cosine_similarity(pet_ratings_matrix_v2)
np.fill_diagonal(cosine, 0 )
pet_similarity_matrix = pd.DataFrame(cosine,index=pet_ratings_matrix_v2.index)
pet_similarity_matrix.columns=pet_ratings_matrix_v2.index
pet_similarity_matrix.head(5)


# In[12]:


def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index, 
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df


# In[13]:


user_rank_neighbours_matrix = find_n_neighbours(user_similarity_matrix,30)
user_rank_neighbours_matrix.head()


# In[14]:


pet_rank_neighbours_matrix = find_n_neighbours(pet_similarity_matrix,30)
pet_rank_neighbours_matrix.head(5)


# In[15]:


user = 100
user_index = data_user_matrix[data_user_matrix.user == user].index.tolist()[0]
    
user_liked_list = data_user_matrix.loc[user_index]
user_liked_list = user_liked_list[user_liked_list>0].index.values
       
#calculate the score
ranking_score = pet_similarity_matrix.dot(data_ratings_matrix.loc[user_index]).div(pet_similarity_matrix.sum(axis=1))

print (user_liked_list)
print (ranking_score.nlargest(10))


# In[17]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


# In[24]:


num_input = pet_rating['pet_id'].nunique()
num_hidden_1 = 10
num_hidden_2 = 5

X = tf.placeholder(tf.float64, [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1], dtype=tf.float64)),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2], dtype=tf.float64)),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1], dtype=tf.float64)),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input], dtype=tf.float64)),
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2], dtype=tf.float64)),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1], dtype=tf.float64)),
    'decoder_b2': tf.Variable(tf.random_normal([num_input], dtype=tf.float64)),
}


# In[25]:


def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    return layer_2

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    return layer_2


# In[26]:


# We will construct the model and the predictions

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op

y_true = X


# In[27]:


loss = tf.losses.mean_squared_error(y_true, y_pred)
optimizer = tf.train.RMSPropOptimizer(0.03).minimize(loss)
eval_x = tf.placeholder(tf.int32, )
eval_y = tf.placeholder(tf.int32, )
pre, pre_op = tf.metrics.precision(labels=eval_x, predictions=eval_y)


# In[28]:


init = tf.global_variables_initializer()
local_init = tf.local_variables_initializer()
pred_data = pd.DataFrame()


# In[ ]:





# In[30]:


# user_count is the amount of pet that they liked

user_count = (rating.groupby(by=['user_id'])['rating'].
                 count().
                 reset_index().
                 rename(columns = {'rating': 'u_rating_count'})
                 [['user_id','u_rating_count']])
user_count.head(100)


# In[31]:


combined = rating.merge(user_count, left_on='user_id'
                            ,right_on="user_id", how ='outer')
#cols = ['pet_type', 'pet_name','rating']
#combined.drop(cols,axis=1,inplace=True)
combined


# In[32]:


# Normalize the ratings.
# MinMaxScaler() 
# fit_transform คือ 

scaler = MinMaxScaler()

combined['rating'] = combined['rating'].values.astype(float)
rating_scaled = pd.DataFrame(scaler.fit_transform(combined['rating']
                                                  .values.reshape(-1,1)))
combined['rating'] = rating_scaled
combined


# In[33]:


combined = combined.drop_duplicates(['user_id', 'pet_id'])
user_pet_matrix = combined.pivot(index='user_id', columns='pet_id', values='rating')
user_pet_matrix.fillna(0, inplace=True)

users = user_pet_matrix.index.tolist()
pets = user_pet_matrix.columns.tolist()

user_pet_matrix = user_pet_matrix.values
user_pet_matrix


# In[34]:


with tf.Session() as session:
    epochs = 20
    batch_size = 32

    session.run(init)
    session.run(local_init)

    num_batches = int(user_pet_matrix.shape[0] / batch_size)
    user_pet_matrix = np.array_split(user_pet_matrix, num_batches)
    
    for i in range(epochs):

        avg_cost = 0
        for batch in user_pet_matrix:
            _, l = session.run([optimizer, loss], feed_dict={X: batch})
            avg_cost += l

        avg_cost /= num_batches
        
        plt.plot(i+1,avg_cost,'g.',label='Training loss')

        print("epoch: {} Loss: {}".format(i + 1, avg_cost))
        
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    user_pet_matrix = np.concatenate(user_pet_matrix, axis=0)

    preds = session.run(decoder_op, feed_dict={X: user_pet_matrix})

    pred_data = pred_data.append(pd.DataFrame(preds))

    pred_data = pred_data.stack().reset_index(name='pet_id')
    pred_data.columns = ['user_id', 'pet_id', 'rating']
    pred_data['user_id'] = pred_data['user_id'].map(lambda value: users[value])
    pred_data['pet_id'] = pred_data['pet_id'].map(lambda value: pets[value])
    
    keys = ['user_id', 'pet_id']
    index_1 = pred_data.set_index(keys).index
    index_2 = combined.set_index(keys).index

    top_ten_ranked = pred_data[~index_1.isin(index_2)]
    top_ten_ranked = top_ten_ranked.sort_values(['user_id', 'pet_id'], ascending=[True, False])
    top_ten_ranked = top_ten_ranked.groupby('user_id').head(10)
    


# In[ ]:

    """
    root = tf.train.Checkpoint()

    export_dir = "/tmp/test_saved_model"
    input_data = tf.constant(1., shape=[1, 1])
    to_save = root.f.get_concrete_function(input_data)
    tf.saved_model.save(root, export_dir, to_save)

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()

    #convert_saved_model.convert(saved_model_dir='/home/.../export',output_arrays="final_result",output_tflite='/home/.../export/graph.tflite')

    """

    keras_file = "petping.h5"
    keras.models.save_model(optimizer,keras_file)

    converter = lite.TocoConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    open("petping.tflite","wb").write(tflite_model)



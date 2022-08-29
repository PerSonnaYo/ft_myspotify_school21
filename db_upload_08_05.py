# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 12:50:11 2019
fedf
@author: user
"""
import pandas as pd
from sqlalchemy import create_engine
import sqlalchemy
import numpy as np
import os
import joblib
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

class Song():
	def __init__(self):
		self.dfs, self.models = self.parse_files()
		self.df_songs = self.dfs['base_songs']
		self.movie_titles = dict(zip(self.df_songs['id'], self.df_songs['song_name']))
		self.movie_artists = dict(zip(self.df_songs['id'], self.df_songs['artist']))
		self.id_songs_df = self.create_lst()
		self.lst_songs = self.id_songs_df['names'].to_list()
		self.genres = ['Rock', 'Rap', 'Jazz', 'Electronic', 'Pop', 'Blues', 'Country', 'Reggae', 'New Age']
		self.colls = ['love', 'war', 'happiness', 'loneliness', 'money']
		self.X_movie, self.movie_mapper, self.movie_inv_mapper = self.create_matrix_2(self.df_songs)
		self.X_user, self.user_mapper, self.user_inv_mapper = self.create_matrix_1(self.df_songs)

	def parse_files(self):
		dirname = 'data'
		files = os.listdir(dirname)
		#temp хранит названия файлов
		temp = map(lambda name: os.path.join(dirname, name), files)
		h = {}
		best_model = {}
		for i, file in enumerate(temp):
			if '.csv' in file:
				f = file.split('\\')[1]
				f = f.split('.')[0]
				h[f] = pd.read_csv(file, sep=';', header=0, encoding='utf-8', low_memory=False)
			elif '.joblib' in file:
				f = file.split('\\')[1]
				f = f.split('.')[0]
				best_model[f] = joblib.load(file)
			elif '.xlsx' in file:
				xl = pd.ExcelFile(file)
				lsts = xl.sheet_names
				for lst in lsts:
					f = file.split('\\')[1]
					f = f.split('.')[0] + '$' + lst
					h[f] = xl.parse(lst, header=0)
		return h, best_model

	def create_lst(self):
		dfr = pd.DataFrame.from_dict(data=self.movie_titles, orient='index', columns=['names'])
		return dfr.reset_index()

	def create_matrix_1(self, df):
      
		N = len(df['user_id'].unique())
		M = len(df['id'].unique())
		
		# Map Ids to indices
		user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
		# for i, row in df.iterrows():
		#     movie_mapper[row['id']] = 
		movie_mapper = dict(zip(np.unique(df["id"]), list(range(M))))
		# print(movie_mapper)
		
		# Map indices to IDs
		user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
		movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["id"])))
		
		user_index = [user_mapper[i] for i in df['user_id']]
		movie_index = [movie_mapper[i] for i in df['id']]
	
		X = csr_matrix((df["count"], (user_index, movie_index)), shape=(N, M))
		
		return X, user_mapper, user_inv_mapper

	def create_matrix_2(self, df):
		
		N = len(df['user_id'].unique())
		M = len(df['id'].unique())
		
		# Map Ids to indices
		user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
		movie_mapper = dict(zip(np.unique(df["id"]), list(range(M))))
		
		# Map indices to IDs
		user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
		movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["id"])))
		
		user_index = [user_mapper[i] for i in df['user_id']]
		movie_index = [movie_mapper[i] for i in df['id']]
	
		X = csr_matrix((df["count"], (movie_index, user_index)), shape=(M, N))
		
		return X,  movie_mapper, movie_inv_mapper

	def find_similar_movies(self, movie_id, X, k, movie_mapper, movie_inv_mapper, show_distance=False):

		neighbour_ids = []
		
		movie_ind = movie_mapper[movie_id]
		movie_vec = X[movie_ind]
		k+=1
		movie_vec = movie_vec.reshape(1,-1)
		# print(movie_vec)
		neighbour = self.models['movie_model'].kneighbors(movie_vec, return_distance=show_distance)
		for i in range(0,k):
			n = neighbour.item(i)
			neighbour_ids.append(movie_inv_mapper[n])
		neighbour_ids.pop(0)
		return neighbour_ids

	def find_similar_users(self, user_id, X, k, user_mapper, user_inv_mapper, show_distance=False):
		neighbour_ids = []
        
		user_ind = user_mapper[user_id]
		user_vec = X[user_ind]
		k+=1
		user_vec = user_vec.reshape(1,-1)
		neighbour = self.models['user_model'].kneighbors(user_vec, return_distance=show_distance)
		for i in range(0,k):
			n = neighbour.item(i)
			neighbour_ids.append(user_inv_mapper[n])
		neighbour_ids.pop(0)
		first = neighbour_ids[:5]
		second = neighbour_ids[5:]
		df4 = self.df_songs[self.df_songs['user_id'].isin(first)]
		df4 = df4[['id', 'count']].groupby('id').agg({'count': ['sum']})
		df4.sort_values(by=[('count', 'sum')], inplace=True, ascending=False)
		df_res = df4[:7]
		jj = df_res.index.to_list()
		df4 = self.df_songs[self.df_songs['user_id'].isin(second)]
		df4 = df4[['id', 'count']].groupby('id').agg({'count': ['sum']})
		df4 = df4[~df4.index.isin(jj)]
		df4.sort_values(by=[('count', 'sum')], inplace=True, ascending=False)
		df_res = pd.concat([df_res, df4[:(10 - len(df_res))]])
		return df_res.index.to_list()

	def start(self):

# print(h.keys())
# print(h['song_names'])

		print("Введите команду:")
		while True:
			strin = input()
			if strin == 'top250':
				print(self.dfs['250$250'].rename(columns={'Unnamed: 0': 'index'}))
			elif strin.capitalize() in self.genres:
				# print(f'genres${strin.capitalize()}')
				df = self.dfs[f'genres${strin.capitalize()}'].drop('genre1', axis=1).rename(columns={'Unnamed: 0': 'index'})
				print(df)
			elif strin.lower() in self.colls:
				df = self.dfs[f'collections${strin.lower()}']
				df1 = df.rename(columns={'Unnamed: 0': 'index', "('artist', 'max')": 'artist', "('song_name', 'max')": 'song_name'})
				df1['count'].fillna(0, inplace=True)
				df1['count'] = df1['count'].astype('int16')
				df1.sort_values(by=['count', 'song_name'], ascending=False, inplace=True)
				print(df1)
			elif strin in self.lst_songs:
				id = self.id_songs_df[self.id_songs_df['names'] == strin].iloc[0][0]
				print(id)
				movie_id = id
  
				similar_ids = self.find_similar_movies(movie_id, self.X_movie, 10, self.movie_mapper, self.movie_inv_mapper)
				movie_title = self.movie_titles[movie_id]
				df5 = self.df_songs[self.df_songs['id'] == movie_id]
				users = df5['user_id'].to_list()
				df6 = self.df_songs[self.df_songs['user_id'].isin(users)]
				knows_ids = df6['id'].to_list()
				print(f'p@k: {apk(knows_ids, similar_ids)}')
				# print('KNOWS: ')
				# for i in knows_ids:
				#     print(movie_titles[i])
				print(f"---NAME SONG {movie_title}")
				dat = []
				for i in similar_ids:
					dat.append({'artist': self.movie_artists[i], 'song_name': self.movie_titles[i]})
				print(pd.DataFrame(dat).reset_index())
			elif strin in self.user_mapper.keys():
				user_id = strin
  
				similar_ids = self.find_similar_users(user_id, self.X_user, 20, self.user_mapper, self.user_inv_mapper)
				df5 = self.df_songs[self.df_songs['user_id'] == user_id]
				knows_ids = df5['id'].to_list()
				# print(similar_ids)
				print(f'p@k: {apk(knows_ids, similar_ids)}')
				# print('KNOWS: ')
				# for i in knows_ids:
				# 	print(movie_titles[i])
				# print(f"---Since you watched {user_id}")
				dat = []
				for i in similar_ids:
					dat.append({'artist': self.movie_artists[i], 'song_name': self.movie_titles[i]})
				print(pd.DataFrame(dat).reset_index())
				# for i in similar_ids:
				# 	print(movie_titles[i], movie_artists[i])
			else:
				if strin == '0':
					exit(0)
				print('Введите что-то из этого: top250, rock, rap, love, happiness, user id(bb233929a2ccd4a9c20043d1665ec357826ea5e7), song name (Ethos of Coercion, Inequality Street)')

p = Song()
p.start()
exit(0)






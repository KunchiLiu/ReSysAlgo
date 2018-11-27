# -*- coding: utf-8 -*-
# Origin resource from MovieLens: http://grouplens.org/datasets/movielens/1m
import pandas as pd

class Channel:
    # simple processing for .dat to .csv

    def __init__(self):
        self.orign_path = 'data/ml-1m/{}'

    def process(self):
        print('Process users data...')
        self._process_users_data()
        print('Process movies data...')
        self._process_movies_data()
        print('Process ratings data...')
        self._process_ratings_data()
        print('Process End.')

    def _process_users_data(self, file = 'users.dat'):
        f = pd.read_table(self.orign_path.format(file), sep='::', engine='python',
                          names=['userID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
        f.to_csv(self.orign_path.format('users.csv'), index=False)

    def _process_ratings_data(self, file = 'ratings.dat'):
        f = pd.read_table(self.orign_path.format(file), sep='::', engine='python',
                          names=['userID', 'MovieID', 'Rating', 'Timestamp'])
        f.to_csv(self.orign_path.format('ratings.csv'), index=False)

    def _process_movies_data(self, file = 'movies.dat'):
        f = pd.read_table(self.orign_path.format(file), sep='::', engine='python',
                          names=['MovieID', 'Title', 'Genres'])
        f.to_csv(self.orign_path.format('movies.csv'), index=False)

if __name__ == '__main__':
    Channel().process()
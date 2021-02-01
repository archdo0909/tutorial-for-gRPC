import pandas as pd
from surprise.dataset import DatasetAutoFolds
from surprise.dataset import Reader
from surprise import SVD
import sys


reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))

# DatasetAutoFolds 클래스를 사용해서 개별적으로 생성
# index와 header가 없는 상태로 재생성했던 ratings_surprise.csv파일에 기반
data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_surprise.csv', reader=reader)

# 영화에 대한 정보 데이터 로딩
movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# 위에서 개별적으로 생성한 csv파일을 학습데이터로 생성
trainset = data_folds.build_full_trainset()

def training(n_factors=50, n_epochs=20, random_state=42):
    algo = SVD(n_factors, n_epochs, random_state)
    algo.fit(trainset)
    return algo

    

def get_unseen_surprise(ratings, movies, userId):
    # 특정 유저가 본 movie id들을 리스트로 할당
    seen_movies = ratings[ratings['userId']==userId]['movieId'].tolist()
    # 모든 영화들의 movie id들 리스트로 할당
    total_movies = movies['movieId'].tolist()
    
    # 모든 영화들의 movie id들 중 특정 유저가 본 movie id를 제외한 나머지 추출
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]
    print(f'특정 {userId}번 유저가 본 영화 수: {len(seen_movies)}\n추천한 영화 개수: {len(unseen_movies)}\n전체 영화수: {len(total_movies)}')
    
    return unseen_movies

def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    # 알고리즘 객체의 predict()를 이용해 특정 userId의 평점이 없는 영화들에 대해 평점 예측
    predictions = [algo.predict(str(userId), str(movieId)) for movieId in unseen_movies]
    
    # predictions는 Prediction()으로 하나의 객체로 되어있기 때문에 예측평점(est값)을 기준으로 정렬해야함
    # est값을 반환하는 함수부터 정의. 이것을 이용해 리스트를 정렬하는 sort()인자의 key값에 넣어주자!
    def sortkey_est(pred):
        return pred.est
    
    # sortkey_est함수로 리스트를 정렬하는 sort함수의 key인자에 넣어주자
    # 리스트 sort는 디폴트값이 inplace=True인 것처럼 정렬되어 나온다. reverse=True가 내림차순
    predictions.sort(key=sortkey_est, reverse=True)
    # 상위 n개의 예측값들만 할당
    top_predictions = predictions[:top_n]
    
    # top_predictions에서 movie id, rating, movie title 각 뽑아내기
    top_movie_ids = [int(pred.iid) for pred in top_predictions]
    top_movie_ratings = [pred.est for pred in top_predictions]
    top_movie_titles = movies[movies.movieId.isin(top_movie_ids)]['title']
    # 위 3가지를 튜플로 담기
    # zip함수를 사용해서 각 자료구조(여기선 리스트)의 똑같은 위치에있는 값들을 mapping
    # zip함수는 참고로 여러개의 문자열의 똑같은 위치들끼리 mapping도 가능!
    top_movie_preds = [(ids, rating, title) for ids, rating, title in zip(top_movie_ids, top_movie_ratings, top_movie_titles)]
    
    return top_movie_preds

def recommend(userId):
    algo = training()
    unseen_lst = get_unseen_surprise(ratings, movies, userId)
    top_movies_preds = recomm_movie_by_surprise(algo, userId, unseen_lst, top_n=5)
    recommend_list = [i[2] for i in top_movies_preds]
    print(recommend_list)
    return recommend_list

### 위에서 정의한 함수를 사용해 특정 유저의 추천 영화들 출력해보기
# unseen_lst = get_unseen_surprise(ratings, movies, 9)
# top_movies_preds = recomm_movie_by_surprise(algo, 9, unseen_lst, top_n=10)
# print()
# print('#'*8,'Top-10 추천영화 리스트','#'*8)

# # top_movies_preds가 여러가지의 튜플을 담고 있는 리스트이기 때문에 반복문 수행
# for top_movie in top_movies_preds:
#     print('* 추천 영화 이름: ', top_movie[2])
#     print('* 해당 영화의 예측평점: ', top_movie[1])
#     print()
    
    
    


if __name__ == "__main__":
    recommend(int(sys.argv[1]))
from __future__ import print_function

import grpc

import recommender_pb2
import recommender_pb2_grpc


def run(user_id):
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = recommender_pb2_grpc.RecommendServiceStub(channel)
        response = stub.RecommendContentEvent(recommender_pb2.RecommendRequest(user_id))
        print("Recommended Contents : ", response.message)
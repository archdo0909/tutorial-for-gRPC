from concurrent import futures
import time

import grpc

import recommender_pb2
import recommender_pb2_grpc

from train import training


class RecommendModel(recommender_pb2_grpc.RecommendServiceServicer):
    def RecommendContentEvent(self, request, context):
        return recommender_pb2.RecommendResponse()


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    recommender_pb2_grpc.add_RecoomendServiceServiver_to_server(RecommendModel(), server)
    server.add_insecure_port('localhost:8080')
    server.start()
    try:
        while True:
            time.sleep(60)
    
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == "__main__":
    serve() 
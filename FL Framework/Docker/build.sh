# Be in the main folder FedOptOptimization

docker build -t fed_opt -f Docker/Dockerfile . --platform linux/amd64

# to run
# docker run -it fed_opt python3 run.py [--mode client]

# to change file
# sudo docker run -it fed_opt bash

# to push
docker tag fed_opt anboiano/fedopt
docker push anboiano/fedopt

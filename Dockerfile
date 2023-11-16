# Use the official Python image from Docker Hub as the base image
FROM python:3.8-slim
# Set working directory
WORKDIR /usr/src/app
# Copy
COPY sparse_recommender.py .

# run
CMD [ "python", "./sparse_recommender.py" ]
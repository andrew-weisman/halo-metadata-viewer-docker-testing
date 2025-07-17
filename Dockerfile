# Get the official Python image from Docker Hub.
FROM python:3.12-slim

# Install git.
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Clone the repo.
RUN git clone https://github.com/andrew-weisman/halo-metadata-viewer-docker-testing.git /app

# Set the working directory.
WORKDIR /app

# Install dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that Streamlit will run on.
EXPOSE 8501

# Command to run the Streamlit app.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]


# To build the Docker image, run:
# docker build -t halo-metadata-viewer .
# OR (if e.g. dependencies in requirements.txt have changed):
# docker build --no-cache -t halo-metadata-viewer .

# To run the Docker container, run:
# docker run -p 8501:8501 halo-metadata-viewer

# To update the tagged image (to ensure the push works) and push to Docker Hub (https://hub.docker.com/r/andrewweisman/halo-metadata-viewer) do, e.g.:
# docker tag 8d33b67455c7 andrewweisman/halo-metadata-viewer:latest
# docker push andrewweisman/halo-metadata-viewer

# Then, open a web browser and go to http://localhost:8501 to see the app.


# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
# Since we don't have a requirements.txt, we'll install directly.
# In a real project, you would create a requirements.txt from your Colab notebook
# by running `pip freeze > requirements.txt` and then `RUN pip install -r requirements.txt`.
RUN pip install Flask gunicorn pandas scikit-learn xgboost joblib

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]

# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_APP=app.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]



# 2N ecample:

FROM python: 3.10
WORKDIR /code
ENV FLASK_APP = app.py
ENV FLASK_RUN_HOST = 0.0.0.0
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["flask", "app.py"]"


Explanation:

This is how it works:
Flask runs on 0.0.0.0:5000 inside container
Accepts external traffic.
Docker maps container port 5000 to your machine’s port 5000
This is done using -p 5000:5000
You visit http://localhost:5000 on your browser
Docker forwards the request into the container → Flask handles it → sends back response.

🔁 Without host="0.0.0.0"?
If you don’t use host="0.0.0.0":

Flask binds to 127.0.0.1 (localhost inside the container).

Outside world (including your browser) can’t reach it — even if Docker maps the ports.

So:

You must use both --host=0.0.0.0 and -p 5000:5000 to make your app visible on localhost:5000.


Why use 0.0.0.0?
Normally, if you run Flask like this:

bash
Copy
Edit
flask run
It binds to 127.0.0.1 — which means:

It's only accessible inside the container.

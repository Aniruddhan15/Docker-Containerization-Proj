# Base Image
FROM python:3.10

# Working Directory
WORKDIR /app

# Copy all project files to /app
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose Streamlit or any service port
EXPOSE 8501

# Command to run the app
CMD ["python", "./app.py"]

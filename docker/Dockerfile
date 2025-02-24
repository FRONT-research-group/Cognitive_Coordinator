# Use an official Python runtime as a parent image
FROM python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose the port FastAPI runs on
EXPOSE 6464

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6464"]

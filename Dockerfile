# Use an official Python runtime as a parent image
FROM python:3.11.7

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Install Jupyter Notebook
RUN pip3 install jupyter

# Install Graphviz
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
 && rm -rf /var/lib/apt/lists/*

# Expose the port for Jupyter Notebook
EXPOSE 8888

# Command to run Jupyter Notebook with "run all" cells
CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0", "--allow-root", "anomaly-detection-using-lightgbm.ipynb","vanilla-LSTM.ipynb"]
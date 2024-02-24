FROM nvcr.io/nvidia/pytorch:23.11-py3

RUN python3 -m pip install --upgrade pip
RUN pip install git+https://github.com/Shixiaowei02/mpi4py.git@fix-setuptools-version

WORKDIR /root/code
COPY environment.txt .

RUN pip install --no-cache-dir -r environment.txt

RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
RUN pip install bitsandbytes


# Set environment variables
ENV MPI_HOME=/usr/local/mpi 

# Start from the official MRtrix3 Docker image
FROM mrtrix3/mrtrix3:latest

# Install required tools
RUN apt-get update && apt-get install -y \
    wget unzip parallel \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

######################################
# Install prebuilt ANTs (Linux x86_64)
######################################

# Set ANTs environment
ENV ANTSPATH=/opt/ants/bin
ENV PATH="${ANTSPATH}:${PATH}"

# Download Neurodocker prebuilt ANTs binaries
# Change version if you want: 2.5.0 is widely used and stable
RUN mkdir -p /opt/ants && \
    wget -qO /tmp/ants.zip \
      https://github.com/ANTsX/ANTs/releases/download/v2.6.3/ants-2.6.3-ubuntu-22.04-X64-gcc.zip && \
    unzip /tmp/ants.zip -d /opt/ants_full && \
    rm /tmp/ants.zip

# Ensure binaries are executable
RUN chmod -R +x /opt/ants_full/ants-2.6.3/bin

RUN pip install numpy nibabel

ENV ANTSPATH=/opt/ants_full/ants-2.6.3/bin
ENV PATH="${ANTSPATH}:${PATH}"


# Default workdir
WORKDIR /data


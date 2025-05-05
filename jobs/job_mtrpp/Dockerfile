# ARG GITHUB_TOKEN

FROM klaymusic/klay-beam:0.13.7-py3.10-beam2.53.0-torch2.1-cuda12.1

WORKDIR /klay/app

# These paths are relatived to the repository root
COPY jobs/job_mtrpp ./job_mtrpp
COPY klay_beam/submodules/klay-data ./submodules/klay-data
COPY klay_beam/submodules/klay-mtrpp ./submodules/klay-mtrpp

# 2) install klay_data FROM THAT WHEEL ONLY  ─────────────────────────────
RUN pip install --no-cache-dir --no-index ./submodules/klay_data
RUN pip install --no-cache-dir --no-index ./submodules/klay_mtrpp

# Install git
RUN apt-get update && apt-get install -y git

# Set git to use token for all GitHub URLs
# RUN git config --global url."https://${GITHUB_TOKEN}@github.com/".insteadOf "https://github.com/"

# Download model
RUN python3 -c "from klay_data.extractors import MTRPPExtractor; MTRPPExtractor()"

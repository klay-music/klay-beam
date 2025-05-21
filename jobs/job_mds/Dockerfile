# ────────────────────────────────────────────────────────────────────────
# Stage 2 – final image: install the wheel + run the app
# ────────────────────────────────────────────────────────────────────────
FROM klaymusic/klay-beam:0.14.6-py3.10-beam2.64.0-torch2.5

WORKDIR /klay/app

COPY klay_beam/submodules/klay-data ./submodules/klay-data

RUN python3 -m pip install --no-cache-dir build pipdeptree && \
    python3 -m build --wheel ./submodules/klay-data -o ./klay-data-wheel

RUN python3 -m pip install --no-cache-dir --find-links=./klay-data-wheel 'klay_data[torch]'

RUN python3 -m pipdeptree

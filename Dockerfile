from continuumio/miniconda3:latest
add environment.yml /environment.yml
run ls /
run apt-get update \
    && apt-get install -y gcc g++ git unzip libgtk2.0-0 libsm6 libxft2 curl \
    && conda env create -n fordead -f /environment.yml \
	&& conda config --remove channels conda-forge \
	&& conda config --add channels conda-forge \
    && conda env list
shell ["/bin/bash", "--login", "-c"]
run conda init bash
# run echo "conda activate fordead" >> ~/.bashrc # activates fordead by default, avoids starting line by `run conda activate fordead && ...`
run conda env list
run conda activate fordead && conda env list && pip install portray python-markdown-math mdx-breakless-lists mkdocs-click

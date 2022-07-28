FROM iglucontest/gridworld_env:latest

RUN pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html 
RUN pip3 install git+https://github.com/iglu-contest/sample-factory.git@1.121.3.iglu
RUN pip3 install git+https://github.com/iglu-contest/gridworld.git@master

ENV CUDA_VISIBLE_DEVICES="0,1"
RUN pip install "pydantic>=1.8.2" "wandb>=0.12.0" "matplotlib"

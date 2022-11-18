# apt-get update && apt-get install -y --no-install-recommends \
#         git tmux curl unrar p7zip-full ffmpeg libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx zip\
#       && \
#     rm -rf /var/lib/apt/lists


pip install pandas scikit-learn tqdm numpy nvitop seaborn transformers datasets evaluate sentencepiece seqeval spacy sacrebleu
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
# spacy download en_core_web_lg
# pip install -r requirements.txt

spacy download en_core_web_sm
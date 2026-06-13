import zipfile
from gdown import download

gdown_fname = download('https://drive.google.com/uc?id=1Lx3YnJ15B5KCRA026KJSbpSQpBF-wh7j', quiet = False)

with zipfile.ZipFile(gdown_fname) as f:
    f.extractall(path='.')


import zipfile
from gdown import download

gdown_fname = download('https://drive.google.com/uc?id=11qDIdeFM9h1NMfIchwOnmd7saRtmk1QF', quiet = False)

with zipfile.ZipFile(gdown_fname) as f:
    f.extractall(path='.')


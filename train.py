import os
import shutil
from model import Model

if os.path.exists("./files"):
    shutil.rmtree("./files")
    os.mkdir("./files")

model = Model()
model.train()

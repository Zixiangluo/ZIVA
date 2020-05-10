import run
import numpy as np
import pandas as pd

count = np.array(pd.read_csv("data/data5.csv", index_col=0, sep="\t"))
#idx = np.argwhere(np.all(count[..., :] == 0, axis=0))
#count = np.delete(count, idx, axis=1)
count = count[:300, :5000]

# Run
model = run.train(np.float32(count), f = "nb", log=True, scale=True, epochs=300, batch_size=50, info_step=1)

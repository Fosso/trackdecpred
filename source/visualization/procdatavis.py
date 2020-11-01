import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../../data/procdata.csv')
data.hist(figsize=(20, 20))
plt.show()

print(data.info())
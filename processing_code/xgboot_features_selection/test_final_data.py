import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pylab as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell



InteractiveShell.ast_node_interactivity = "all"


data = pd.read_csv(r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023\ready_files\filtered_modified_combined1.csv")

print(data.head())
x=data.loc[:, data.columns != 'label']
y=data['label']

 
print(x.head())
print(y.head())

_ = sns.countplot(y)
plt.show()
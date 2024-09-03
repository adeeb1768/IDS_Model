import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"


df = pd.read_csv(r'C:\Users\adeeb\Desktop\master thesis\test file\part-00168-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv')


# Filter data by label (optional)
filtered_labels = ['DDoS-RSTFINFlood', 'DDoS-TCP_Flood', 'DDoS-ICMP_Flood', 'DoS-UDP_Flood', 'DoS-SYN_Flood',
                   'Mirai-greeth_flood', 'DDoS-SynonymousIP_Flood', 'Mirai-udpplain', 'DDoS-SYN_Flood', 'DDoS-PSHACK_Flood',
                   'DDoS-UDP_Flood', 'BenignTraffic']
df = df[df['label'].isin(filtered_labels)]  # Filter based on specified labels (optional)


# Sample data (Optional)
df = df.groupby('label', group_keys=False).apply(lambda x:x.sample(4000))

filtered_data = df[['IAT', 'Rate', 'Header_Length', 'Number', 'flow_duration', 'syn_count', 'Variance','Protocol Type',
                    'Weight', 'Tot sum', 'rst_count', 'HTTPS', 'Min', 'Covariance', 'Max', 'Tot size','label']]


filtered_data.to_csv('validation_data.csv', sep=',', index=False, encoding='utf-8')

# Split data into features and target variable (ensuring label consistency)
target = filtered_data.loc[:, filtered_data.columns != 'label']
features = filtered_data['label']

print(features.head())
print(target.head())

_ = sns.countplot(target)
plt.show()

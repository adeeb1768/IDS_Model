import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical




def get_data (client_id):
        data_dir = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023\ready_files"
        file_map = {
                1: "filtered_modified_combined1.csv",
                2: "filtered_modified_combined2.csv",
                3: "filtered_modified_combined3.csv",
                4: "filtered_modified_combined4.csv",
                5: "filtered_modified_combined5.csv",
                6: "filtered_modified_combined6.csv",
                7: "filtered_modified_combined7.csv",
                8: "filtered_modified_combined8.csv",
                9: "filtered_modified_combined9.csv",
                10: "filtered_modified_combined10.csv",
                11: "filtered_modified_combined11.csv",
                12: "filtered_modified_combined12.csv",
                13: "filtered_modified_combined13.csv",
                14: "filtered_modified_combined14.csv",
                15: "filtered_modified_combined15.csv",
                16: "filtered_modified_combined16.csv",
                17: "filtered_modified_combined17.csv",
                18: "filtered_modified_combined18.csv",
                19: "filtered_modified_combined19.csv",
                20: "filtered_modified_combined20.csv",
                21: "filtered_modified_combined21.csv",
                
        }
        try:
               file_path = os.path.join(data_dir, file_map[client_id])
               df = pd.read_csv(file_path)
        except FileNotFoundError:
               raise FileNotFoundError(f"CSV file not found for client ID {client_id}: {file_path}")
        

        rowsNum, columnsNum = df.shape

        df = df.groupby('label', group_keys=False).apply(lambda x:x.sample(10000))

        # shuffle the DataFrame rows
        df = df.sample(frac = 1).reset_index(drop=True)
        print('df shape = {rows} x {columns}'.format(rows=rowsNum, columns=columnsNum))
        print(df.info())
        print(df["label"].value_counts())
        print(df.isna().sum())
        #get every common value from all columns
        #modes = df.mode().iloc[0]
        # fill missing values with common value of each column
        #df.fillna(modes, inplace=True)

        #choose input and output
        x=df.loc[:, df.columns != 'label']
        y=df['label']
        
        #normalize input columns
        scaler = StandardScaler()
        x=scaler.fit_transform(x)
        label_encoder = LabelEncoder()
        y=label_encoder.fit_transform(y)
        y= to_categorical(y)
        
       # split data into train test group
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=40)
        print(X_train.shape); print(X_val.shape)
        return X_train, X_val, y_train, y_val
        

def get_validation_data():
       validation_data_dir = r"C:\Users\adeeb\Desktop\master thesis\CICIoT2023\ready_files\validation_data.csv"

       df = pd.read_csv(validation_data_dir)
       # shuffle the DataFrame rows
       df = df.sample(frac = 1).reset_index(drop=True)
       rowsNum, columnsNum = df.shape
       print('df shape = {rows} x {columns}'.format(rows=rowsNum, columns=columnsNum))
       print(df.info())
       print(df["label"].value_counts())
       print(df.isna().sum())
       x=df.loc[:, df.columns != 'label']
       y=df['label']

       scaler = StandardScaler()
       x=scaler.fit_transform(x)
       label_encoder = LabelEncoder()
       y=label_encoder.fit_transform(y)
       y= to_categorical(y)
       
       print(x.shape)
       print(y.shape)

       return x,y


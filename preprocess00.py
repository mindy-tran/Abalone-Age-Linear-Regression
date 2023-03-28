# Write a simple python script to convert nominal data into numeric 0 or 1 values based on mutually exclusive categories.
from sklearn.model_selection import train_test_split
import pandas as pd

# opening the CSV file
abalone = pd.read_csv('abalone.csv', names=['Sex','Length','Diameter','Height','WholeWt','ShuckedWt','VisceraWt','ShellWt','Rings'])

for col, i in zip(['IsInfant', 'IsMale', 'IsFemale'], ['I', 'M', 'F']):
    # add columns for IsFemale, IsMale, IsInfant
    abalone.insert(0, col, abalone['Sex'] == i)
    abalone[col] = abalone[col].astype(int)

# delete Sex column
abalone.pop('Sex')
# update csv file
abalone.to_csv('abalone1.csv', index=False)
# 4177 rows

train, test = train_test_split(
    abalone, train_size = 0.9)


train.to_csv('abalone_train.csv', index=False)
test.to_csv('abalone_test.csv', index=False)


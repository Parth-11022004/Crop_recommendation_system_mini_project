import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv('Crop.csv')
x = df.drop(columns='label', inplace=False)
y = df['label']
model = RandomForestClassifier()
model.fit(x,y)
features = ['N','P','K','temperature','humidity','ph','rainfall']
new_obs = []
for feature in features:
    k = float(input("Enter " + feature + " value : "))
    new_obs.append(k)
recommended_crop = model.predict([new_obs])
print("Recommended crop: ", recommended_crop)
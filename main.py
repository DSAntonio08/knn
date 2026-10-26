import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import pandas as pd
import random

plt.rcParams["toolbar"] = "None"

csv_fish_data = pd.read_csv("fish.csv")
x = csv_fish_data["Width"].tolist()
y = csv_fish_data["Length3"].tolist() # somehow no "Length" exists, js Length 1,2,3
species = csv_fish_data["Species"].astype("category")
labels = species.cat.codes.tolist() # 1Nr per fish, to know which fish is meant, which we can then use for the following names
species_names = species.cat.categories # .cat means categorize

german_names = {
    "Bream": "Brachse",
    "Parkki": "Güster",
    "Perch": "Flussbarsch",
    "Pike": "Hecht",
    "Roach": "Rotauge",
    "Smelt": "Stint",
    "Whitefish": "Renke",
}

fish_data = list(zip(x,y))

# 75% for training, 25% for testing; stratify so every species is in both parts
fish_data_train, fish_data_test, labels_train, labels_test = train_test_split(
    fish_data, labels, test_size=0.25, random_state=42, stratify=labels
)

knn = KNN(n_neighbors=3) # 3 is just a placeholder, idk yet what it'll be
knn.fit(fish_data_train, labels_train)

# grabs any rnmd fish out of fish_data_test
new_width,new_length = random.choice(fish_data_test) 
new_fish = [(new_width, new_length)]

prediction = knn.predict(new_fish)
predicted_fish_name = species_names[prediction[0]]

accuracy = knn.score(fish_data_test, labels_test)
print(f"Accuracy: {accuracy:.0%}")

# sets colors per fish
for name, group in csv_fish_data.groupby("Species"):
    plt.scatter(group["Width"], group["Length3"], label=german_names[name])

plt.scatter(new_width, new_length, color="red", marker="x", s=150, label="new fish")
plt.text(new_width, new_length, f"predicted: {german_names[predicted_fish_name]}", ha="right", va="bottom") # ha = horizontal alignment; va = vertical al.
    

plt.xlabel("Width [cm]")
plt.ylabel("Length3 [cm]")
plt.legend()
plt.show()
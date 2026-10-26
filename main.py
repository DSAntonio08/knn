import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN
import pandas as pd

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
knn = KNN(n_neighbors=3) # 3 is just a placeholder, idk yet what it'll be
knn.fit(fish_data, labels)

# still need to get new fish from csv
# placeHolders for now

new_width = 8
new_length = 20
new_fish = [(new_width, new_length)]

prediction = knn.predict(new_fish)

predicted_fish_name = species_names[prediction[0]]

# sets colors per fish
for name, group in csv_fish_data.groupby("Species"):
    plt.scatter(group["Width"], group["Length3"], label=german_names[name])

plt.scatter(new_width, new_length, color="red", marker="x", s=150, label="new fish")
plt.text(new_width, new_length, f"predicted: {german_names[predicted_fish_name]}", ha="right", va="bottom") # ha = horizontal alignment; va = vertical al.
    

plt.xlabel("Width [cm]")
plt.ylabel("Length3 [cm]")
plt.legend()
plt.show()
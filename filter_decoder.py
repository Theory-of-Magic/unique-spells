import numpy as np

with open("filters2.txt","r") as f:
    data = f.readlines()
    f.close()

names = ["name","level","school","range","concentration","V","S","M","Material Consumed","Material Cost (CP)",
 "area type","ability check","ability check","saving throw","cantrip scale","damage type", "n","s","m"]
for i in np.arange(len(data)):
    data[i] = data[i].replace("r is 0: ","")
    data[i] = data[i].replace("\n","")
    data[i] = data[i].replace("[","")
    data[i] = data[i].replace("]","")
    data[i] = data[i].replace(" ","")
    data[i] = data[i].split(",")

    for j in np.arange(len(data[i])):
        data[i][j] = int(data[i][j])
    print(data[i])
    

for d in data:
    msg = ""
    for i in d:
        msg += f'{names[i]}, '
    print(msg)
    

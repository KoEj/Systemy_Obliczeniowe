import threading
import requests

url = ["https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
       "https://static.scientificamerican.com/sciam/cache/file/92E141F8-36E4-4331-BB2EE42AC8674DD3_source.jpg?w=590&h=800&1966AE6B-E8E5-4D4A-AACA385519F64D03",
       "https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187.jpg?w=636&h=424",
       "https://instagram.com/favicon.ico"]

dir = "/Users/pawelniedziolka/Documents/Dev/Python/Systemy Obliczeniowe/Lab01/"


def download(name, item):
    response = requests.get(item)
    open(name, "wb").write(response.content)


def read(names, directory):
    i = 0
    for item in names:
        fileName = directory + str(i) + ".png"
        x = threading.Thread(target=download, args=(fileName, item))
        threads.append(x)
        x.start()
        i += 1


threads = list()
read(url, dir)

for index, thread in enumerate(threads):
    print("Before joining thread %d", index)
    thread.join()
    print("Thread %d done", index)

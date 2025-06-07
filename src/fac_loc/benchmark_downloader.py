
import requests
 

def load_pmedian_ORLIB_UNCAP():

    base_folder_to_save = '/Users/adeza3/Documents/Nuni__PhD/Year_1_2024_2025/foundationalCO/data/real_world/ORLIB__LocAss/pmedian_uncap'
    url = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

    for i in range(1,41):
        file_to_download = f"pmed{i}.txt"
        response = requests.get(url + file_to_download)
        if response.status_code == 200:
            with open(f"{base_folder_to_save}/{file_to_download}", "w") as file:
                file.write(response.text)
            print(f"{file_to_download} downloaded successfully!")
        else:
            print(f"Failed to download file, status code: {response.status_code}")

def load_pmedian_ORLIB_CAP():
    base_folder_to_save = '/Users/adeza3/Documents/Nuni__PhD/Year_1_2024_2025/foundationalCO/data/real_world/ORLIB__LocAss/pmedian_cap'
    file_to_download = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/pmedcap1.txt"
    response = requests.get(file_to_download)
    if response.status_code == 200:
        with open(f"{base_folder_to_save}/pmedcap1.txt", "w") as file:
            file.write(response.text)
        print(f"pmedcap1.txt downloaded successfully!")
    else:
        print(f"Failed to download file, status code: {response.status_code}")

def load_UncapWarehouseLoc():
    base_folder_to_save = '/Users/adeza3/Documents/Nuni__PhD/Year_1_2024_2025/foundationalCO/data/real_world/ORLIB__LocAss/warehouse_location'
    url = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/"

    ids = [41,42,43,44,51,61,62,63,64,71,72,73,74,81,
           82,83,84,91,92,93,94,101,102,103,104,111,112,
           113,114,121,122,123,124,131,132,133,134]
    for id in ids:
        file_to_download = f"cap{id}.txt"
        response = requests.get(url + file_to_download)
        if response.status_code == 200:
            with open(f"{base_folder_to_save}/{file_to_download}", "w") as file:
                file.write(response.text)
            print(f"{file_to_download} downloaded successfully!")
        else:
            print(f"Failed to download file, status code: {response.status_code}")

    for id in ["capa.txt","capb.txt","capc.txt",
               "capa.gz","capb.gz","capc.gz"]:
        response = requests.get(url + id)
        if response.status_code == 200:
            with open(f"{base_folder_to_save}/{id}", "w") as file:
                file.write(response.text)
            print(f"{id} downloaded successfully!")
        else:
            print(f"Failed to download file, status code: {response.status_code}")
 

#load_pmedian_ORLIB_UNCAP()
#load_pmedian_ORLIB_CAP()
load_UncapWarehouseLoc()
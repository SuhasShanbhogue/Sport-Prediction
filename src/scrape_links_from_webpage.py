import requests
from bs4 import BeautifulSoup
import re
import csv
import tqdm 
from pathlib import Path
data_folder = Path("../Data/")
def getData(yearNumber):
    URL = f"http://fs.ncaa.org/Docs/stats/w_volleyball_champs_records/{str(2010 + yearNumber)}/d1/html/confstat.htm"
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    breaker = [ "ucla-ill.htm", "tex-ore.htm",  "wiscpnst.htm", "finals3.htm", "final15.htm"]
    printing_on = False
    year = 2010 + yearNumber
    file_to_write = data_folder/f"links_{str(year)}.csv"
    with open(file_to_write, "w", newline = '') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["URL's"])
        for a in soup.find_all('a', href=True):
            if(a['href'] == breaker[yearNumber - 1]):
                printing_on = True
            if(printing_on):
                #print(a['href'])
                url = f"http://fs.ncaa.org/Docs/stats/w_volleyball_champs_records/{year}/d1/html/"
                k = a['href']
                b = [url + (str(k))]
                csvwriter.writerow(b)
            else:
                pass

if __name__ == "__main__":
    for k in tqdm.tqdm(range(5)):
        getData(k+1)
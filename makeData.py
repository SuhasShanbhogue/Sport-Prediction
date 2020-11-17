import scraper_links_of_matches as linkscrape
import scrape_table_of_matches as tablescrape
import tqdm
import argparse
from balanceClass import balanceClass

def main(decay=0.9):
  print("Data Scraping begins")
  print("Webscarper is scraping all the Match Links from 2011-2015")
  for k in tqdm.tqdm(range(5)):
    linkscrape.getData(k+1)
  print("All links scarped and dumped into links_$year.csv files . ")
  tablescrape.scrapeData(decay)
  print("All tables scraped and processed to trainable format.")
  print("Lets balance the Data now")
  balanceClass()
  print("Balancing algorithm finished execution")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=' Add the decay')
  parser.add_argument('--decay', type=float, default = 0.9,
                    help='an integer for the decay constant')
  args = parser.parse_args()
  decay = args.decay  

  main(decay)
  
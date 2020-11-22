
There is a directory "Data" which holds all the necessary data files generated during scraping, feature enggineering and everything.

This directory has the data scraped when WE scraped from the website.

"Data1" will contain the files that YOU generate while running the code.


We have coded the .py files in such a way that they consume the data from "Data" but write files in directory "Data1".

This is done to maintain uniformity in results as NCAA links time out unpredictably and the data is minutely dynamic in Nature.
For eg., our current dataset has data worth 299 matches. But it is possible, that due to some server issues/timeout errors, while you are executing the scrape
code, only 297 matches get scraped. Hence, this was necessary. This discrepancy is very small but we chose to consume data from the original directory - "Data"
so that you are able to reproduce the same results. We ensure you that same data would be generated via the scrape code, given the website acts the same.
If you want to check results for the newly generated data, please change the "Data" to "Data1" in the .py files. The files have a var named "data_folder", so if you change it, it will directly take the files from "Data1".


\src - contains all the .py files
\Data - contains original data files
\Data1 - files would get created here if you run the .py files

Run getData.py which is a driver for all the scraper functions.
First we scrape the links using scrape_links_from_webpage which generates csv files having all the matchwise hyper-links.
Then using these links we scrape the match data from NCAA using scrape_table_from_links.py
After that the data is balanced using balanceClass.py [Please find the explanation in the report]
Finally, players_final_dataset_generation.py generates the 3 player representatives data for the teams.

phase1_teamdata_allmodels.py
to run: python phase1_teamdata_allmodels.py
This code runs the 3 models about which we have described about in phase1 of the report and outputs their metric. The data for (2011-2015) has been taken from the Data folder for training.

metrics.py has the code to generate the necessary metrics for each test set.

plot_pruning_graph.py generates the ccp_alpha vs Impurities graph

neuralNetworksplayers, dtplayers and logisticRegplayers.py files have the model training code for the player-wise data.


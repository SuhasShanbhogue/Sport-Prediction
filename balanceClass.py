import pandas as pd
import numpy as np
import csv
import random 
random.seed(10)
import tqdm

def balanceClass():
	for year in tqdm.tqdm(["2011","2012","2013","2014","2015"]):
		df = pd.read_csv("./Data/players_"+year+".csv")
		df1 = pd.read_csv("./Data/team_"+year+".csv")
		assert len(df.index) == len(df1.index) , "check the Files"
		assert df1['winner'].sum() ==  df["Winner"].sum() , "check the files"
		ones = df1['winner'].sum()
		
		total = len(df1.index)
		delta = ones - (total/2)	
		print("Imbalance before:" + str(delta))
		listRows = []
		for i in range(len(df.index)):
			if int(df.iloc[i,-1:]) == 1:
				listRows.append(i)
		random.shuffle(listRows)
		for i in range(int(delta)):
			#print(i)
			rowNumber = listRows[i]
			#print(rowNumber)
			assert df.loc[rowNumber][-1] == df1.loc[rowNumber][-1] , "check the files"	
			teamData = df1.loc[rowNumber].to_numpy()
			playersData = df.loc[rowNumber].to_numpy()
			whoWonInit = df.loc[rowNumber][-1]
			
				#delta = delta- 1
			dataPlayers = playersData[:460]
			dataPlayersTeam1 = dataPlayers[:230]
			dataPlayersTeam2 = dataPlayers[230:460]
			dataTeams = teamData[:48]
			dataTeamsTeam1 = dataTeams[:24]
			dataTeamsTeam2 = dataTeams[24:48]
			assert len(dataPlayersTeam2) == len(dataPlayersTeam2) , "Error"
			assert len(dataTeamsTeam2) == len(dataTeamsTeam1) , "Error"
			newFirstTeam = dataTeamsTeam2
			lineAfterSecondTeam = np.append(newFirstTeam,dataTeamsTeam1)
			newFirstPlayerTeam = dataPlayersTeam2 
			lineAfterPlayerSecondTeam = np.append(newFirstPlayerTeam,dataPlayersTeam1)
			whoWonInit = not (whoWonInit)
			finalTeamLine = np.append(lineAfterSecondTeam,whoWonInit)
			finalPlayerLine = np.append(lineAfterPlayerSecondTeam,whoWonInit)
			df1.loc[rowNumber] = finalTeamLine
			#print(df1.loc[rowNumber])
			df.loc[rowNumber] = finalPlayerLine
			#print(df1.loc[rowNumber])
		
		#print(df)
		df.to_csv("./Data/playersBal_"+year+".csv",index=False)
		df1.to_csv("./Data/teamBal_"+year+".csv",index=False)
		#print(df['Winner'])
		print("Imbalance Now : " + str(int(df["Winner"].sum()-len(df.index)/2)))
		#print(len(df.index))


balanceClass()



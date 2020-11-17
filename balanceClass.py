import pandas as pd
import numpy as np
import csv
import random 
random.seed(10)
import tqdm
def balanceClass():

	for year in tqdm.tqdm(["2011","2012","2013","2014","2015"]):
		balance = 0 
		df = pd.read_csv("./Data/players_"+year+".csv")
		df1 = pd.read_csv("./Data/team_"+year+".csv")
		# writerP = csv.writer(file1)
		# writer = csv.writer(file2)
		assert len(df.index) == len(df1.index) , "check the Files"
		assert df1['winner'].sum() ==  df["Winner"].sum() , "check the files"
		ones = df1['winner'].sum()
		total = len(df1.index)
		delta = ones - (total/2)	
		imbalance = ((ones-(total/2))/(ones))
		for i in range(len(df.index)):
			assert df.loc[i][-1] == df1.loc[i][-1] , "check the files"	
			teamData = df1.loc[i].to_numpy()
			playersData = df.loc[i].to_numpy()
			whoWonInit = df.loc[i][-1]
			
			if(whoWonInit==1):	# we take care of only class 1 as in ncaa class 1 has the imbalance
				x = random.random()
				if(x>imbalance):
					continue
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
				#print(df1.loc[i])
				df1.loc[i] = finalTeamLine
				df.loc[i] = finalPlayerLine
				# if delta == 0:
				# 	break
			else:
				continue
		#print(df)
		df.to_csv("playersBal_"+year+".csv",index=False)
		df1.to_csv("teamBal_"+year+".csv",index=False)
		print(df1['winner'].sum())
		#print(df["Winner"].sum())
		print(len(df.index))



balanceClass()


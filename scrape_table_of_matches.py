import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import tqdm
from pathlib import Path
data_folder = Path("./Data/")
#dataDict = dict()
#dataPDict = dict()

def rowchef():
  datastr = ["SetsPlayed","AttackKills","AttackErrors","AttackTotalAttempts","AttackPCT","Assists","AssistErrors",
                "AssistTotalAttempts","AssistPCT","ServeAces","ServeErrors","ServeTotalAttempts","ServePCT",
                "BlockingDig","BallHandlingErrors","BlockSolos","BlockAssists","BlockErrors","ReceptO","ReceptErrors","ReceptPCT",
                "Block_Efficiency_Player","Ace_Strength"
                
                ]
  finalDatastr = []
  for i in range (20):
    for str1 in datastr:
      finalDatastr.append(str1)
  finalDatastr.append("Winner")
  #print(finalDatastr.shape)
  return finalDatastr

def tablechef(soupObj):
  dataTable = []
  for element in soupObj: 
      sub_data = [] 
      for sub_element in element: 
          try: 
              string = sub_element.text
              string = string[:len(string)-1]
              sub_data.append(string) 
          except: 
              continue
      dataTable.append(sub_data) 
  return pd.DataFrame(data = dataTable)

def weightschef(shape,decay=0.9):
  wt = []
  wt = np.array(wt,dtype=float)
  for i in range(shape):
    wt = wt*(decay)
    wt = np.append(wt,1)
  return wt

import csv
import numpy as np

def dataChef(dataDict,dataPDict,file1name="./Data/links_2011.csv",file2name="./Data/innovators_2011.csv",file3name="./Data/swarriors_2011.csv",decay=0.9):

  dflinks = pd.read_csv(file1name)
 
  
  file1 =  open(file2name, 'w', newline='');
  file2 = open(file3name,'w',newline='');
  writer = csv.writer(file1)
  writerP = csv.writer(file2)

  writer.writerow(["SetsPlayed","AttackKills","AttackErrors","AttackTotalAttempts","AttackPCT","Assists","AssistErrors",
                "AssistTotalAttempts","AssistPCT","ServeAces","ServeErrors","ServeTotalAttempts","ServePCT",
                "BlockingDig","BallHandlingErrors","BlockSolos","BlockAssists","BlockErrors","ReceptO","ReceptErrors","ReceptPCT",
                "TotalTeamBlocks", "winrate","Set_win_ratio","SetsPlayed","AttackKills","AttackErrors","AttackTotalAttempts","AttackPCT","Assists","AssistErrors",
                "AssistTotalAttempts","AssistPCT","ServeAces","ServeErrors","ServeTotalAttempts","ServePCT",
                "BlockingDig","BallHandlingErrors","BlockSolos","BlockAssists","BlockErrors","ReceptO","ReceptErrors","ReceptPCT",
                "TotalTeamBlocks", "winrate", "Set_win_ratio",
                  "winner"])

  writerP.writerow(rowchef())

  #len(dflinks.index)
  for i in tqdm.tqdm(range(len(dflinks.index))):
    try:
        res = requests.get(dflinks.loc[i][0])
        soup = BeautifulSoup(res.content,'html.parser')
        
        if "2012" not in file1name:
          table = (soup.find_all("table", border=0 ,cellspacing=0 ,cellpadding= 2))[5].find_all('tr')
          settable = (soup.find_all("table", border=0 ,cellspacing=0 ,cellpadding= 2))[7].find_all('tr')
          table1 = (soup.find_all("table", border=0 ,cellspacing=0 ,cellpadding= 2))[8].find_all('tr')
        else :
          table = (soup.find_all("table", border=0 ,cellspacing=0 ,cellpadding= 2))[6].find_all('tr')
          settable = (soup.find_all("table", border=0 ,cellspacing=0 ,cellpadding= 2))[8].find_all('tr')
          table1 = (soup.find_all("table", border=0 ,cellspacing=0 ,cellpadding= 2))[9].find_all('tr')

        
        df=tablechef(table)
        #print(list(df.columns))
        if len(df.columns)>23:
          #print(len(df.columns))
          df.drop(df.columns[len(df.columns)-1], axis=1, inplace=True)
        dfset = tablechef(settable)
        team1 = str(dfset.loc[1][0]).strip()
        team2 = str(dfset.loc[3][0]).strip()
        df1 = tablechef(table1)
        if len(df1.columns)>23:
          df1.drop(df1.columns[len(df1.columns)-1], axis=1, inplace=True)

        team1_score = (str(dfset.loc[1][1]).split('('))[1][0]
        team1_score = int(team1_score)
       
        
        team2_score = (str(dfset.loc[3][1]).split('('))[1][0]
        team2_score = int(team2_score)

        team1Array = []
        team2Array = []

        if i>0:

            if (team1.strip() in dataDict.keys()):
                team1Array = np.average(dataDict[team1],0,weightschef(len(dataDict[team1]),decay))
            else:
                team1Array = np.average(dataDict["goldStandard"],0,weightschef(len(dataDict["goldStandard"]),decay))

            if (team2.strip() in dataDict.keys()) :
                team2Array = np.average(dataDict[team2],0,weightschef(len(dataDict[team2]),decay))
            else:
                team2Array = np.average(dataDict["goldStandard"],0,weightschef(len(dataDict["goldStandard"]),decay))

            dataArray = np.append(team1Array,team2Array)
            dataArraywithPrediction = np.append(dataArray,team2_score > team1_score)
                                                    
            # print(dataArraywithPrediction)
            writer.writerow(dataArraywithPrediction)
            playerWriter = []
            team1PlayerCount = 0
            team2PlayerCount = 0

            for tuples in dataPDict.keys():
              if tuples[0]==team1:
                playerWriter.append(np.average(dataPDict[tuples],0,weightschef(len(dataPDict[tuples]),decay)))
                team1PlayerCount += 1
              if team1PlayerCount==10:
                break

            while team1PlayerCount<10:
              playerWriter.append(np.average(dataPDict[("goldStandard","X","X")],0,weightschef(len(dataPDict[("goldStandard","X","X")]),decay)))
              team1PlayerCount+=1

            for tuples in dataPDict.keys():  
              if tuples[0]==team2:
                playerWriter.append(np.average(dataPDict[tuples],0,weightschef(len(dataPDict[tuples]),decay)))
                team2PlayerCount += 1
              if team2PlayerCount==10:
                break

            while team2PlayerCount<10:
              playerWriter.append(np.average(dataPDict[("goldStandard","X","X")],0,weightschef(len(dataPDict[("goldStandard","X","X")]),decay)))
              team2PlayerCount+=1

            
            #print(len(playerWriter))
            playerWriterFlat = np.array(playerWriter,dtype=float).flatten()
            playerWriterFlatWithWinner = np.append(playerWriterFlat,dataArraywithPrediction[-1])
            writerP.writerow(playerWriterFlatWithWinner)
                

        
        set_win_ratio1 = str(team1_score/(team1_score + team2_score))
        set_win_ratio2 = str(team2_score/(team1_score + team2_score))

        line = (df.loc[df[1].str.contains('Total',na=False)]).to_numpy()
        lineClean = np.delete(line,0)
        lineClean1 = np.delete(lineClean,0)
        lineClean2 = np.append(lineClean1,(str(df.iloc[-1][1]).split())[3])
        #print(lineClean2)
        lineClean3 = np.append(lineClean2 , set_win_ratio1)
        #(lineClean3)
        line1 = (df1.loc[df1[1].str.contains('Total',na=False)]).to_numpy()     
        line1Clean = np.delete(line1,0)
        line1Clean1 = np.delete(line1Clean,0)
        #---------CHANGES
        line1Clean2 = np.append(line1Clean1,(str(df1.iloc[-1][1]).split())[3])
        #---------CHANGES
        line1Clean3 = np.append(line1Clean2 , set_win_ratio2)
        line2 = np.append(lineClean3,line1Clean3)
        #print(line2)
        lineAfterTeam2 = np.insert(line2,0,team2);
        lineAfterTeam1 = np.insert(lineAfterTeam2,0,team1);

        newLine1 = np.append(lineAfterTeam1 ,(str(dfset.loc[1][1]).split('('))[1][0])
        newLine2  = np.append(newLine1,(str(dfset.loc[3][1]).split('('))[1][0])
        winnerLine = np.append(newLine2 ,int(int(newLine2[len(newLine2)-1]) > 
                                                    int(newLine2[len(newLine2)-2]))) #team2 wins implies 1    
        #---------CHANGES
        data1 = np.append(winnerLine[2:25],int(winnerLine[-1]==0))
        data1 = data1.astype(float)
        data2 = np.append(winnerLine[25:48],int(winnerLine[-1]==1))
        data2 = data2.astype(float)
        dataDict.setdefault(team1.strip(), []).append(data1)
        dataDict.setdefault(team2.strip(), []).append(data2)
        dataDict.setdefault("goldStandard",[]).append(data1)
        dataDict.setdefault("goldStandard",[]).append(data2)
        for k in range(len(df.index)-3):
          if k<2:
            continue
          player = (df.loc[k]).to_numpy()
          playerStats = player[2:].astype(float)
          if (playerStats[13]+playerStats[15]+playerStats[16]+playerStats[14]+playerStats[17])==0:
              Block_Efficiency=0
          else:
              Block_Efficiency = (playerStats[13]+playerStats[15]+playerStats[16]-playerStats[14]-playerStats[17])/(playerStats[13]+playerStats[15]+playerStats[16]+playerStats[14]+playerStats[17])
          if playerStats[11]==0:
              ace_strength =0
          else:
              ace_strength = playerStats[9]/playerStats[11]
          playerStatsAfterBlockEngineering = np.append(playerStats,Block_Efficiency)
          playerStatsAfterAceEngineering = np.append(playerStatsAfterBlockEngineering,ace_strength)
          dataPDict.setdefault((team1.strip(),player[0].strip(),player[1].strip()),[]).append(playerStatsAfterAceEngineering)
          dataPDict.setdefault(("goldStandard","X","X"),[]).append(playerStatsAfterAceEngineering)

        for k in range(len(df1.index)-3):
          if k<2:
            continue
          player = (df1.loc[k]).to_numpy()
          playerStats = player[2:].astype(float)
          if (playerStats[13]+playerStats[15]+playerStats[16]+playerStats[14]+playerStats[17])==0:
              Block_Efficiency=0
          else:
              Block_Efficiency = (playerStats[13]+playerStats[15]+playerStats[16]-playerStats[14]-playerStats[17])/(playerStats[13]+playerStats[15]+playerStats[16]+playerStats[14]+playerStats[17])
          # --- Ace Strength
          if playerStats[11]==0:
              ace_strength =0
          else:
              ace_strength = playerStats[9]/playerStats[11]
          #----- MADE CHANGES
          playerStatsAfterBlockEngineering = np.append(playerStats,Block_Efficiency)
          playerStatsAfterAceEngineering = np.append(playerStatsAfterBlockEngineering,ace_strength)
          dataPDict.setdefault((team2.strip(),player[0].strip(),player[1].strip()),[]).append(playerStatsAfterAceEngineering)
          dataPDict.setdefault(("goldStandard","X","X"),[]).append(playerStatsAfterAceEngineering)

    except:
        #print(str(i) + " has an ERROR")
        pass #LINK has errors
    
      
      
  file1.close()

def scrapeData(decay=0.9):
  for i in ["2011","2012","2013","2014","2015"]:
    print(i+" will be scraped now.")
    dataDict = dict()
    dataPDict = dict()
    dataChef(dataDict,dataPDict,"./Data/links_"+i+".csv","./Data/team_"+i+".csv","./Data/players_"+i+".csv",decay)


if __name__ == "__main__":
  scrapeData()


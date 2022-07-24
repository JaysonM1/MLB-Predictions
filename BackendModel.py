#imports

import matplotlib.pyplot as plt
import pybaseball
from baseball_scraper import pitching_stats_range
from pybaseball import team_batting
from pybaseball import team_pitching
from pybaseball import pitching_stats
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from baseball_scraper import statcast as bs
from baseball_scraper import playerid_lookup
from baseball_scraper import statcast_pitcher
import mlbgame as mlb
import pandas, numpy
import requests, bs4
import re, os
from datetime import datetime
import random

name2init = {"Astros": "HOU","Dodgers": "LAD", "Rays": "TBR", "White Sox": "CHW",
        "Giants": "SFG","Blue Jays": "TOR","Padres": "SDP","Red Sox": "BOS",
        "Athletics": "OAK", "Cubs": "CHC", "Mets": "NYM",  "Yankees": "NYY",
        "Braves": "ATL", "Phillies": "PHI", "Marlins": "MIA", "Brewers": "MIL",
        "Angels": "LAA", "Reds": "CIN", "Indians": "CLE", "Nationals": "WSN",
        "Rangers": "TEX",  "Cardinals": "STL", "Mariners": "SEA","Royals": "KCR",
        "Rockies": "COL","Orioles": "BAL","Tigers": "DET", "Twins": "MIN", "D-backs": "ARI",
        "Pirates": "PIT"
       }

pitching = pitching_stats_range("2021-04-01","2021-09-30")

gameStatsCols =  ["Team1ID","p1ERA", "p1WHIP", "p1K9", "t1BA","t1OBP", "t1SLG", "t1ERA", "t1WHIP", "t1K9",
                    "Team2ID","p2ERA", "p2WHIP","p2K9", "t2BA", "t2OBP", "t2SLG","t2ERA" , "t2WHIP","t2K9"]
'''
makeGameDataFrame
params: 
frame - data frame for stats on each team
data - dataframe with prefilled column names 
        to be filled with data 

returns:
data - each row is a game, cols 0-9 are teams 1
       stats, 10-20 are team 2 stats
results - if result[i][0] = 1, team 1 was the winnner
            if result[i][1] = 2, team 2 was the winner
            index with 0 is loser
            result[0] will have winner of first game in data 
'''
def makeGameDataFrame(teamStats, gameData):
    ###list that will have game data
    result = []
    ##nested for loop that will go through
    ##each day
    for month in range(4,8):
        for day in range(1,31):
            schedules = mlb.games(2021,month,day)
            games = mlb.combine_games(schedules)
            ##games has data on each game on a given day
            for game in games:
                ##tries to get data on each game
                ##if there is no reported pitcher than 
                ##it is an attribute error, skip the game
                try:
                    ##random assigns team1 and team 2
                    ran = random.randint(0,1)
                    if not ran:
                        t1 = True
                        pitcher1 = game.w_pitcher
                        team1 = game.w_team
                        pitcher2 = game.l_pitcher
                        team2 = game.l_team
                    else:
                        t1 = False
                        pitcher2 = game.w_pitcher
                        team2 = game.w_team
                        pitcher1 = game.l_pitcher
                        team1 = game.l_team
                except AttributeError:
                    continue
                ##then attempts to add the data to teamStatsFrame
                # some pitchers arent in the api so it becomes index error 
                ##continue to next game
                try:
                    team1 = teamStats.loc[teamStats['Team'] == name2init[team1]]
                    df = {"Team1ID": team1.teamIDfg.values[0]}
                    p1 = pitching.loc[pitching['Name'] == pitcher1]
                    p1df = {"p1ERA": p1.ERA.values[0], "p1WHIP": p1.WHIP.values[0], "p1K9": p1.SO9.values[0]}
                    Merge(p1df,df)
                    t1bdf = {"t1BA": team1.AVG.values[0] ,"t1OBP": team1.OBP.values[0], "t1SLG": team1.SLG.values[0]}
                    Merge(t1bdf, df)
                    t1pdf = {"t1ERA": team1.ERA.values[0], "t1WHIP": team1.WHIP.values[0], "t1K9": team1.K9.values[0]}
                    Merge(t1pdf,df)
                    p2 = pitching.loc[pitching['Name'] == pitcher2]
                    team2 = teamStats.loc[teamStats['Team'] == name2init[team2]]
                    p2df = {"Team2ID": team2.teamIDfg.values[0], "p2ERA": p2.ERA.values[0], "p2WHIP": p2.WHIP.values[0] ,"p2K9": p2.SO9.values[0]}
                    Merge(p2df,df)
                    t2bdf = {"t2BA": team2.AVG.values[0], "t2OBP": team2.OBP.values[0], "t2SLG": team2.SLG.values[0]}
                    Merge(t2bdf,df)
                    t2pdf = {"t2ERA": team2.ERA.values[0] , "t2WHIP": team2.WHIP.values[0] ,"t2K9": team2.K9.values[0]}
                    Merge(t2pdf,df)
                    gameData = gameData.append(df,ignore_index=True)
                    if t1:
                        result.append([1,0])
                    else:
                        result.append([0,1])
                except IndexError:
                    continue
                except KeyError:
                    continue
    return gameData, result


##helper function that will 
##merge two dictionaries together
def Merge(dict1, dict2):
    return(dict2.update(dict1))


def makeModel(gameData,results):
    model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(20,)),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2,activation = 'softmax')])

    gameData = np.array(gameData)
    result = np.array(results)
    X_train, X_test, y_train, y_test = train_test_split(gameData,
                                                        result,
                                                        test_size=0.33,
                                                        random_state=42)
    model.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10)

    test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
    return model

def fetch_data(awayTeam, awayPitcher, homeTeam, homePitcher,teamStats):
    tdata = pandas.DataFrame(columns = gameStatsCols)
    team1 = teamStats.loc[teamStats['Team'] == name2init[awayTeam]]
    df = {"Team1ID": team1.teamIDfg.values[0]}
    p1 = pitching.loc[pitching['Name'] == awayPitcher]
    p1df = {"p1ERA": p1.ERA.values[0], "p1WHIP": p1.WHIP.values[0], "p1K9": p1.SO9.values[0]}
    Merge(p1df,df)
    t1bdf = {"t1BA": team1.AVG.values[0] ,"t1OBP": team1.OBP.values[0], "t1SLG": team1.SLG.values[0]}
    Merge(t1bdf, df)
    t1pdf = {"t1ERA": team1.ERA.values[0], "t1WHIP": team1.WHIP.values[0], "t1K9": team1.K9.values[0]}
    Merge(t1pdf,df)
    p2 = pitching.loc[pitching['Name'] == homePitcher]
    team2 = teamStats.loc[teamStats['Team'] == name2init[homeTeam]]
    p2df = {"Team2ID": team2.teamIDfg.values[0], "p2ERA": p2.ERA.values[0], "p2WHIP": p2.WHIP.values[0] ,"p2K9": p2.SO9.values[0]}
    Merge(p2df,df)
    t2bdf = {"t2BA": team2.AVG.values[0], "t2OBP": team2.OBP.values[0], "t2SLG": team2.SLG.values[0]}
    Merge(t2bdf,df)
    t2pdf = {"t2ERA": team2.ERA.values[0] , "t2WHIP": team2.WHIP.values[0] ,"t2K9": team2.K9.values[0]}
    Merge(t2pdf,df)
    tdata = tdata.append(df,ignore_index=True)
    tdata = np.array(tdata)
    return tdata

def dayPredictions(month, day, year, model,teamStats):
    games = mlb.day(year,month,day)
    for a_game in games:
        try:
            ateam = a_game.away_team
            apitcher = a_game.p_pitcher_home
            hteam = a_game.home_team
            hpitcher = a_game.p_pitcher_away
            if apitcher and hpitcher != '.':
                try:
                    print(ateam, apitcher, hteam, hpitcher)
                    print(model.predict(fetch_data(ateam ,apitcher, hteam, hpitcher,teamStats), verbose = 2))
                except IndexError:
                    print("Pass")
                    continue
        except AttributeError:
            continue 
        except KeyError:
            continue
def main():
    ##pulling data as a data frame
    teamBatting = team_batting(2021)
    teamPitching = team_pitching(2021)

    teamPitching = teamPitching[["teamIDfg", "Team","ERA","WHIP","K/9"]]
    teamBatting = teamBatting[ ["teamIDfg","AVG","OBP","SLG"]]

    teamStats = teamPitching.merge(teamBatting)
    teamStats.columns = ["teamIDfg", "Team","ERA","WHIP","K9","AVG","OBP","SLG"]
    #make dataframe that will contain all the data for each game
    gameStatsCols =  ["Team1ID","p1ERA", "p1WHIP", "p1K9", "t1BA","t1OBP", "t1SLG", "t1ERA", "t1WHIP", "t1K9",
                    "Team2ID","p2ERA", "p2WHIP","p2K9", "t2BA", "t2OBP", "t2SLG","t2ERA" , "t2WHIP","t2K9"]
    data = pandas.DataFrame(columns = gameStatsCols)
    ##data that could be normalized
    scaled_data= ["p1ERA","p1WHIP", "p1K9", "t1BA","t1OBP", "t1SLG", "t1ERA", "t1WHIP"
           ,"t1K9","p2ERA", "p2WHIP","p2K9", "t2BA", "t2OBP", "t2SLG","t2ERA" , "t2WHIP","t2K9"]

    gameData, result = makeGameDataFrame(teamStats, data)
    model = makeModel(gameData,result)
    dayPredictions(6,12,2021,model,teamStats)


if __name__ == "__main__":
    main()
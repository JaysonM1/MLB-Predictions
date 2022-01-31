# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 00:49:19 2021

@author: jayson
"""

from baseball_scraper import statcast as bs
from baseball_scraper import playerid_lookup
from baseball_scraper import statcast_pitcher
import mlbgame as mlb
import pandas, numpy
import requests, bs4
import re, os

max_era = 6.47
min_era = .56
max_whip = .53
min_whip = 1.76
min_war = .4
max_war = 4

schedules = mlb.games(2021,6,14)
games = mlb.combine_games(schedules)
print(games[0].w_pitcher)
w_pitcher = games[0].w_pitcher.split()
pitcher = playerid_lookup(w_pitcher[1],w_pitcher[0])
num = pitcher["key_mlbam"]
pitch_stats = statcast_pitcher('2021-04-1', '2021-06-16', num)
print(pitch_stats.to_string())
    

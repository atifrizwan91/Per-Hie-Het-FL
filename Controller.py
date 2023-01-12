# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:40:21 2022

@author: user
"""
# from pathlib import Path
# from server import server
from ThermalClient import ThermalClient
import sys
import json
from threading import Thread
# def start_server():
#     s = server()
#     s.start_parallel()


def get_configuration():
        with open('config.json') as json_file:
             conf = json.load(json_file)
        return conf
    
def start_client(country):
    print('-------------',country,' Started-----------------------')
    c = ThermalClient(country)
    t = Thread(target=c.start_parallel)
    t.start()

if __name__ == '__main__':
    conf = get_configuration()
    if conf['common_features_nodes'] is not None:
        for i in conf['common_features_nodes']:
            start_client(i)
    for i in conf['clients']:
        if( i =="Common"):
            continue
        start_client(i)
        
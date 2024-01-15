import networkx as nx
import pylab 
import numpy as np
import matplotlib.pyplot as plt





class PTline:
    '''Parameters:  idx : the name of this line. e.g. line RER B
                    waiting_time : average waiting_time
                    metro_station_list : all the metro_station ids by a list
                    metro_station_pos : all the metro_station positions by dict{ metro_station_id: (x,y) }
    '''
    def __init__(self, idx,waiting_time, metro_station_list, metro_station_pos,times):
        self.idx = idx
        self.waiting_time = waiting_time
        self.metro_station_list = metro_station_list
        self.metro_station_pos = metro_station_pos
        self.line_length = len( metro_station_list ) -1
        
        self.times = times
        
        self.line = []    

        for i in range(self.line_length):
            start_id = self.metro_station_list[i]
            stop_id = self.metro_station_list[i+1]
            time = self.times[i]
            self.line.append( ( start_id, stop_id, time ) )
            self.line.append( ( stop_id, start_id, time ) )
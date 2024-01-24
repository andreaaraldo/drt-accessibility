import numpy as np
import networkx as nx
import collections
import ptline
import matplotlib.pyplot as plt

# seed = np.random.seed(120)

def edgeOfBetweenCentroids(a, walking_speed):
    node_1,node_2,node_3,node_4 = a[0],a[1],a[2],a[3]
    # distance between node_1 and node_2 is 1 km, and walking is 4.5 km/h, so time cost is 1/4.5 h.
    edge_list =[(node_1,node_2,1.0/walking_speed),(node_2,node_1,1.0/walking_speed)
                ,(node_2,node_3,1.0/walking_speed),(node_3,node_2,1.0/walking_speed)
                ,(node_3,node_4,1.0/walking_speed),(node_4,node_3,1.0/walking_speed)
                ,(node_4,node_1,1.0/walking_speed),(node_1,node_4,1.0/walking_speed)
               ,(node_1,node_3,1.0*1.414/walking_speed),(node_3,node_1,1.0*1.414/walking_speed)
               ,(node_2,node_4,1.0*1.414/walking_speed),(node_4,node_2,1.0*1.414/walking_speed)]
    return edge_list

def edgeCentroidAndStation(centroid,metro_pos,all_pos,metro_waiting_time, walking_speed):
    metro_station_name = list( metro_pos.keys())
    metro_station_list = list( metro_pos.values() )
    centroid_pos = np.array( all_pos[centroid] )
    list_out = []
    for i in range(len(metro_station_name)):
        station_name = metro_station_name[i]
        if np.linalg.norm(centroid_pos-np.array(metro_station_list[i]))<=3:#if distance between centroid and station < 3km, people can walk to station.
            time_cost_walk_centroid_station = np.linalg.norm(centroid_pos-np.array(metro_station_list[i]))/walking_speed
            list_out.append( (centroid,station_name,time_cost_walk_centroid_station+metro_waiting_time[station_name]) )
            list_out.append( (station_name,centroid,time_cost_walk_centroid_station) )
    return list_out

def compute_Akinson(list_):
    #Akinson
    y_mean = np.mean( list_ )   
    sum_ = 0.
    for i in range( len(list_) ):
        sum_ = sum_ + list_[i]**(-1) 
    sum_ = 1 - (sum_/len(list_))**(-1)/y_mean
    return sum_

def compute_Pietra(list_):    
    #Pietra     
    sum_ = 0.
    y_mean = np.mean( list_ ) 
    for i in range( len(list_) ):
        sum_ = sum_ +  np.abs(list_[i]-y_mean)/y_mean
        sum_ = sum_/(2* len(list_) )
    return sum_
    
def compute_Theil(list_):    
    #Theil
    sum_ = 0.
    y_mean = np.mean( list_ ) 
    for i in range( len(list_) ):
        sum_ = sum_ +  list_[i]/y_mean*np.log( list_[i]/y_mean )
    sum_ = sum_/len(list_)
    return sum_

def compute_Gini(list_):  
    n_ = len(list_)
    list_.sort()
    sum_ = 0.
    for i in range( n_ ):
        sum_ = sum_ + (i+1)*list_[i]

    gini_ = 2*sum_/n_/np.sum(list_) - (n_+1)/n_
    return gini_

class Graph:
    
    def __init__(self, list_waiting_time, walking_speed):
        self.g = nx.DiGraph()
        
        self.metro_node = []
        self.metro_pos = {} 
        
        self.centroid_node = []
        self.centroid_pos = []  #It is an array of tuples. In the i-th elemtn of this array
                                # you find the x,y position of the i-th centroid
        
        self.all_node = []
        self.all_pos = {} 
        
        self.number_of_metro_stations = 0
        self.metro_waiting_time = {}
        self.list_waiting_time = list_waiting_time

        self.walking_speed = walking_speed

        self.node_color = []
        self.node_size = []

        self.all_stations = []
        self.all_edges = []
        self.centr_id_matr = np.zeros(shape=(1, 1)) # shape is temporary
        self.acc_matr = np.zeros(shape=(1, 1)) # shape is temporary
        self.pop_matr = np.zeros(shape=(1, 1)) # shape is temporary # Population matrix

        # These are the extreme x and y positions of all centroids
        leftmost=float('inf')
        rightmost=-float('inf')
        bottommost=float('inf')
        upmost=-float('inf')

        # Inequality indices
        self.ineq_Atkinson = 0
        self.ineq_Gini = 0
        
        
    def add_metro_line(self,metro_line):
        self.g.add_nodes_from(metro_line.metro_station_list)
        self.g.add_weighted_edges_from(metro_line.line)
        self.metro_node += metro_line.metro_station_list
        self.metro_pos = {**self.metro_pos.copy(), **metro_line.metro_station_pos}.copy()
        self.all_node = self.metro_node.copy()
        self.all_pos = self.metro_pos.copy()
        self.number_of_metro_stations += len( metro_line.metro_station_list)
        self.metro_waiting_time = {**self.metro_waiting_time.copy(), **dict.fromkeys(metro_line.metro_station_list,metro_line.waiting_time)}.copy()
        
        
    def add_connection(self, connection_and_transfer_time ):
        self.g.add_weighted_edges_from(connection_and_transfer_time)
        
    def add_centroids(self):
        centroid_node = []
        centroid_pos = []
        
        self.centroid_to_pos = {}
        
        old_centroid_node = [ i+80 for i in range(0,500) ]
        old_centroid_pos = np.reshape([[ (i,j) for j in range(20) ] for i in range(25)],(500,2))
        
        # Only centroids within 3 km from the bus station are kept.
        for i in range(len(old_centroid_pos)):
            if min(np.linalg.norm( np.array(old_centroid_pos[i])-np.array( list(self.metro_pos.values() )),axis=1 ))<3:
                centroid_node.append( old_centroid_node[i] )
                centroid_pos.append( old_centroid_pos[i] )
                self.centroid_to_pos[old_centroid_node[i]] = old_centroid_pos[i]

        
        self.g.add_nodes_from(centroid_node)
        self.centroid_node = centroid_node
        self.centroid_pos = centroid_pos
        
        
        self.all_node +=  centroid_node
        for i in range(len(self.centroid_node)):
            self.all_pos[  centroid_node[i] ] = tuple(centroid_pos[i]) 
            
    def add_edge_between_centroids(self):

        old_point_list = [[i,i+20,i+21,i+1] for i in self.centroid_node.copy()]
        
        point_list = []
        for point in old_point_list:
            if point[0] in self.centroid_node and point[1] in self.centroid_node and point[2] in self.centroid_node and point[3] in self.centroid_node:
                point_list.append(point)
        
        
        list_edge_a = []
        for point in point_list:
            list_edge_a+= edgeOfBetweenCentroids(point, self.walking_speed)
        list_edge_a = list(set(list_edge_a))
        self.g.add_weighted_edges_from(list_edge_a)
    
    def add_edge_between_centroid_and_station(self):
        list_edge = []
        for i in self.centroid_node:
            list_edge +=  edgeCentroidAndStation(i,self.metro_pos,self.all_pos,self.metro_waiting_time, self.walking_speed)
        self.g.add_weighted_edges_from(list_edge)
     
    
    def show(self):
        fig=plt.figure(figsize=(16,16))
        
        node_color=["r" for i in range(self.number_of_metro_stations)] + ['b' for i in range(len(self.centroid_node)) ]
        node_size=[50 for i in range(self.number_of_metro_stations)] + [10 for i in range(len(self.centroid_node)) ]
        nx.draw(self.g, self.all_pos, with_labels=True, node_color=node_color, node_size = node_size)
        plt.show()

    #aa: I would rename it into "compute_accessibility" as "get" methods do not usually modify the objects, while in this
    # case we modify the object graph, since we fill the attribute list_acc
    def compute_accessibility(self):
        
        #len(popu_list) = 500
        popu_list = [0,0,2417,2,0,0,0,0,85,1281,1872,2638,5300,0,1203,1115,3879,1073,708,1829,
        0,1352,1549,0,0,0,0,0,0,417,2960,1607,4986,1232,2461,3118,3667,2517,2212,1150,
        0,2296,1637,0,0,0,0,1,217,3723,2532,3127,4961,2946,0,3655,5159,2442,998,91,
        0,4472,2793,0,0,0,0,0,1428,5711,2013,4942,3201,4288,2516,1407,3852,2715,2063,811,
        0,2987,4912,32,7,0,0,0,1770,2920,3958,5486,2900,2837,1424,1755,3712,3565,2698,2014,
        0,3493,2978,147,612,99,219,0,2962,3682,3165,4247,7429,4133,1981,0,2550,2299,1751,2025,
        705,2821,2570,987,3016,1449,3490,1,105,903,2566,4635,10070,7094,4359,2992,0,0,1498,1960,
        2561,803,601,1470,6245,3930,3537,0,25,1271,1750,476,370,985,3803,4169,4954,2330,0,1424,
        2823,992,2761,2493,4307,2960,2497,1980,2035,1935,1873,2193,249,2055,1033,4388,4030,4764,3681,0,
        1187,713,2575,2779,4562,7686,1911,4495,5067,2235,2876,2184,6263,3913,3791,3668,3322,4797,4585,5420,
        3576,1322,206,3283,4242,4224,4195,6375,11354,6228,2067,2124,7398,2680,5891,2722,2040,4463,3552,6023,
        4784,7010,162,0,7314,5120,7602,6652,9853,7916,8972,2768,10572,3748,9009,3668,346,7243,2869,2366,
        4334,4108,43,0,3655,5298,5467,5170,5710,4273,6173,4091,6094,7824,7126,3437,368,5363,5366,1909,
        2248,2699,1754,60,788,6473,4264,2237,3248,733,2709,7914,4039,6882,8372,7303,3314,1088,6144,5937,
        4482,1907,1834,3701,284,2176,5261,1889,1139,20,1354,8010,5892,7640,8603,6177,5076,3150,6852,4090,
        4154,2474,3013,6785,4600,2669,3635,3471,990,107,1514,8378,4916,6165,7545,6022,7536,3626,3704,4817,
        0,2173,1919,4700,4916,3909,5401,5087,4921,536,580,8545,7891,6255,6597,5192,5357,4047,4416,2714,
        0,0,4898,5894,3907,911,4313,6773,12477,5571,5134,8742,8786,6946,6608,6053,6919,5386,5487,6101,
        0,0,2070,9360,7918,4396,4404,5516,2398,3434,9042,7645,5273,6154,5324,7713,4676,4544,4870,5501,
        0,0,0,0,3977,3227,6010,0,3587,2204,4716,8196,8312,5627,5728,6976,4,2104,3974,4500,
        0,0,0,1193,2646,134,2991,183,1336,2058,4642,6309,4599,5802,6554,4017,61,2478,3594,3293,
        0,0,0,4974,4786,393,126,106,0,0,499,435,3484,2963,8861,4953,2501,2251,5113,1363,
        0,0,0,0,0,341,0,0,0,0,0,0,0,0,6719,6990,1885,3362,5612,2587,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,3159,1426,1276,892,2202,2527,
        0,0,0,0,0,0,0,0,0,0,141,850,4020,0,0,0,0,1406,671,1133]
        
        
  
        self.centroid_population = {}
        for i in range(len(self.centroid_node)):
            self.centroid_population[self.centroid_node[i]] = popu_list[self.centroid_node[i]-80]
        self.total_population = np.sum( list(self.centroid_population.values()) )
        
        #claculate acc
        self.centroid_to_acc = {}
        length =  nx.all_pairs_dijkstra_path_length(self.g)
        list_acc = []
        for i, dict_ in length:
            if i in self.centroid_node:
                acc_i = 0.0
                for j in dict_.keys():
                    if j in self.centroid_node and j != i:
                        acc_i = acc_i + self.centroid_population[j]/dict_[j]
                list_acc.append(acc_i)
                self.centroid_to_acc[i] = acc_i
                
        list_acc_0 = list_acc.copy()
        list_acc_0.sort()
        
        #Akinson
        self.ineq_Atkinson = compute_Akinson(list_acc)
        
        ##Pietra     
        #sum_P = compute_Pietra(list_acc)
        
        #Gini
        self.ineq_Gini = compute_Gini(list_acc)

        
        return [np.mean(list_acc_0),list_acc,self.ineq_Atkinson,self.ineq_Gini]

    def find_limits(self):
        """
        Returns centroids the limits of the x- and y-positions of the
        centroids
        """
        leftmost=float('inf')
        rightmost=-float('inf')
        bottommost=float('inf')
        upmost=-float('inf')
        for pos in self.centroid_pos:
            leftmost = min(leftmost, pos[0])
            rightmost = max(rightmost, pos[0])
            bottommost = min(bottommost, pos[1])
            upmost = max(upmost, pos[1])
        
        return leftmost, rightmost, bottommost, upmost

    def build_accessibility_matrix(self):
          self.compute_accessibility()
        
          rows = self.upmost-bottommost+1
          cols = self.rightmost-leftmost+1
          acc_matr = np.array([([float('nan')]*cols) for i in range(rows)])
          centr_id_matr = np.array([([float('nan')]*cols) for i in range(rows)])
        
          for centr in self.centroid_node:
            acc = self.centroid_to_acc[centr]
            pos = self.centroid_to_pos[centr]
            centr_id_matr[ self.upmost-pos[1], pos[0] ]=centr
            acc_matr[ self.upmost-pos[1], pos[0] ]=acc
        
          return centr_id_matr, acc_matr

    def build_population_matrix(self):
          rows = self.upmost-bottommost+1
          cols = self.rightmost-leftmost+1
          pop_matr = np.array([([float('nan')]*cols) for i in range(rows)])
          
          for centr in self.centroid_node:
            population = self.centroid_population[centr]
            pos = self.centroid_to_pos[centr]
            pop_matr[ self.upmost-pos[1], pos[0] ]=population
        
          return pop_matr

    #}end of Graph class

    



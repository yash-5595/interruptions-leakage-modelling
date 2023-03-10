# direct_det_mapping_siia = {
#     1430:{
#         'stp':{1:[1,2],2:[3,4,5],3:[9], 4:[10,11],5:[12], 6:[13, 14],7:[],8:[19, 20]},
#         'adv':{}
#     },
    
#     1435:{
#          'stp':{1:[1],2:[2,3,4],3:[18,19,20], 4:[18,19,20],5:[10], 6:[11,12,13],7:[18,19,20],8:[18,19,20]},
#         'adv':{2:[5,6,7],6:[14,15,16]}
#     }
    
# }


exp_dict_all = {

  'segment_1425_1430':{
    
    'inputs':{'isc':1425,
              'type':'stp',
             'phases':[6,8],
             'sig':True},
    
    'outputs' : {
        'isc':1430,
        'type':'adv',
        'phases' :[2],
        'sig':False}
},

'segment_1430_1435':{
    
    'inputs':{'isc':1430,
              'type':'stp',
             'phases':[2,4,8],
             'sig':True},
    
    'outputs' : {
        'isc':1435,
        'type':'adv',
        'phases' :[2],
        'sig':False}
},

'segment_1435_1440':{
    
    'inputs':{'isc':1435,
              'type':'stp',
             'phases':[2,4,8],
             'sig':True},
    
    'outputs' : {
        'isc':1440,
        'type':'adv',
        'phases' :[2],
        'sig':False}
},


'segment_1440_1445':{
    
    'inputs':{'isc':1440,
              'type':'stp',
             'phases':[2,4,8],
             'sig':True},
    
    'outputs' : {
        'isc':1445,
        'type':'adv',
        'phases' :[2],
        'sig':False}
},

'segment_1445_1455':{
    
    'inputs':{'isc':1445,
              'type':'stp',
             'phases':[2,4,8],
             'sig':True},
    
    'outputs' : {
        'isc':1455,
        'type':'adv',
        'phases' :[6],
        'sig':False}
},


'segment_1455_1460':{
    
    'inputs':{'isc':1455,
              'type':'stp',
             'phases':[6,4,8],
             'sig':True},
    
    'outputs' : {
        'isc':1460,
        'type':'adv',
        'phases' :[2],
        'sig':False}
},


'segment_1460_1465':{
    
    'inputs':{'isc':1460,
              'type':'stp',
             'phases':[2,4,8],
             'sig':True},
    
    'outputs' : {
        'isc':1465,
        'type':'adv',
        'phases' :[2],
        'sig':False}
},


    
}



det_channel_mapping = {1425: {'adv': {2: [5, 6, 7], 6: [13, 14, 15]},
  'stp': {1: [1], 2: [2, 3, 4], 5: [9], 6: [10, 11, 12], 8: [8, 16]}},
 1430: {'adv': {2: [6, 7, 8], 6: [16, 17, 18]},
  'stp': {1: [1, 2],
   2: [3, 4, 5],
   3: [9, 10],
   4: [19, 11],
   5: [12],
   6: [13, 14, 15],
   8: [20]}},
 1435: {'adv': {2: [5, 6, 7], 6: [14, 15, 16]},
  'stp': {1: [1],
   2: [2, 3, 4],
   4: [17, 18, 19, 9],
   5: [10],
   6: [11, 12, 13],
   8: [8, 20]}},
 1440: {'adv': {2: [5, 6, 7], 6: [15, 16, 17]},
  'stp': {1: [1],
   2: [2, 3, 4],
   4: [10],
   5: [11],
   6: [12, 13, 14],
   8: [8, 9, 18]}},
 1445: {'adv': {2: [5, 6, 7], 6: [14, 15, 16]},
  'stp': {1: [1],
   2: [2, 3, 4],
   4: [9, 17],
   5: [10],
   6: [11, 12, 13],
   8: [8, 18]}},
 1455: {'adv': {2: [5, 6, 7], 6: [14, 15, 16]},
  'stp': {1: [1], 2: [2, 3, 4], 4: [9], 5: [10], 6: [11, 12, 13], 8: [8, 17]}},
 1460: {'adv': {2: [5, 6, 7], 6: [15, 16, 17]},
  'stp': {1: [1],
   2: [2, 3, 4],
   3: [8, 9],
   4: [10, 18],
   5: [11],
   6: [12, 13, 14],
   8: [19]}},
 1465: {'adv': {2: [5, 6, 7], 6: [14, 15, 16]},
  'stp': {1: [1],
   2: [2, 3, 4],
   4: [17, 9],
   5: [10],
   6: [11, 12, 13],
   8: [8, 18]}}}





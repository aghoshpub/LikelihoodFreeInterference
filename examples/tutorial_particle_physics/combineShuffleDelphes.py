# run this from terminal with madminer stuff installed to be safe
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from madminer.sampling import combine_and_shuffle



# MadMiner output
logging.basicConfig(
    format='%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s',
    datefmt='%H:%M',
    level=logging.DEBUG
)

# Output of all other modules (e.g. matplotlib)
for key in logging.Logger.manager.loggerDict:
    if "madminer" not in key:
        logging.getLogger(key).setLevel(logging.WARNING)
        
        
        
mg_dir = '/home/software/MG5_aMC_v2_6_2/'
        
delphesDatasetList = ['data/delphes_data{}.h5'.format(i) for i in range (1,101)]
#delphesDatasetList = ['data/delphes_data{}.h5'.format(i) for i in range (1,6)] #for testing

combine_and_shuffle(
    delphesDatasetList,
    'data/delphes_data_shuffledBig.h5'
)
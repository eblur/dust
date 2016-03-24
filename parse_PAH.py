
## Created by Lia Corrales to parse PAH optical constant tables (PAHion_30, PAHneu_30)
## November 11, 2013 : lia@astro.columbia.edu

import os
import numpy as np

def find_cmfile( name ):
    if os.path.exists(name): 
        return name
    path_list = os.getenv("PYTHONPATH").split(':')
    for path in path_list:
        for root, dirs, files in os.walk(path+"/"):
            if name in files:
                return os.path.join(root, name)
        else:
            return ""

ION_FILE  = find_cmfile('PAHion_30')
NEU_FILE  = find_cmfile('PAHneu_30')

def parse_PAH( option, ignore='#', flag='>', verbose=False ):
    
    if option == 'ion': filename = ION_FILE
    if option == 'neu': filename = NEU_FILE
    
    try : f = open( filename, 'r' )
    except:
        print('ERROR: file not found')
        return
    
    COLS = ['w(micron)', 'Q_ext', 'Q_abs', 'Q_sca', 'g=<cos>' ]
    result = {}
    
    end_of_file = False
    while not end_of_file:
        try:
            line = f.readline()
            
            # Ignore the ignore character
            if line[0] == ignore : pass
            
            # Characters flagged with '>' earn a dictionary entry with grain size
            elif line[0] == flag : 
                gsize = np.float( line.split()[1] )
                if verbose : print('Reading data for grain size:', gsize)
                result[ gsize ] = {}
                # Initialize dictionaries with lists
                for i in range( len(COLS) ) : result[gsize][COLS[i]] = []
            
            # Sort the columns into the correct dictionary
            else:
                row_vals = line.split()
                for i in range( len(COLS) ) :
                    result[ gsize ][ COLS[i] ].append( np.float( row_vals[i] ) )
        except:
            if verbose : print(line)
            end_of_file = True
    
    f.close()
    
    return result

#test_ion = parse_PAH('ion')
#test_neu = parse_PAH('neu')

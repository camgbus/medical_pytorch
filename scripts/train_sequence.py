import os
import sys
from subprocess import Popen

if __name__ == '__main__':
    py_file_path = sys.argv[1]
    device = sys.argv[2]
    tcr = sys.argv[3]
    idle_time = sys.argv[4]
    artefacts = ['noise', 'resolution', 'spike'] #['noise', 'resolution', 'spike'] ['blur', 'ghosting', 'motion']

    for artefact in artefacts:
        Popen('python ' + py_file_path + ' --mode train --device ' + device + ' --datatype train --noise_type ' +\
              artefact + ' --use_telegram_bot --store_data', shell=True)

        # Execute python train command using telegram bot and try catch repeat of tcr with idle time of idle_time seconds
        #os.system('python ' + py_file_path + ' --mode train --device ' + device + ' --datatype train --noise_type ' +\
        #        artefact + ' --use_telegram_bot --store_data')# --try_catch_repeat ' + tcr + ' --idle_time ' + idle_time)
        #os.system('python ' + py_file_path + ' --mode testOOD --device ' + device + ' --datatype test --noise_type ' +\
        #        artefact + ' --use_telegram_bot --store_data')#--try_catch_repeat ' + tcr + ' --idle_time ' + idle_time)
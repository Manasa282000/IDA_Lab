import sys
import csv
import json


from collectors.run_single_configuration import run_single_configuration
if len(sys.argv) < 2:
    "No command specified."
    exit(1)

command = sys.argv[1]

if command == "csv" :
    if len(sys.argv) < 4 :
        print("Usage python3 python/run.py csv  <configs.csv> INDEX")
        exit(1)
    csvfilename = sys.argv[2]
    index   = int(sys.argv[3])

    force = False
    if len(sys.argv) == 5 :
        force = (sys.argv[4] == 'rerun')

    try :
        csvfile = open(csvfilename, newline='')
    except :
        print("Failed to open the CSV file: {}".format(csvfilename))
        exit(1)

    configreader = csv.DictReader(csvfile, delimiter=',', quotechar='|')

    config = None
    numconfigs = 0
    for i, x in enumerate(configreader):
        numconfigs += 1
        if i == index :
            config = x
            break

    if config == None :
        print("Configuration index {} out of range [0, {})".format(index, numconfigs))
        exit(1)

    # set the configuration index
    config['idx'] = index

    run_single_configuration(config, force)

if command == "test" :
    if len(sys.argv) < 3:
        # change the config here
        config = {
            'dataset': 'compas',
            'thresh': False,
            'lb_guess': True,
            'fold': 2,
            'algorithm': 'gosdt',
            'regularization': 0.001,
            'depth_budget': 4,
            'optimaldepth': 0,
            'time_limit': 1800,
            'max_depth': 3,
            'n_est': 20,
            'lb_max_depth': 3,
            'lb_n_est': 20,
            'cross_validate': False, }
    else :
        #pass some json as cmdline
        config = json.loads(sys.argv[2])
    run_single_configuration(config, force=True)
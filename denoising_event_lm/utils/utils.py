import json, pickle
import sys, os
import traceback


def read_data(path, args):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        data_type = 'json'
    except:
        #traceback.print_exc(limit=None, file=sys.stdout)
        json_tb = traceback.format_exc(limit=None)
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            data_type = 'pickle'
        except:
            #traceback.print_exc(limit=None, file=sys.stdout)
            pkl_tb = traceback.format_exc(limit=None)
            print("*** JSON Traceback:\n"+json_tb)
            print("*** Pickle Traceback:\n"+pkl_tb)
            exit()
    if args is not None and hasattr(args, 'start'):
        if (not args.start < 0) and (not args.end < 0):
            print("data %d ~ %d" % (args.start, args.end))
            data = data[args.start:args.end]
        elif (not args.start < 0):
            print("data %d ~ %d" % (args.start, len(data)))
            data = data[args.start:]
        elif (not args.end < 0):
            print("data %d ~ %d" % (0, args.end))
            data = data[:args.end]
        else:
            print("data %d ~ %d" % (0, len(data)))
    else:
        print("no start end args")
        print("data %d ~ %d" % (0, len(data)))
    return data, data_type



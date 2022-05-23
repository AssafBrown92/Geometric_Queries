import code, sys
from model import *

def main():

    #Reading the model path
    if len(sys.argv) != 2:
        print "USAGE: %s <MODEL_PATH>" % sys.argv[0]
        return
    model_path = sys.argv[1]

    #Loading the model and presenting and interactive shell
    model = None
    try:
        print "Loading Model..."
        model = Model.load(model_path)
    except Exception, ex:
        print "Failed to load model %s" % model_path
        return
    code.interact(banner="Loaded Model into local 'model'", local=locals())

if __name__ == "__main__":
    main()

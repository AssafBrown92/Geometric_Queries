import sys
from nltk.tokenize import sent_tokenize

def main():

    #Reading the arguments
    if len(sys.argv) != 3:
        print("USAGE: %s <INPUT_CORPUS> <OUTPUT_PATH>" % sys.argv[0])
        return
    input_file = open(sys.argv[1], "r", errors="replace")
    output_file = open(sys.argv[2], "w")

    #Processing the original corpus file
    counter = 1
    for line in input_file:

        #Skipping source lines
        if line.startswith("CURRENT URL"):
            continue
       
        #Tokenising the content into lines
        for sentence in sent_tokenize(line):
            output_file.write("%d\t%s\n" % (counter, sentence))
            counter += 1

    output_file.close()

if __name__ == "__main__":
    main()

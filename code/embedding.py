import sys, string, pickle, scipy.sparse
import numpy as np
from progress.bar import Bar

from nltk.tokenize.nist import NISTTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.preprocessing import normalize
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import TruncatedSVD

from miniball import miniball
from model import Model

#The default size of the context window used when creating the cooccurrence matrix.
DEFAULT_CONTEXT_SIZE = 6

#A set containing all the stopwords in the English language.
STOPWORDS = set(stopwords.words('english'))

#The default epsilon value used for the Johnson-Lindenstrauss Lemma when
#performing a random projection to reduce the dimension. Increasing this
#value will result in a less accurate embedding, with a smaller dimension.
EPSILON = 0.1

#The size to which each word vector is scaled
SCALING_FACTOR = 100

def is_nonword(word):
    '''
    :brief Checks whether the word should be filtered, according to the following rules:
      1. Must contain only alphanumeric characters
      2. Must not be a stopword
    :param word The word to be checked.
    :return True iff the word should be filtered.
    '''

    if not word.isalpha():
        return True
    if word in STOPWORDS:
        return True
    return False

def parse_corpus(corpus_path, vocabulary=None):
    '''
    :brief Parses the corpus at the given path, returning the parsed sentences (where
           each sentence is a list of words). Words are filtered using the "is_nonword"
           function, and tokenisation is one using the NLTK NIST tokenizer.
    :param corpus_path The path of the corpus file to parse.
    :param vocabulry The set of 'allowed' words in the vocabulry.
    :return A tuple of two lists (parsed_sentences, original_sentences) produced from the corpus.
    '''

    original_sentences = []
    parsed_sentences = []
    nist = NISTTokenizer()
    with open(corpus_path, "r") as corpus:
        for line in Bar("Parsing Corpus").iter(corpus.readlines()):
            
            #Tokenising each line
            line_number, line_content = line.split("\t", 1)
            if not line_number.isdigit():
                raise Exception("Invalid corpus file, unexpected line: %s" % line)
            
            original_sentence = line_content
            try:
                #Sanitising the contents
                line = []
                for word in nist.tokenize(line_content, lowercase=True):
                    if vocabulary and not word in vocabulary:
                        continue
                    if is_nonword(word):
                        continue
                    line.append(word)
                        
                if len(line) > 0:
                    original_sentences.append(original_sentence)
                    parsed_sentences.append(line)
            except UnicodeDecodeError, e:
                #Skipping non-ascii characters
                continue
 
    return (parsed_sentences, original_sentences)

def create_cooccurrence_matrix(sentences, context_size=DEFAULT_CONTEXT_SIZE):
    '''
    :brief Creates a sparse co-occurrence matrix for the given sentences.
    :param sentences A list of sentences (where each sentence is a list of words)
    :param context_size The size of the context window
    :return A tuple containing the following elements (in order):
              matrix - is the sparse (lil_matrix) cooccurrence matrix
              word_to_idx - A dictionary mapping words to their indices in the matrix
              idx_to_word - A dictionary mapping indices to the corresponding words
              wordidx_to_sentences - A dictionary mapping word indices to sentence indices
              indexed_sentences - A collection of sentences, containing only word indices
    '''

    #Collecting all the words in the corpus and creating mapping from word <-> index
    #(where the index corresponds to that in the resulting matrix)
    vocab = set.union(*[set(sentence) for sentence in sentences])
    word_to_idx = {}
    idx_to_word = {}
    wordidx_to_sentences = {}
    for idx, word in enumerate(vocab):
        word_to_idx[word] = idx
        idx_to_word[idx] = word

    #Creating a *sparse* cooccurrence matrix. This is necessary since a full matrix
    #will not fit in memory for a decent-sized vocabulary. Since the resulting
    #matrix is indeed sparse, this method is far more efficient.
    n = len(vocab)
    matrix = scipy.sparse.lil_matrix((n,n))

    #Populating the cooccurrence matrix
    indexed_sentences = []
    for sentence_idx, sentence in Bar("Creating Cooccurrence Matrix",
                                      max=len(sentences)).iter(enumerate(sentences)):
       
        #Converting the sentence into the indices of each word in the vocab
        sentence = [word_to_idx[word] for word in sentence]
        indexed_sentences.append(sentence)

        #Collecting the stats for each window
        for idx in range(0, len(sentence)):
            
            current = sentence[idx]

            #Updating the word->sentences mapping
            if current not in wordidx_to_sentences:
                wordidx_to_sentences[current] = []
            wordidx_to_sentences[current].append(sentence_idx)
            
            #Updating the cooccurrence matrix
            window = sentence[max(idx-context_size, 0) : idx+context_size]
            for word_idx in window:
                if word_idx == current: continue
                matrix[current, word_idx] += 1
       
    return (matrix, word_to_idx, idx_to_word, wordidx_to_sentences, indexed_sentences)

def parse_embedding(embedding_path):
    '''
    :brief Parses the pre-existing embedding at the given path.
    :param embedding_path The path of the embedding file to parse. Each line in the file
                          has a single lowercase word, followed by a list of space-separated
                          coordinates of the corresponding embedding.
    :return A tuple of (vocab, embedding, n), where:
              vocab - A set containing the words in the vocabulary
              embedding - A dictionary from word -> embedding vector
              n - The dimension
    '''

    vocab = set()
    embedding = {}
    n = None

    #Parsing each word in the embedding
    with open(embedding_path, "r") as embedding_file:
        for line in Bar("Parsing Embedding").iter(embedding_file.readlines()):

            #Parsing the line
            tokens = line.split(" ")
            word, vect = tokens[0], tokens[1:]
            vect = [float(v) for v in vect]

            #Updating the data structures
            vocab.add(word)
            embedding[word] = vect
            if n == None:
                n = len(vect)

    return (vocab, embedding, n)

def print_usage():
    '''
    :brief Prints usage instructions for this script
    '''

    print "USAGE: %s create <CORPUS> [pca/random] <OUTPUT_DIMENSION> <OUTPUT_MODEL>" % sys.argv[0]
    print "       %s import <CORPUS> <EMBEDDING> <DIMENSION> <OUTPUT_MODEL>" % sys.argv[0]

def main():

    if len(sys.argv) != 6:
        print_usage()
        return

    #Creating the model used to compute queries on the given corpus.
    #The key component used by the model is a cooccurrence matrix which is transformed
    #into a lower dimension using either random sparse projections (relying on the JL-Lemma),
    #or by applying Sparse PCA to retrive the N primary components.
    if sys.argv[1] == "create":
    
        #Parsing the input arguments
        corpus_path = sys.argv[2]
        reduction_mode = sys.argv[3]
        output_dimension = int(sys.argv[4])
        output_path = sys.argv[5]

        if reduction_mode not in ["pca","random"]:
            print "Unknown dimensionality reduction mode"
            return

        #Creating a cooccurrence matrix from the corpus
        parsed_sentences, original_sentences = parse_corpus(corpus_path)
        matrix, word_to_idx, idx_to_word, wordidx_to_sentences, indexed_sentences = \
            create_cooccurrence_matrix(parsed_sentences)

        #Normalising the rows of the matrix as a probability simplex, and scaling by
        #a constant factor to "stretch" out the resulting embedding.
        #Note: Normalisation also helps prevent biasing embedding of frequent words.
        matrix = normalize(matrix, axis=1, norm='l1')
        matrix = np.multiply(matrix, SCALING_FACTOR)
       
        #Performing the chosen dimensionality reduction
        print "Reducing dimension..."
        if reduction_mode == "pca":
            reduced_matrix = TruncatedSVD(n_components=output_dimension).fit_transform(matrix)
        else:
            reduced_matrix = SparseRandomProjection(eps=EPSILON).fit_transform(matrix).todense()
        matrix = None #Allowing the full matrix to be collected
        
        #Saving the resulting model
        model = Model(indexed_sentences, reduced_matrix,
                      word_to_idx, idx_to_word,
                      wordidx_to_sentences, original_sentences)
        model.save(output_path)

    elif sys.argv[1] == "import":
        
        #Parsing the input arguments
        corpus_path = sys.argv[2]
        embedding_path = sys.argv[3]
        dimension = int(sys.argv[4])
        output_path = sys.argv[5]

        #Parsing the imported embedding
        vocabulary, embedding, n = parse_embedding(embedding_path)
        if n != dimension:
            print "Mismatching dimension - user requested %d, but imported model is %d" % (dimension, n)

        #Analysing the corpus, while only using the imported model's vocabulary
        #Note that the cooccurrence matrix is discarded, and instead we use the
        #embedding derived from the imported model.
        parsed_sentences, original_sentences = parse_corpus(corpus_path, vocabulary)
        _, word_to_idx, idx_to_word, wordidx_to_sentences, indexed_sentences = \
            create_cooccurrence_matrix(parsed_sentences)

        #Creating a dense matrix from the imported embedding
        known_words = set(word_to_idx.keys())
        matrix = np.ndarray((len(known_words),n))
        for word in known_words:
            matrix[word_to_idx[word], :] = embedding[word]

        #Saving the resulting model
        model = Model(indexed_sentences, matrix,
                      word_to_idx, idx_to_word,
                      wordidx_to_sentences, original_sentences)
        model.save(output_path)

    else:
        print "Unknown option %s" % sys.argv[1]

if __name__ == "__main__":
    main()

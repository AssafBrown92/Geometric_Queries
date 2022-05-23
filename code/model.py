import pickle, heapq
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from contextlib import closing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from miniball import miniball
from ball import *

#The default batch size used when performing large-scale operations on the
#matrix, to avoid duplicating the entire matrix in memory.
DEFAULT_BATCH_SIZE = 10000

#The default number of neighbours to retrieve when finding "closest" words.
DEFAULT_NEIGHBOUR_COUNT = 10

#The default size of the context window used around a word.
DEFAULT_CONTEXT_SIZE = 6

#The default number of senses attempted when performing polysemy analysis.
DEFAULT_POLYSEMY_SENSE_COUNT = 2

#The default number of representatives printed when performing polysemy analysis.
DEFAULT_POLYSEMY_REPRESENTATIVES = 3

#The maximal dimension to which we reduce vectors prior to computing their volume.
DIMENSION_BALL_VOLUME = 50

#The colour palette used when plotting
PLOT_COLOURS = 'bgrcmk'

#The default alpha value used for Tversky prototype analysis
DEFAULT_TVERSKY_ALPHA = 1.0

#The default beta value used for Tversky prototype analysis
DEFAULT_TVERSKY_BETA = 0.5

class Model(object):
    '''
    A model which allows geometric analysis of word embeddings in order to deduce
    semantic relations, including polysemy, context "breadth", and similarity measures.
    '''

    def __init__(self, sentences, matrix, word_to_idx, idx_to_word, wordidx_to_sentences, original_sentences):
        '''
        :brief Creates a new model using the given parameters.
        :param sentences The sentences from which the cooccurrence matrix was formed.
        :param matrix The (low-dimension) embedding matrix.
        :param word_to_idx A dictionary mapping words to their indices in the matrix.
        :param idx_to_word A dictionary mapping indices to the corresponding words.
        :param wordidx_to_sentences A dictionary mapping word indices to sentence indices.
        :param original_sentences The original sentences from the corpus.
        '''
        self.sentences = sentences
        self.matrix = matrix
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.wordidx_to_sentences = wordidx_to_sentences
        self.original_sentences = original_sentences

    def save(self, output_path):
        '''
        :brief Saves the current model into the given output path.
        :param output_path The path to which the model is saved.
        '''
        np.save("%s_matrix" % output_path, self.matrix)
        model = {"sentences"            : self.sentences,
                 "word_to_idx"          : self.word_to_idx,
                 "idx_to_word"          : self.idx_to_word,
                 "wordidx_to_sentences" : self.wordidx_to_sentences,
                 "original_sentences"   : self.original_sentences}
        with open(output_path, "wb") as output_file:
            pickle.dump(model, output_file)

    @staticmethod
    def load(model_path):
        '''
        :brief Loads the model at the given path
        :param model_path The path of the saved model.
        :return The loaded model.
        '''
        matrix = np.load("%s_matrix.npy" % model_path)
        model_dict = pickle.load(open(model_path, "r"))
        return Model(model_dict["sentences"],
                     matrix,
                     model_dict["word_to_idx"],
                     model_dict["idx_to_word"],
                     model_dict["wordidx_to_sentences"],
                     model_dict["original_sentences"])

    def matrix_size(self):
        '''
        :brief Returns the shape of the internal representation matrix.
        :return The shape of the internal representation matrix.
        '''
        return self.matrix.shape

    def get_num_sentences(self):
        '''
        :brief Returns the number of sentences in the model.
        :return The number of sentences in the model.
        '''
        return len(self.sentences)

    def get_word_vect(self, word):
        '''
        :brief Returns the vector representing the given word.
        :return The vector representing the given word.
        '''
        return [v for v in self.matrix[self.word_to_idx[word], :]]

    def get_dist_squared(self, word_a, word_b):
        '''
        :brief Returns the squared distance between the two given words.
        :param word_a The first word.
        :param word_b The second word.
        :return The squared distance between both words.
        '''
        return np.sum((np.array(self.get_word_vect(word_a)) -
                       np.array(self.get_word_vect(word_b))) ** 2)

    def get_closest_words(self,
                          word,
                          num_words=DEFAULT_NEIGHBOUR_COUNT,
                          batch_size=DEFAULT_BATCH_SIZE):
        '''
        :brief Returns the closest N "closest" words to the given word. This is done
               by batching the operations done on the matrix to avoid duplicating the
               entire matrix in memory. The results are stored in a max-heap in order
               to allow for efficient updates using constant space.
        :param word The word whose neighbours should be returned.
        :param num_words The number of words to find.
        :param batch_size The number of matrix rows processed in each "batch".
        :return The N "closest" words to the given word.
        '''
      
        if word not in self.word_to_idx:
            raise Exception("Word %s is not in model vocabulary" % word)

        #Since duplicating the entire matrix would require too much memory, we resort
        #to batching instead. We split the matrix into chunks of batch_size rows, and
        #process each of those separately (allowing memory to be reclaimed).
        word_vect = np.array(self.get_word_vect(word))
        word_vect = word_vect.reshape(1, word_vect.shape[0])
        word_idx = self.word_to_idx[word]
        
        heap = []
        num_rows = self.matrix.shape[0]
        for i in Bar("Finding Closest Words").iter(range(0, num_rows - batch_size, batch_size)):
          
            #Calculating this batch of squared L2 norms
            lower = i
            upper = min(i + batch_size, num_rows)
            m = np.linalg.norm(self.matrix[lower:upper,:] - word_vect, axis=1)

            #Making sure we ignore the self distance
            if lower <= word_idx < upper:
                m[word_idx-lower] = float("inf")

            #Partitioning to find the n-minimal items
            partition_size = min(num_words, m.shape[0])
            min_indices = np.argpartition(m, partition_size)[:partition_size]
            min_vals = [(-m[min_idx], min_idx+lower) for min_idx in min_indices]

            #Pushing the elements into the heap
            #Note that heapq is a min-heap, so we invert distances
            for v in min_vals:
                if len(heap) >= num_words:
                    heapq.heappushpop(heap, v)
                else:
                    heapq.heappush(heap, v)

        return list(reversed([(self.idx_to_word[idx], -dist) for dist, idx in sorted(heap)]))

    def get_ball(self, words, scaling_factor=1.0):
        '''
        :brief Returns a miniball containing all the given words.
        :param words The list of words to bound.
        :param scaling_factor A factor by which each word vector is multiplied.
        :return A miniball containing all the given words' embeddings.
        '''
        vectors = [list(np.array(self.get_word_vect(word)) * scaling_factor) for word in words]
        return miniball.Miniball(vectors)

    def get_sentence(self, sentence_idx):
        '''
        :brief Returns the sentence at the given index.
        :param sentence_idx The index of the sentence to retrieve.
        :return The sentence at the given index.
        '''
        return self.sentences[sentence_idx]

    def get_word_sentences(self, word):
        '''
        :brief Returns all the sentences then given word appears in.
        :param word The word to search for.
        :return All sentences in which the word ppears, or None if
                the word isn't in the corpus.
        '''
        if not word in self.word_to_idx.keys():
            return None
        word_idx = self.word_to_idx[word]
        return self.wordidx_to_sentences[word_idx]

    def idxs_to_words(self, idxs):
        '''
        :brief Converts a list of word indices to words.
        :param indx The word indices.
        :return A sentence containing the words at the given indices.
        '''
        return " ".join([self.idx_to_word[idx] for idx in idxs])

    def sanitise_context(self, context):
        '''
        :brief Sanitises the give context, ensuring it only contains known words.
        :param context A list of words to be sanitised.
        :return The sanitised list of words containing only known words.
        '''
        known_words = set(self.idx_to_word.values())
        context = [word.lower() for word in context]
        return filter(lambda word: word in known_words, context)
    
    def context_breadth(self, context):
        '''
        :brief Measures the "breadth" of the given context. This is done by bounding the
               given context in a miniball, and measuring the radius of the resulting ball.
        :param context The context to measure, as a list of words.
        :param plot Whether to plot a 2D PCA representation of the points
        :return The "breadth" of the given context.
        '''
        context = self.sanitise_context(context)
        if len(context) == 0:
            raise Exception("Context contains only unknown words")

        return np.sqrt(self.get_ball(context).squared_radius())

    def context_similarity_ball_distance(self, context_a, context_b):
        '''
        :brief Measures the "context similarity" between two given contexts, by bounding
               each context using a miniball, and measuring the distances between the centers
               of the resulting balls. If the centers are identical, this measure returns
               infinity. Otherwise, the inverse of the L2 distance is returned.
        :param context_a The first context, as a list of words.
        :param context_b The second context, as a list of words.
        :return The "context similarity", using miniball center L2-distance.
        '''

        #Sanitising the given input against the known words
        context_a = self.sanitise_context(context_a)
        context_b = self.sanitise_context(context_b)
        for context in [context_a, context_b]:
            if len(context) == 0:
                raise Exception("Context %s contains only unknown words" % " ".join(context))

        #Finding the distance between the ball centers
        ball_a = self.get_ball(context_a)
        ball_b = self.get_ball(context_b)
        c_a = np.array(ball_a.center())
        c_b = np.array(ball_b.center())
        if np.array_equal(c_a, c_b):
            return float('inf')
        return 1.0/np.linalg.norm(c_a - c_b)
    
    def context_similarity_ball_volume(self, context_a, context_b, reduce_dimension=True):
        '''
        :brief Measures the "context similarity" between two given contexts, by bounding
               each context using a miniball, and measuring the intersection volume of the
               two resulting balls. The result is then divided by the maximal ball volume
               of the two, resulting in a metric between [0,1]. Note that a value of 1 is
               returned iff the two balls are identical (both in radii and location).
        :param context_a The first context, as a list of words.
        :param context_b The second context, as a list of words.
        :param reduce_dimension Whether to reduce dimension before calculating the balls.
        :return The "context similarity", using miniball center L2-distance.
        '''

        #Sanitising the given input against the known words
        context_a = self.sanitise_context(context_a)
        context_b = self.sanitise_context(context_b)
        for context in [context_a, context_b]:
            if len(context) == 0:
                raise Exception("Context %s contains only unknown words" % " ".join(context))

        if reduce_dimension:
            #Reducing the dimension so the volume computation can be completed
            #The calculation is very sensitive w.r.t the dimension, so when dealing
            #with sufficiently large (>500) dimensions, we suggest using this option.
            vects = [self.get_word_vect(word) for word in (context_a + context_b)]
            low_dim_points = PCA(n_components=DIMENSION_BALL_VOLUME).fit_transform(vects)
            vects_a = low_dim_points[:len(context_a), :]
            vects_b = low_dim_points[len(context_a):, :]
            ball_a = miniball.Miniball([list(v) for v in vects_a])
            ball_b = miniball.Miniball([list(v) for v in vects_b])
        else:
            #Directly computing the balls for each context
            ball_a = self.get_ball(context_a)
            ball_b = self.get_ball(context_b)

        #Bounding each context
        c_a = np.array(ball_a.center())
        c_b = np.array(ball_b.center())
        n = c_a.shape[0]
        r_a = np.sqrt(ball_a.squared_radius())
        r_b = np.sqrt(ball_b.squared_radius())
        v_a = ball_volume(r_a, n)
        v_b = ball_volume(r_b, n)
        return intersection_volume(c_a, c_b, r_a, r_b) / max(v_a, v_b)

    def get_context_centers(self, word, context_size, include_self):
        '''
        :brief Returns the centers of the contexts in which the given word appears.
        ;param word The word whose contexts should be retrieved.
        :param context_size The size of the context window used.
        :param include_self Whether the include the word itself in the shapes being calculated.
        :return A tuple of (contexts, context_idx_to_sentence_idx, centers), where:
                  contexts - A list of the contexts in which the word appears
                  context_idx_to_sentence_idx - A list indexed by context index, where each value contains the
                                                sentence index from which this context was taken.
                  centers - A list of the centers of the contexts in which the word appears.
        '''
        contexts = []
        context_idx_to_sentence_idx = []
        centers = []
        word_idx = self.word_to_idx[word]
        for sentence_idx in Bar("Creating context balls").iter(self.get_word_sentences(word)):

            #Finding all occurrences of the word in the sentence
            sentence = self.get_sentence(sentence_idx)
            appearance_indices = [i for i, x in enumerate(sentence) if x == word_idx]
            if len(appearance_indices) == 0:
                continue

            #Locating the center of each context
            for find_idx in appearance_indices:
                context = sentence[max(find_idx - context_size, 0) : find_idx + context_size]
                if include_self:
                    words = set(context)
                else:
                    words = set(context) - set([word_idx])
                if len(words) == 0:
                    continue
                contexts.append(context)
                centers.append(list(np.average([np.array(self.get_word_vect(self.idx_to_word[idx])) for idx in words], axis=0)))
                context_idx_to_sentence_idx.append(sentence_idx)
        return (contexts, context_idx_to_sentence_idx, centers)

    def compute_volume_ratio(self, word_a, word_b,
                             alpha=1.0, beta=0.0,
                             context_size=DEFAULT_CONTEXT_SIZE, include_self=True):
        '''
        :brief Computes the balls corresponding to the given words, then returns a value
               corresponding to the ratio between the intersection volume and the weighted
               sum of the volumes, according to Tversky's formula.
        :param word_a The first word.
        :param word_b The second word.
        :param alpha The scaling factor to A\B
        :param beta The scaling factor for B\A
        :param context_size The size of the context used.
        :param include_self Whether the words themselves should be included when computing the context centers
        :return The Tversky weighted ratio between the volumes of the balls corresponding to both words.
        '''
        _, _, centers_a = self.get_context_centers(word_a, context_size, include_self) 
        _, _, centers_b = self.get_context_centers(word_b, context_size, include_self)
        
        ball_a = miniball.Miniball(centers_a)
        ball_b = miniball.Miniball(centers_b)

        c_a, c_b = np.array(ball_a.center()), np.array(ball_b.center())
        n = c_a.shape[0]
        r_a, r_b = np.sqrt(ball_a.squared_radius()), np.sqrt(ball_b.squared_radius())
        v_a, v_b = ball_volume(r_a, n), ball_volume(r_b, n)

        inter = intersection_volume(c_a, c_b, r_a, r_b)
        return inter/((alpha * (v_a - inter)) + inter + (beta * (v_b - inter)))


    def semantic_containment(self, word_a, word_b, context_size=DEFAULT_CONTEXT_SIZE, include_self=True):
        '''
        :brief Checks whether the first word is semantically contained in the second word. This is done by
               first creating a set of points representing each context in which each word appears. Then,
               each such set is bounded by an n-dimensional ball. Lastly, we return the ratio between the
               intersection volume of the two balls, and the volume of the ball corresponding to the first word.
        :param word_a The first word.
        :param word_b The second word.
        :param context_size The size of the context window used.
        :param include_self Whether the words themselves should be included when computing the context centers
        :return The containment ratio (between 0 and 1) of the first word in the second word. 
        '''
        return self.compute_volume_ratio(word_a, word_b, context_size=context_size, include_self=include_self)
       
    def tversky_prototype(self, word_a, word_b, context_size=DEFAULT_CONTEXT_SIZE, include_self=False):
        '''
        :brief Checks which of the two words is more likely to be a prototype.
        :param word_a The first word.
        :param word_b The second word.
        :param context_size The size of the context window used.
        :param include_self Whether the words themselves should be included when computing the context centers.
        :return word_a if the first word is more likely to be a prototype, word_b if the second is more
                likely to be the prototype, and None if the result is unknown (for example, if the balls do
                not intersect)
        '''
        v_a = self.compute_volume_ratio(word_a, word_b,
                                        DEFAULT_TVERSKY_ALPHA, 
                                        DEFAULT_TVERSKY_BETA,
                                        context_size,
                                        include_self)
        v_b = self.compute_volume_ratio(word_b, word_a,
                                        DEFAULT_TVERSKY_ALPHA, 
                                        DEFAULT_TVERSKY_BETA,
                                        context_size,
                                        include_self)
        if v_a == v_b:
            return None
        return word_b if v_a > v_b else word_a

    def plot_polysemy(self, context_centers, kmeans):
        '''
        :brief Utility function to scatter-plot the centers corresponding to context in
               which a word appears, during polysemy analysis. Called by the 'polysemy' method.
        :param context_centers The high-dimensional locations of all the context centres.
        :param kmeans The k-means model with the labels for each context.
        '''

        #Plotting the 2D PCA of the labeled points
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(context_centers)

        #Plotting the projection of all points corresponding to each cluster
        fig, ax = plt.subplots()
        for i,label in enumerate(set(kmeans.labels_)):
            colour = PLOT_COLOURS[i % len(PLOT_COLOURS)]
            data = np.array([points_2d[idx] for (idx, _) in filter(lambda (idx,l): l==label, enumerate(list(kmeans.labels_)))])
            ax.scatter(data[:,0], data[:,1], color=colour, linewidth=1.0, label="Label %d" % label)
        plt.legend(loc='best')
        plt.title("PCA Analysis of Context Ball Centers")
        plt.show()

    def polysemy(self,
                 word,
                 context_size=DEFAULT_CONTEXT_SIZE,
                 n_clusters=DEFAULT_POLYSEMY_SENSE_COUNT,
                 include_self=False,
                 num_representatives=DEFAULT_POLYSEMY_REPRESENTATIVES,
                 plot=False):
        '''
        :brief Performs polysemy analysis on the given word, attempting to find occurrences
               in which the word is used in different senses. This is done by bounding each
               context in which the word occurs by a miniball, then clustering the centers
               of the resulting balls using K-Means.
        :param word The word to analyse.
        :param context_size The size of the context around the word for which a miniball is created.
        :param n_clusters The number of clusters (and therefore senses) to search for.
        :param include_self Whether each computed miniball should also contain the word itself.
        :param num_representatives The number of representative sentences printed per label.
        :param plot Whether or not to plot a 2D PCA of the classified sentences.
        :return A list of the labeled sentences, clustered according to senses.
        '''

        #Finding the centers of the contexts in which the word appears
        contexts, context_idx_to_sentence_idx, context_centers = \
            self.get_context_centers(word, context_size, include_self) 

        #Using a K-Means model on the centers to find the clustering
        kmeans = KMeans(n_clusters=n_clusters).fit(context_centers)

        #Storing scores, labels and indices using the model
        scores = [(kmeans.score([context_centers[idx]]), kmeans.labels_[idx], idx) for idx in range(len(contexts))]

        #Sorting each label according to the score
        sorted_instances = {}
        for label in set(kmeans.labels_):

            #Sorting in descending score order
            sorted_instances[label] = []
            observed_sentences = set()
            sorted_scores = list(reversed(sorted(filter(lambda (s, l, i): label == l, scores))))
            
            #Only keeping unique sentences
            for (score,label,idx) in sorted_scores:
                sentence_idx = context_idx_to_sentence_idx[idx]
                sentence = self.original_sentences[sentence_idx]
                if not sentence in observed_sentences:
                    observed_sentences.add(sentence)
                    sorted_instances[label].append((score,label,sentence))

        #Printing the top k candidates per label
        for label in set(kmeans.labels_):
            print "Label %d" % label
            print "--------------------"
            for idx in range(num_representatives):
                _, _, sentence = sorted_instances[label][idx]
                print sentence

        #Showing a 2D PCA representation of the clusters
        if plot:
            self.plot_polysemy(context_centers, kmeans)

        #Returning the sorted instances, clustered according to labels
        return sorted_instances

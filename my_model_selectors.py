import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN

    (own comment added:)
    where L is the likelihood of the fitted model, p is the number of parameters,
    and N is the number of data points.
    """

    def calculate_bic(self, model, n):
        score = model.score(self.X, self.lengths)
        log_n = np.log(len(self.X))
        d = model.n_features
        p = n ** 2 + 2 * d * n - 1

        bic = -2.0 * score + p * log_n
        return bic

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        n_components = np.arange(self.min_n_components, self.max_n_components + 1)

        current_best_model = self.base_model(n_components[0])
        current_best_score = self.calculate_bic(current_best_model, n_components[0])

        for n in n_components[1:]:
            try:
                this_model = self.base_model(n)
                model_score = self.calculate_bic(this_model, n)

                if model_score > current_best_score:
                    current_best_model = this_model
            except:
                continue

        return current_best_model

        # Was getting an error for this code for n_params = 6
        # `rows of transmat_ must sum to 1.0 (got [ 1.  1.  1.  1.  0.  1.])`
        # Apparently, this is a problem with the library.
        # katie_tiwariForum MentorJul '17
        # @AnselmoT It could be the case for the word you are trying, there is not enough data.
        # Please use try/except block to catch these errors in for loop of number of components.


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    '''Own note:
    The Discriminative Information Criterion has been proposed as a replacement for the
    Bayesian Information Criterion. The author points out experimental evidence that
    applying the Occam's razor and Bayesian-based selection criteria to classification problems
    often does not result in selecting the best model.

    Quote: "The goal is not to select the simplest model that best explains the data,
    but to select the model that is the less likely to have generated data belonging to
    competing classification categories."

    "The model selection problem consists of selecting a single topology T as sole representative of the class C. "

    '''

    def calculate_dic(self, model, n):
        score = model.score(self.X, self.lengths)
        other_words_scores = []
        for word in self.words:
            if word != self.this_word:
                other_words_scores.append(model.score(self.hwords[word][0], self.hwords[word][1]))

        return score - np.mean(other_words_scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores

        n_components = np.arange(self.min_n_components, self.max_n_components + 1)

        current_best_model = self.base_model(n_components[0])
        current_best_score = self.calculate_dic(current_best_model, n_components[0])

        for n in n_components[1:]:
            try:
                this_model = self.base_model(n)
                model_score = self.calculate_dic(this_model, n)

                if model_score > current_best_score:
                    current_best_model = this_model
            except:
                continue

        return current_best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    def get_training_and_test(self):
        scores = []
        split_method = KFold(n_splits=2)

        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            tr_X, tr_lengths = combine_sequences(cv_train_idx, self.sequences)
            ts_X, ts_lengths = combine_sequences(cv_test_idx, self.sequences)

        return [(tr_X, tr_lengths), (ts_X, ts_lengths)]

    def calculate_cv_avg(self, model):
        training_set, test_set = self.get_training_and_test()
        ts_X, ts_lengths = test_set
        scores = []
        scores.append(model.score(ts_X, ts_lengths))
        return np.mean(scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        n_components = np.arange(self.min_n_components, self.max_n_components + 1)

        training_set, test_set = self.get_training_and_test()

        tr_X, tr_lengths = training_set
        current_best_model = self.base_model(n_components[0]).fit(tr_X, tr_lengths)

        current_best_score = self.calculate_cv_avg(current_best_model)

        for n in n_components[1:]:
            try:
                this_model = self.base_model(n).fit(tr_X, tr_lengths)
                model_score = self.calculate_cv_avg(this_model)

                if model_score > current_best_score:
                    current_best_model = this_model
            except:
                continue

        return current_best_model

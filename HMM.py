

import random
import argparse
import codecs
import os
import numpy as np

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        with open(basename+'.trans', 'r') as file:
            lines = file.readlines()
        for line in lines:
            current_state, next_state, prob_val = line.split()
            if(current_state not in self.transitions):
                self.transitions[current_state] = {}
            self.transitions[current_state][next_state] = float(prob_val)

        with open(basename+'.emit', 'r') as file:
            lines = file.readlines()
        for line in lines:
            current_state, next_state, prob_val = line.split()
            if (current_state not in self.emissions):
                self.emissions[current_state] = {}
            self.emissions[current_state][next_state] = float(prob_val)


    ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        observation = ["#"]
        for i in range(n-1):
            last_state = observation[-1]
            next_states = list(self.transitions[last_state].keys())
            probabilities = [self.transitions[last_state][state] for state in next_states]
            # print(next_states)
            # print(probabilities)
            next_state = np.random.choice(next_states, p=probabilities)
            # print(next_state)
            observation.append(next_state)

        return observation

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """



hmm_obj = HMM()
hmm_obj.load('two_english')
print(hmm_obj.transitions)
print(hmm_obj.emissions)
print(hmm_obj.generate(20))

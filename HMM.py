

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
        last_state = "#"
        states = list(self.transitions[last_state].keys())
        # print(states)
        state = random.choices(states, weights=[self.transitions[last_state][s] for s in states])[0]
        stateseq = [state]

        for _ in range(n - 1):
            w = [self.transitions[state][s] for s in states]
            state = random.choices(states, weights=w)[0]
            stateseq.append(state)

        outputseq = []
        for i in stateseq:
            w = [self.emissions[i][j] for j in self.emissions[i]]
            states = list(self.emissions[i].keys())
            state = random.choices(states, weights=w)[0]
            outputseq.append(state)

        return Observation(stateseq, outputseq)

    def forward(self, observation):
        # possible_states = list(self.transitions.keys())
        # observation_len = len(observation)
        #
        # matrix = np.zeros((len(possible_states), observation_len))
        # for s in possible_states:
        #         matrix[possible_states.index(s), 0] = self.transitions[possible_states[0]][s] * self.emissions[s][observation[0]]
        #
        # for i in range(1, observation_len):
        #     for s in possible_states:
        #         total = 0
        #         for s2 in possible_states:
        #             total += matrix[possible_states.index(s2), i - 1] * self.transitions[s2][s] * self.emissions[s][observation[i]]
        #         matrix[possible_states.index(s), i] = total
        #
        # return matrix
        outputseq = observation.outputseq
        states = list(self.transitions["#"].keys())
        total_states = len(states)
        total_observations = len(observation.stateseq)
        mat = np.zeros((total_states, total_observations))
        mat[0][0] = 1.0;
        for i in range(1, total_observations):
            for s in states:
                total = 0
                for s_prev in states:
                    total += mat[states.index(s_prev)][i-1] * self.transitions[s_prev][s] * self.emissions[s_prev][outputseq[i]]
            mat[states.index(s_prev)][i] = total

        return mat

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """



hmm_obj = HMM()
hmm_obj.load('two_english')
# print(hmm_obj.transitions)
# print(hmm_obj.emissions)
observations = hmm_obj.generate(20)
# print(observations)
matrix = hmm_obj.forward(observations)
print(matrix)

import random
import argparse
import codecs
import os
import numpy as np
import sys

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
        outputseq = observation.outputseq
        states = list(self.transitions.keys())
        total_states = len(states)
        total_observations = len(observation.outputseq)
        mat = np.zeros((total_states, total_observations))

        # '#' state should be at index 0
        for i in range(total_states):
            if states[i]=='#':
                states[i] = states[0]
                states[0] = '#'
                break

        # Fill out first column
        for s in states:
            if s != '#' and outputseq[0] in self.emissions[s]:
                mat[states.index(s)][0] = self.transitions['#'][s] * self.emissions[s][outputseq[0]]

        # Fill out rest of the columns
        for i in range(1, total_observations):
            for s in states:
                if s != '#':
                    total = 0.0
                    for s2 in states:
                        if s2 != '#' and outputseq[i] in self.emissions[s]:
                            total += mat[states.index(s2)][i-1] * self.transitions[s][s2] * self.emissions[s][outputseq[i]]
                    mat[states.index(s)][i] = total

        final_output = []
        max_indices = np.argmax(mat, axis=0)
        # print(max_indices)
        # print(mat)
        for i in max_indices:
            final_output.append(states[i]);
        return final_output

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.
    """given an observation,
            find and return the state sequence that generated
            the output sequence, using the Viterbi algorithm.
            """
    def viterbi(self, observation):
        outputseq = observation.outputseq
        states = list(self.transitions.keys())
        total_states = len(states)
        total_observations = len(observation.outputseq)
        mat = np.zeros((total_states, total_observations))
        backpointer = np.zeros((total_states, total_observations))
        # '#' state should be at index 0
        for i in range(total_states):
            if states[i] == '#':
                states[i] = states[0]
                states[0] = '#'
                break

        # Fill out first column
        for s in states:
            if s != '#' and outputseq[0] in self.emissions[s]:
                mat[states.index(s)][0] = self.transitions['#'][s] * self.emissions[s][outputseq[0]]

        # Fill out rest of the columns
        for i in range(1, total_observations):
            for s in states:
                if s != '#':
                    # total = 0.0
                    max_prob = 0.0
                    max_index = 0
                    for s2 in states:
                        if s2 != '#' and outputseq[i] in self.emissions[s]:
                            val = mat[states.index(s2)][i - 1] * self.transitions[s][s2] * self.emissions[s][outputseq[i]]
                            if max_prob < val:
                                max_prob = val
                                max_index = states.index(s2)

                    mat[states.index(s)][i] = max_prob
                    backpointer[states.index(s)][i] = max_index

        final_output = []
        max_indices = np.argmax(mat, axis=0)
        # print(max_indices)
        # print(mat)
        for i in max_indices:
            final_output.append(states[i]);
        return final_output


def read_obs_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    observations = []
    for line in lines:
        if line != '\n':
            observations.append(Observation([],line.split()))
    return observations

if __name__== '__main__':
    if len(sys.argv) < 3:
        print("Usage: hmm.py filename {--generate --forward --viterbi} n")
        sys.exit(-1)
    else:
        hmm_obj = HMM()
        parser = argparse.ArgumentParser(description="Usage: hmm.py filename {--generate --forward --viterbi}")
        parser.add_argument('filename', type=str, help='Input file path')
        parser.add_argument('--generate', type=int, help='Observation length')
        parser.add_argument('--forward', type=str, help='Observation file')
        parser.add_argument('--viterbi')
        args = parser.parse_args()
        generate = args.generate
        forward = args.forward
        viterbi = args.viterbi
        filename = args.filename
        if generate:
            hmm_obj.load(filename)
            print(hmm_obj.generate(generate))
        if forward:
            hmm_obj.load(filename)
            observations = read_obs_file(forward)
            for observation in observations:
                output = hmm_obj.forward(observation)
                print(output)
                print(observation.outputseq)
        if viterbi:
            hmm_obj.load(filename)
            observations = read_obs_file(viterbi)
            for observation in observations:
                output = hmm_obj.viterbi(observation)
                print(output)
                print(observation.outputseq)


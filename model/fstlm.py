#!/usr/bin/env python

# Copyright 2018 Mitsubishi Electric Research Laboratories (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import math
import numpy as np
import itertools
from collections import defaultdict, deque
from toposort import toposort_flatten
from picklable_itertools.extras import equizip
try:
    import fst
except ImportError:
    print("No PyFST module, trying to work without it. If you want to run the "
          "language model, please install openfst and PyFST")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

EPSILON = 0
MAX_STATES = 7
NOT_STATE = -1


class FST(object):

    """Picklable wrapper around FST."""
    def __init__(self, path):
        self.path = path
        self.fst = fst.read(self.path)
        self.isyms = dict(self.fst.isyms.items())

    def __getitem__(self, state):
        """Returns all arcs of the state i"""
        return self.fst[state]

    def combine_weights(self, *args):
        # Protection from underflow when -x is too small
        m = max(args)
        return m - math.log(sum(math.exp(m - x) for x in args if x is not None))

    def get_arcs(self, state, character):
        return [(state, arc.nextstate, arc.ilabel, float(arc.weight))
                for arc in self[state] if arc.ilabel == character]

    def transition(self, states, character):
        arcs = list(itertools.chain(
            *[self.get_arcs(state, character) for state in states]))
        next_states = {}
        for next_state in {arc[1] for arc in arcs}:
            next_states[next_state] = self.combine_weights(
                *[states[arc[0]] + arc[3] for arc in arcs
                  if arc[1] == next_state])
        return next_states

    def expand(self, states):
        seen = set()
        depends = defaultdict(list)
        queue = deque()
        for state in states:
            queue.append(state)
            seen.add(state)
        while len(queue):
            state = queue.popleft()
            for arc in self.get_arcs(state, EPSILON):
                depends[arc[1]].append((arc[0], arc[3]))
                if arc[1] in seen:
                    continue
                queue.append(arc[1])
                seen.add(arc[1])

        depends_for_toposort = {key: {state for state, weight in value}
                                for key, value in depends.items()}
        order = toposort_flatten(depends_for_toposort)
        
        next_states = states
        for next_state in order:
            next_states[next_state] = self.combine_weights(
                *([next_states.get(next_state)] +
                  [next_states[prev_state] + weight
                   for prev_state, weight in depends[next_state]]))

        return next_states

    def explain(self, input_):
        input_ = list(input_)
        states = {self.fst.start: 0}
        print("Initial states: {}".format(states))
        states = self.expand(states)
        print("Expanded states: {}".format(states))

        for char, ilabel in zip(input_, [self.isyms[char] for char in input_]):
            states = self.transition(states, ilabel)
            print("{} consumed: {}".format(char, states))
            states = self.expand(states)
            print("Expanded states: {}".format(states))

        result = None
        for state, weight in states.items():
            if np.isfinite(weight + float(self.fst[state].final)):
                print("Finite state {} with path weight {} and its own weight {}".format(
                    state, weight, self.fst[state].final))
                result = self.combine_weights(
                    result, weight + float(self.fst[state].final))

        print("Total weight: {}".format(result))
        return result


class FSTTransitionOp():
    """Performs transition in an FST.

    Given a state and an input symbol (character) returns the next state.

    Parameters
    ----------
    fst : FST instance
    remap_table : dict
        Maps neutral network characters to FST characters.

    """

    def __init__(self, fst, remap_table):
        self.fst = fst
        self.remap_table = remap_table

    def pad(self, arr, value):
        return np.pad(arr, (0, MAX_STATES - len(arr)),
                         mode='constant', constant_values=value)
    
    def __call__(self, all_states, all_weights, all_inputs):       
        # Each row of all_states contains a set of states
        # padded with NOT_STATE.

        all_next_states = []
        all_next_weights = []
        for states, weights, input_ in equizip(all_states, all_weights, all_inputs):
            states_dict = dict(zip(states, weights))
            if NOT_STATE in states_dict:
                del states_dict[NOT_STATE]
            next_states_dict = self.fst.transition(
                states_dict, self.remap_table[input_])
            next_states_dict = self.fst.expand(next_states_dict)
            if next_states_dict:
                next_states, next_weights = zip(*next_states_dict.items())
            else:
                # No adequate state when no arc exists for now
                next_states, next_weights = [], []
            all_next_states.append(self.pad(next_states, NOT_STATE))
            all_next_weights.append(self.pad(next_weights, 0))

        return np.array(all_next_states, dtype='int64'), np.array(all_next_weights)


class FSTCostsOp():
    """Returns transition costs for all possible input symbols.

    Parameters
    ----------
    fst : FST instance
    remap_table : dict
        Maps neutral network characters to FST characters.
    no_transition_cost : float
        Cost of going to the start state when no arc for an input
        symbol is available.

    Notes
    -----
    It is assumed that neural network characters start from zero.

    """

    def __init__(self, fst, remap_table, no_transition_cost):
        self.fst = fst
        self.remap_table = remap_table
        self.no_transition_cost = no_transition_cost

    def __call__(self, all_states, all_weights):

        all_costs = []
        for states, weights in zip(all_states, all_weights):
            states_dict = dict(zip(states, weights))
            if NOT_STATE in states_dict:
                del states_dict[NOT_STATE]
            costs = np.ones(len(self.remap_table), dtype=np.float32) * self.no_transition_cost
            if states_dict:
                total_weight = self.fst.combine_weights(*states_dict.values())
                for nn_character, fst_character in self.remap_table.items():
                    next_states_dict = self.fst.transition(states_dict, fst_character)
                    next_states_dict = self.fst.expand(next_states_dict)
                    if next_states_dict:
                        next_total_weight = self.fst.combine_weights(*next_states_dict.values())
                        costs[nn_character-1] = next_total_weight - total_weight
            all_costs.append(costs)

        return np.array(all_costs)
        

class FSTTransition():
    def __init__(self, fst, remap_table, no_transition_cost):
        """Wrap FST in a recurrent brick.

        Parameters
        ----------
        fst : FST instance
        remap_table : dict
            Maps neutral network characters to FST characters.
        no_transition_cost : float
            Cost of going to the start state when no arc for an input
            symbol is available.

        """
        super(FSTTransition, self).__init__()
        self.fst = fst
        self.transition = FSTTransitionOp(fst, remap_table)
        self.probability_computer = FSTCostsOp(
            fst, remap_table, no_transition_cost)

        self.out_dim = len(remap_table)

    def predict(self, inputs, states, weights, add):
        new_states, new_weights = self.transition(states, weights, inputs)
        new_add = self.probability_computer(new_states, new_weights)
        return new_states, new_weights, new_add

    def initial_states(self):
        states_dict = self.fst.expand({self.fst.fst.start: 0.0})
        states = [states_dict.keys()]
        weights = [states_dict.values()]
        add = self.probability_computer(states, weights)
        return states, weights, add

    def get_dim(self, name):
        if name == 'states' or name == 'weights':
            return MAX_STATES
        if name == 'add':
            return self.out_dim
        if name == 'inputs':
            return 0
        return super(FSTTransition, self).get_dim(name)


class NgramFstLM():
    def __init__(self, path, nn_char_map_file, no_transition_cost=1e12):
        super(NgramFstLM, self).__init__()
        self.fst = FST(path)
        fst_char_map = dict(self.fst.fst.isyms.items())
        del fst_char_map['<eps>']
        
        with open(nn_char_map_file, 'r', encoding='utf-8') as f:
            tmp_chars = f.readlines()
        self.nn_char_map = {x.strip().split(' ')[0]: int(x.strip().split(' ')[1]) for x in tmp_chars}
    
        if not len(fst_char_map) == len(self.nn_char_map):
            raise ValueError()
        remap_table = {self.nn_char_map[character]: fst_code
                       for character, fst_code in fst_char_map.items()}
        ##print(nn_char_map, fst_char_map, remap_table)
        self.transition = FSTTransition(self.fst, remap_table, no_transition_cost)
        
    def initial_states(self):
        states, weights, add = self.transition.initial_states()
        return states, weights, add
        
    def predict(self, state, x):
        # update state with input label x
        score = np.zeros(len(self.nn_char_map), dtype=np.float32)
        if state is None:  # make initial states and log-prob vectors
            states, weights, add = self.initial_states()
            ##score[0] = float('-inf')
            ##score[1:] = -add[0][:-1] 
            final_score = np.array(score, dtype=np.float32)    
            state = {'states': states, 'weights': weights, 'add': add}
            return state, torch.from_numpy(final_score).unsqueeze(0)
        else:
            states = state['states'] 
            weights = state['weights']    
            add = state['add']
            inputs = [int(x)]
            new_states, new_weights, new_add = self.transition.predict(inputs, states, weights, add)
            score[0] = float(-20)
            score[1:] = -new_add[0][:-1]        
            if np.argmax(score) == len(self.nn_char_map) - 1:
                print('space is found, input is {}'.format(int(x)))
                inputs = [len(self.nn_char_map) - 1]
                new_states, new_weights, new_add = self.transition.predict(inputs, new_states, new_weights, new_add) 
                score[1:] = -new_add[0][:-1]
                final_score = np.array(score, dtype=np.float32)              
            else:
                final_score = np.array(score, dtype=np.float32)    
            state = {'states': new_states, 'weights': new_weights, 'add': new_add}
            return state, F.log_softmax(torch.from_numpy(final_score).unsqueeze(0), dim=1)
        
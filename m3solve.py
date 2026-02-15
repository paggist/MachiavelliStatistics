#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:46:50 2024

@author: spaggi
"""

import sys
import os
import itertools
from collections import defaultdict, Counter
import re
import random
import numpy as np
import cvxpy as cp
import time
import csv

NUMBERS = '123456789tjqk'  # Allowed alternate inputs: a->1, 0->t, 10->t, emoji
SUITS = 'scdh'
CARDS = [
    n+s
    for n, s in itertools.product(NUMBERS, SUITS)
]

def problem_solve_suppress_stdout(problem, **kwargs):
    '''Suppress diagnostic messages printed by CVXPY C-libraries.'''
    dup_success = False
    stdout_cp = None
    devnull = None
    try:
        stdout_cp = os.dup(1)
        devnull = open(os.devnull, 'w')
        os.dup2(devnull.fileno(), 1)
        dup_success = True
        return problem.solve(**kwargs)
    except:
        if dup_success:
            raise
        else:
            # Give up and run without stdout suppression
            return problem.solve(**kwargs)
    finally:
        if devnull is not None:
            devnull.close()
        if stdout_cp is not None:
            os.dup2(stdout_cp, 1)
            os.close(stdout_cp)


def sort_key(s):
    s = s.lower()
    s = s.replace('a', '1').replace('q', 'i')
    s = s.replace('j', 'h').replace('s', 'b').replace('t', 'b')
    # Sort sequences after sets
    if ',' in s and not all(ss[:1] == s[:1] for ss in s.split(',')):
        s = 'z_' + s
    return s

def sort_key_k(s):
    return sort_key(s).replace('1', 'l')
def sorted_cards(cards):
    '''Returns the cards sorted nicely.'''
    cards = list(cards)
    if (any('k' in card for card in cards)
            and not any('2' in card for card in cards)):
        cards.sort(key=sort_key_k)
    else:
        cards.sort(key=sort_key)
    return cards

def cards_to_codes(cards):
    '''Converts a list of card names to (suit_index, rank_index) tuples.'''
    return np.array([
        (SUITS.index(s), NUMBERS.index(n))
        for n, s in cards
    ])

def codes_to_cards(codes):
    '''Converts a list of (suit_index, rank_index) tuples to card names.'''
    return [
        f'{NUMBERS[j]}{SUITS[i]}'
        for i, j in codes
    ]

def codes_to_str(codes):
    '''Converts a list of card codes (index tuples) to a sorted,
    comma-separated, printable string.'''
    return cards_to_str(codes_to_cards(codes))

def cards_to_str(cards):
    '''Converts a list of card names to a sorted, comma-separated, printable
    string.'''
    return ','.join(sorted_cards(cards))

#%%
"""
cards is table + hand
optional cards is hand
cards structure [(suits,number),(suits,number),...]
"""
def solve(cards, optional_cards=()):
    # Count cards
    cards = list(cards)
    card_codes = cards_to_codes(cards)
    optional_cards = list(optional_cards)
    assert min((Counter(cards)-Counter(optional_cards)).values(),
               default=0) >= 0, (
        'optional_cards must be a subset of cards')
    card_counts = Counter(cards)
    optional_counts = Counter(optional_cards)

    # Make grid of cards to find sets and sequences
    seq_arr = np.zeros((len(SUITS), len(NUMBERS+NUMBERS[0:1])), dtype=int)
    # print(card_codes)
    for i, j in card_codes:
        seq_arr[i, j] += 1
    
    seq_arr[:, -1] = seq_arr[:, 0]
    seq_nonzero = (seq_arr > 0).astype(int)

    # Extract sets, sequences, and find disconnected cards
    find_seq = np.concatenate([
            seq_nonzero[:, 0:1] + seq_nonzero[:, 1:2],
            seq_nonzero[:, :-2] + seq_nonzero[:, 1:-1] + seq_nonzero[:, 2:],
        ], axis=1)
    alone = find_seq * seq_nonzero[:, :-1] == 1
    alone[:, 0] &= seq_nonzero[:, -2] == 0
    find_set = np.sum(seq_nonzero, axis=0)[:-1]
    alone[:, find_set >= 3] = False

    # Build a list of all possible sequences using the current cards
    seq_list = []  # Only the max length sequences, no overlap
    seq_list_full = []
    for i in range(len(SUITS)):
        j = 0
        while j < len(NUMBERS):
            if find_seq[i, j] >= 3:
                jj = j+1
                while jj < len(NUMBERS) and find_seq[i, jj] >= 3:
                    jj += 1
                seq = (np.array([np.full(jj-j+2, i), range(j-1, jj+1)]).T
                       % len(NUMBERS))
                seq_list.append(seq[:13])
                seq_list_full.append(seq[:13])
                for l in reversed(range(3, min(13, len(seq)))):
                    for ll in range(len(seq)-l+1):
                        seq_list_full.append(seq[ll:l+ll])
                j = jj - 1
            j += 1

    # Build a list of all possible sets using the current cards
    set_list = []  # Only the max length sets, no overlap
    set_list_full = []
    arange4 = np.arange(4)
    for j in range(len(NUMBERS)):
        if find_set[j] < 3:
            continue
        group = np.array([
                arange4[seq_nonzero[:, j] > 0],
                np.full(find_set[j], j)
            ]).T
        set_list.append(group)
        set_list_full.append(group)
        if len(group) == 4:
            set_list_full.append([group[0], group[1], group[2]])
            set_list_full.append([group[1], group[2], group[3]])
            set_list_full.append([group[0], group[2], group[3]])
            set_list_full.append([group[0], group[1], group[3]])

    # Map each card to the list of groups it could participate in
    card_options = defaultdict(list)
    for seq in itertools.chain(set_list_full, seq_list_full):
        for i, j in seq:
            card_options[i, j].append(frozenset((i, j) for i, j in seq))

    def clean_solution(sol):
        '''Make pretty strings to display a solution'''
        if sol is None:
            return 'no solution', ''
        selected = sorted((
                name
                for name, cnt in sol.items()
                for _ in range(cnt)
            ),
            key=sort_key
        )
        
        remaining_cnt = (
            Counter(cards)
            - Counter(c
                      for name, cnt in sol.items()
                      for name2 in [name] * cnt
                      for c in name.replace('_', '').split(','))
        )
        # print(Counter(optional_cards) - remaining_cnt)
        hand = ','.join(sorted_cards(remaining_cnt.elements()))
        using_cnt = Counter(optional_cards) - remaining_cnt
        hand_use = ','.join(sorted_cards(using_cnt.elements()))
        out = f"({') ('.join(selected)})"
        # for card in using_cnt.keys():
        #     if using_cnt[card] <= 1:
        #         i = out.find(card)
        #         if i >= 0:
        #             out = f'{out[:i]}[ {card} ]{out[i+2:]}'
        #     else:
        #         out = out.replace(card, f'[ {card} ]')
        # if not sol:
        #     out = 'no cards'
        # else:
        #     out = self.pretty_cards(out)
        # if not hand:
        #     hand = 'WIN'
        # else:
        #     hand = self.pretty_cards(hand)
        # if hand_use:
        #     hand = f'{hand} -- use: {self.pretty_cards(hand_use)}'
        return out, hand

    def solution_size(sol):
        '''The number of cards used by the solution.'''
        if sol is None:
            return 0
        return sum(
            len(name.split(',')) * cnt
            for name, cnt in sol.items()
        )

    def print_sol(sol):
        '''Print the solution.'''
        extra = len(card_codes)-solution_size(sol)
        table, hand = clean_solution(sol)
        # if optional_cards:
        #     msg = f'--- {extra} left ---'
        #     # if self.color:
        #     #     import termcolor
        #     #     if extra >= len(optional_cards):
        #     #         self.print(termcolor.colored(msg, 'yellow'))
        #     #     elif extra > 0:
        #     #         self.print(termcolor.colored(msg, 'green'))
        #     #     else:
        #     #         self.print(termcolor.colored(msg, 'green',
        #     #                                      attrs=['reverse']))
        #     # else:
        #     self.print(msg)
        #     self.print(f'hand: {hand}')
        # self.print(f'table: {table}')
        return table, hand

    # Check for an empty solution (otherwise causes the solver to fail)
    if len(seq_list_full) + len(set_list_full) == 0:
        # No solutions
        if cards == optional_cards:  # Set comparison
            sol = {}
        else:
            sol = None
        # print_sol(sol)
        return sol

    # Encode as an integer program
    possible_groups = [
        codes_to_cards(group)
        for group in itertools.chain(set_list_full, seq_list_full)
    ]
    card_idx = {card: i for i, card in enumerate(card_counts)}
    card_mat = np.zeros((len(possible_groups), len(card_counts)),
                        dtype=bool)
    for i, group in enumerate(possible_groups):
        for card in group:
            card_mat[i, card_idx[card]] = True
    card_min = np.zeros(len(card_counts), dtype=int)
    card_max = np.zeros(len(card_counts), dtype=int)
    for card in card_counts.keys():
        max_count = card_counts[card]
        min_count = max_count - optional_counts[card]
        card_min[card_idx[card]] = min_count
        card_max[card_idx[card]] = max_count
    group_max = np.array([
        1 + all(card_counts[card] >= 2 for card in group)
        for group in possible_groups
    ])

    # Setup CVXPY
    x = cp.Variable(len(possible_groups), integer=True)
    constraints = [
        card_mat.T @ x >= card_min,
        card_mat.T @ x <= card_max,
        x <= group_max,
        x >= 0
    ]

    # obj = cp.Maximize(sum(card_mat.T @ x) - sum(x) / 1024)
   # removed penalty for big solutions, since it had difficulty recognizing solution
   # e.g. table=['8d', '9d', 'td', '7d', 'jd', 'qd', 'kd', 'td', 'qs', 'qc', 'qh', '5c', '5d', '5h', '5s', '3c', '4c', '6d', '5d', '7d', '8d', '9h', 'th', 'jh', '8h', 'qh', '8s', '7c', '7h', '5c', '5h', '1h', 'kh', '8h', 'jh', 'qd', '2s', '2d', '2h', '1d', '1s', '1h', 'js', 'jd', '3d', '4d', '5s', '2c', '3h', 'qc', 'kc', '4h', '1s', '3s', '6h', '6s', '2s', '2d', 'kd', '3h', '4s', '6s', '9d', '9c', '9h', '6c', '9s', '4s', '7c', '3d', '3c', '4d', '2c']

    obj = cp.Maximize(sum(card_mat.T @ x)) 
    problem = cp.Problem(obj, constraints)

    # Solve
    cost = problem_solve_suppress_stdout(problem, verbose=False)
    if isinstance(cost, str) or x.value is None:
        print(f'solver failed: {problem.status} ({cost})',
                       RuntimeError)
        return -1
    if x.value is None:
        sol = None
        # print_sol(sol)
        return -1
    else:
        sol = {
            cards_to_str(group): round(val)
            for group, val in zip(possible_groups, x.value)
            if round(val) > 0
        }

    # Verify valid solution
    sol_card_counts = sum((
            Counter(group_str.split(','))
            for group_str, cnt in sol.items()
            for _ in range(cnt)
        ),
        start=Counter()
    )
    remaining_counts = Counter(card_counts)
    remaining_counts.subtract(sol_card_counts)
    used_hand_counts = Counter(optional_counts)
    used_hand_counts.subtract(remaining_counts)
    if (any(val < 0 for val in remaining_counts.values())
            or any(val < 0 for val in used_hand_counts.values())):
        # Invalid solution
        sol = None
        return -1

    # Print and return in a sorted, easy-to-use form
    # table,hand=print_sol(sol)
    # selected = sorted((
    #         name
    #         for name, cnt in sol.items()
    #         for _ in range(cnt)
    #     ),
    #     key=sort_key
    # )
    # selected = [
    #     name.split(',')
    #     for name in sorted((
    #             name
    #             for name, cnt in sol.items()
    #             for _ in range(cnt)
    #         ),
    #         key=sort_key
    #     )
    # ]
    # selected = [
    #     part
    #     for name in sorted(
    #         (name for name, cnt in sol.items() for _ in range(cnt)),
    #         key=sort_key
    #     )
    #     for part in name.split(',')
    # ]

    if sol==None:
        # print(cards)
        # print(optional_cards)
        # sys.exit()
        return -1
    remaining_cnt = (
        Counter(cards)
        - Counter(c
                  for name, cnt in sol.items()
                  for name2 in [name] * cnt
                  for c in name.replace('_', '').split(','))
    )
    # print(Counter(optional_cards) - remaining_cnt)
    # hand = ','.join(sorted_cards(remaining_cnt.elements()))
    using_cnt = Counter(optional_cards) - remaining_cnt
    # hand_use = ','.join(sorted_cards(using_cnt.elements()))
    selected = [
        part
        for name in sorted(
            (name for name, cnt in using_cnt.items() for _ in range(cnt)),
            key=sort_key
        )
        for part in name.split(',')
    ]
    # clean_solution(sol)
    if selected==Counter():
        return []
    return selected

#%%


data_to_write_to_file=[]



#create shuffled deck

def game(nstartcards=5,ndecks=2,nplayers=2):
    global data_to_write_to_file
    # nstartcards=5
    # ndecks=2
    # nplayers=3
    DECK=CARDS*ndecks
    random.shuffle(DECK)
    rounds=0
    # print(solve(DECK[:20],DECK[:20]))

    TABLE=[]
    HANDS=[]
    for i in range(nplayers):
        HANDS.append(DECK[:nstartcards])
        DECK=DECK[nstartcards:]
    gameLoop=True
    while gameLoop:
        for nturn in range(nplayers):
            data=[len(h) for h in HANDS]
            data.append(len(TABLE))
            data.append(rounds)
            data_to_write_to_file.append(data)
            print("####"+"#"*nturn)
            print("Turn N.",nturn)
            print("Total Turn N.",rounds)
            print("Hand of current player ",nturn)
            print(HANDS[nturn])
            solution=solve(HANDS[nturn]+TABLE,HANDS[nturn])
            print("Solution of table and hand")
            print(solution)
            if solution==-1:
                return -99,rounds
                gameLoop=False
                break
            if len(solution)!=0:
                print("Not an empty solution")
                calledHand,calledTable=False,False
                for item in solution:
                    print("removed card " + str(item))
                    HANDS[nturn].remove(item)
                    TABLE.append(item)
                # if calledHand==False and calledTable==False:
                #     if len(DECK)==0:
                #         print("No cards left")
                #         gameLoop=False
                #         break
                    # HANDS[nturn].append(DECK[:1][0])
                    # DECK=DECK[1:]
            else:
                
                if len(DECK)==0:
                    print("No cards left player ",nturn)
                    return -1,rounds
                    gameLoop=False
                    break
                print("Pick a card")
                HANDS[nturn].append(DECK[:1][0])
                print("Player "+str(nturn)+" picks "+str(DECK[:1][0]))
                DECK=DECK[1:]
                
            print("table at total turn "+str(rounds)+" player "+str(nturn)+" turn")
            print(TABLE)
            if len(HANDS[nturn])==0:
                data=[len(h) for h in HANDS]
                data.append(len(TABLE))
                data.append(rounds+1)
                data_to_write_to_file.append(data)
                print("Player "+ str(nturn)+" WINS!")
                return nturn,rounds
                gameLoop=False
                break
            rounds+=1
            

def write_csv(filename, data, append=False):
    """
    Writes or appends data to a CSV file.

    Parameters:
    - filename: The name of the CSV file.
    - data: A list of rows, where each row is a list of values.
    - append: If True, appends to the file; otherwise, overwrites it.
    """
    mode = 'a' if append else 'w'
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode, newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Write header only if the file doesn't exist and not appending
        if not file_exists and not append:
            writer.writerow(['Column1', 'Column2', 'Column3'])  # Example header
        # Write the data
        writer.writerows(data)

filename="machiagame"+str(time.time())+".csv"


starttime=time.time()
nplayers=2
nstartcards=5
ndecks=2
result=[]
resultrounds=0



for i in range(nplayers+2):
    result.append(0)
for i in range(20000):
    
    if len(data_to_write_to_file)>1000:
        write_csv(filename, data_to_write_to_file,append=True)
        data_to_write_to_file=[]
    g,rounds=game(nstartcards,ndecks,nplayers)
    if g!=-1 and g!=-99:
        result[g]+=1
    elif g==-99:
        result[nplayers+1]+=1
    else:
        result[nplayers]+=1
    if resultrounds==0:
        resultrounds=rounds
        continue
    resultrounds=(resultrounds+rounds)/2
print(result)
print(resultrounds)
endtime=time.time()
elapsed_time=endtime-starttime
print(f"Elapsed time: {elapsed_time:.4f} seconds")

# import solverAltered as sa

# print("And now the other guys!")
# s=sa.Solver()
# s.solve(DECK[:20],DECK[:20])

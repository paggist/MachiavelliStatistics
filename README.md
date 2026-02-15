# MachiavelliStatistics
This project uses a [convex optimization solver](https://en.wikipedia.org/wiki/Convex_optimization) for the Italian card game [Machiavelli](https://en.wikipedia.org/wiki/Machiavelli_(Italian_card_game))
to perform statistical analysis on game outcomes based on strategies.

The Python codes are based on the [Machiavelli command-line game by Casey Duckering](https://github.com/cduck/machiavelli). Here, given starting hands, number of decks, and number of players,
the game is then played between computers. The goal is to simulate e.g. a large amount of games to gain data on how well certain starting hands, e.g. only spades in hand, players
starting with different numbers of cards, players hiding cards, etc... compete against others.

It also produces a the card distribution at each turn, showing how long a game on average lasts, what is a normal amount of cards to expect on each turn, etc.

# Requirements

- Python 3
- Numpy
- [cvxpy](https://www.cvxpy.org/)
- 

## Installation

m3solve.py is found under /src. With this file one can simulate many iterations of the entire game. The output is then
saved on a .csv file.

With create_card_dist_matrix_2players_table.py one can create the distribution plots.

## Examples

These plots show the card distribution on each round for a player in a 4-player game with two 52 French cards decks.
Starting hand are 5 cards.
The distribution was created from 10k simulated games.
The red step-line shows the maximum attainable cards for that round. The first rounds players are mainly picking cards from
the deck, and the most likely cards to shed are three-of a kinds. In the middle section of the game, players are often just shedding
one card per 4 rounds. 
The upper plot shows the absolute number of occurences, whereas the lower plot displays the number normalized to each round.
![4-player distribution](images/machia_4player_stats.png)
From the picture one can read that a 4-player game ends after between 70-90 rounds. The number of cards per hand peaks at 17 cards per hand, and
then declines linearly after the 40th round. Very few games reach the 100th round, from where players usually just have 1 card per hand which they
cannot discard, so they keep taking card, and shedding it in the next 4 rounds.

## License

[MIT](https://choosealicense.com/licenses/mit/)

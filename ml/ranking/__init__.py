"""Ranking (M4): the home feed's candidates and the features of the LightGBM ranker.

`candidates` replays the api's recall and fusion offline; `features` computes the ranker's
features from the candidates, the history and the item tables the bundle ships. The model
server computes the same features online and checks itself against the bundle's golden queries.
"""

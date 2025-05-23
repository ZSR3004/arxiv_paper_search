"""
File: ArxivQuery.py
Author: Ziyad Rahman
Date: 2025-22-05

Brief: This file contains the find_similar_papers function which is used to find similar research papers from arxiv.org 
based on a starting keyword.
"""

import pandas as pd
from queue import Queue
from google import genai

# Modules
import Query as aq
import Embed as eb

def keyword_dict_merge(dict1, dict2):
    """ This just mereges two keyword dictionaries.
    It basically removes duplicate keyword values and
    correctly appends papers to the value list.
    """
    merged = {}
    all_keys = set(dict1) | set(dict2)

    for key in all_keys:
        combined_scores = {}
        for d in (dict1, dict2):
            if key in d:
                for idx, score in d[key]:
                    if idx in combined_scores:
                        combined_scores[idx] = max(combined_scores[idx], score)
                    else:
                        combined_scores[idx] = score
        merged[key] = [(idx, score) for idx, score in combined_scores.items()]
    return merged

def find_similar_papers(keyword, api_key, paper_n=3, rounds=2, similiarity_threshold=0.75, q_size=5):
    """ Finds and links similar research papers from Arxiv based on a starting keyword.
    """
    client = genai.Client(api_key=api_key)
    emb = eb.Embed(client, similiarity_threshold)
    df = pd.DataFrame(columns=['title', 'summary', 'published', 'authors', 'link'])

    # Get initial query
    query = aq.ArxivQuery(keyword, paper_n)
    query.df = query.df.reset_index(drop=True)
    df = pd.concat([df, query.df], ignore_index=True)

    keywords = emb.extract_keywords(query.df)
    all_keywords = {}

    for key, values in keywords.items():
        all_keywords[key] = [(int(idx), score) for idx, score in values]

    # Iterate through the keywords to add papers to our dataframe
    q = Queue()
    seen_terms = set()
    for i in range(rounds):
        for term in keywords:
            if term not in seen_terms:
                q.put(term)
                seen_terms.add(term)

        j = 0
        while j < q_size:
            if q.empty():
                break

            new_term = q.get()
            query = aq.ArxivQuery(new_term, paper_n)
            qdf = query.df.reset_index(drop=True)
            start_idx = len(df)
            df = pd.concat([df, qdf], ignore_index=True)
            keywords = emb.extract_keywords(qdf)
            shifted_keywords = {}
            for key, values in keywords.items():
                shifted_keywords[key] = [(start_idx + int(idx), score) for idx, score in values]
            all_keywords = keyword_dict_merge(all_keywords, shifted_keywords)
            j += 1

    # Link papers and create graph
    linked_papers = emb.link_papers(df, all_keywords)
    G = emb.build_graph_from_links(linked_papers)
    emb.display_graph_with_weights(G)
    return G, all_keywords, df
"""
File: Embed.py
Author: Ziyad Rahman
Date: 2025-22-05

Brief: This file contains the Embed class, which is used to extract keywords from the summary of a research paper.
"""

import json
import time
import numpy as np
import pandas as pd
import networkx as nx
from pydantic import BaseModel
import matplotlib.pyplot as plt

class WordClass(BaseModel):
    word: str
    score: float

class Embed():
    """ This class is used to extract keywords from the summary of a research paper. It uses the Gemini API to
    generate keywords and their scores. It also uses the Gemini API to embed the keywords and calculate the
    cosine similarity between them. The keywords are then combined based on their similarity and the final
    keywords are returned. The class also has a function to link papers based on the keywords and their scores.
    The links are then used to build a graph of the papers and their similarities. The graph is displayed using
    matplotlib.
    """
    def __init__(self, client, similarity_threshold=0.75):
        self.client = client
        self.similarity_threshold = similarity_threshold # Used for cosine similarity

    def _get_keywords(self, summary):
        """ Uses Gemini to extract keywords from the summary of the paper. It used pydantic's BaseModel to define the
        WordClass class for structured outputs.
        """
        prompt = f"""
            You are given a summary of a research paper. Your task is to extract keywords from it that can be used to 
            find other, related papers.

            You will also assign a score from 0 to 1 to each keyword, with 1 being the most relevant. The score should 
            reflect how well the keyword relates to the summary.

            Please keep the keywords as short as possible, ideally one or two words. Avoid using phrases longer than 
            three words. Please give the most relevant keywords first and no more than 3 keywords.

            Return a JSON array of objects. Each object must have two keys:
            - "word": the keyword (a string)
            - "score": a float between 0 and 1

            Example:
            [
            {{ "word": "graph neural network", "score": 0.9 }},
            {{ "word": "neuron", "score": 0.7 }}
            ]

            Here is the summary text:
            {summary}
        """

        response = self.client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[WordClass],
            }
        )

        content = response.text if hasattr(response, 'text') else response.model_dump().get("text", "")
        return json.loads(content)

    def _embed_word(self, word):
        """ Uses Gemini to embed the keyword. This function only handles embedding.
        """
        response = self.client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=word
        )
        return response.model_dump()["embeddings"][0]['values']

    def _cosine_similarity(self, vec1, vec2):
        """ Returns the cosine similarity between two vectors. If either vector is zero, returns 0.0.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _combine_keywords(self, words):
        """ Combines keywords based on their cosine similarity. If the similarity is above the threshold, the keywords
        are combined. The function returns a dictionary with the merged keywords and their scores.

        There's a call to the built-in time module to delay the API calls to avoid rate limits.
        """
        embeddings = {}
        for word in words:
            embeddings[word] = self._embed_word(word)
            time.sleep(1)  # Delay API calls to avoid rate limits

        keys = list(embeddings.keys())
        merged = {}
        to_remove = set()

        # Iterate through the keys and compare their embeddings
        for i in range(len(keys)):
            base = keys[i]
            if base in to_remove:
                continue
            for j in range(i + 1, len(keys)):
                compare = keys[j]
                if compare in to_remove:
                    continue
                sim = self._cosine_similarity(embeddings[base], embeddings[compare])
                if sim >= self.similarity_threshold:
                    merged.setdefault(base, []).append((compare, sim))
                    to_remove.add(compare)

        # Remove duplicates and keep the best score
        for target, merges in merged.items():
            for keyword, _ in merges:
                if keyword in words:
                    combined = {}
                    for idx, score in words[target]:
                        combined[idx] = max(combined.get(idx, 0), score)
                    for idx, score in words[keyword]:
                        combined[idx] = max(combined.get(idx, 0), score)
                    words[target] = [(idx, score) for idx, score in combined.items()]
                    words.pop(keyword, None)

        return merged

    def _reduce_keywords(self, df, words):
        """ This is just a wrapper for the _combine_keywords function.
        """
        self._combine_keywords(words)
        return words

    def extract_keywords(self, df):
        """ This function extracts keywords from the summary of the paper. It uses the _get_keywords function to get the
        keywords and their scores. It then uses the _combine_keywords function to combine the keywords based on their
        cosine similarity. The function returns a dictionary with the keywords and their scores.
        The function also handles exceptions and prints a message if the keyword extraction fails.
        The function also handles the case where the summary is empty or None.

        This is one of the few functions that can/should be called from the ipynb
        """
        word_map = {}
        for i in range(len(df)):
            index = df.index[i]
            summary = df.loc[index, "summary"]
            try:
                keywords = self._get_keywords(summary)
            except Exception as e:
                # skips the paper if the keyword extraction fails (as in is None type)
                continue
            
            # Removes reundant keywords (like if two papers have the same keywords, it combines them)
            for word in keywords:
                key = word["word"]
                score = word["score"]
                if key in word_map:
                    word_map[key].append((index, score))
                else:
                    word_map[key] = [(index, score)]

        # Combines similar keywords
        reduced = self._reduce_keywords(df, word_map)
        return reduced

    def _filter_keywords(self, all_keywords):
        """ This function filters the keywords based on the number of papers they are associated with. If a keyword is
        associated with only one paper, it is removed from the list. The function returns a dictionary with the keywords
        and their associated papers.

        We do this to reduce the number of iterations in the next step (building the graph). Basically, the link_papers 
        function will perform n^2 iterations (where n is the number of keywords), so we want to reduce the number of keywords.
        """
        d = {}
        for keyword, values in all_keywords.items():
            if (len(values) <= 1):
                continue
            d[keyword] = values
        return d

    def _get_score(self, paper1, paper2):
        """ This function calculates the score between two papers. It uses the average of the two scores.
        """
        score1 = paper1
        score2 = paper2
        avg_score = (score1 + score2) / 2
        return avg_score

    def _idx_to_title(self, df, idx):
        """ Converts the index of the paper to the title of the paper. If the index is not found, it returns None.
        """
        try:
            title = df.iloc[idx]['title']
        except KeyError:
            title = None
        return title
    
    def link_papers(self, papers, all_keywords):
        """ This function links the papers based on the keywords and their scores. It uses the _filter_keywords function
        to filter the keywords and then iterates through the keywords to find the papers that are associated with them.
        
        It returns a dataframe, where each column has paper 1, paper 2 and the score between them."""
        d = self._filter_keywords(all_keywords)
        df = pd.DataFrame(columns = ['paper_1', 'paper_2', 'score'])

        for keyword, values in d.items():
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    score = self._get_score(values[i][1], values[j][1])
                    df.loc[len(df)] = [self._idx_to_title(papers, values[i][0]), 
                                       self._idx_to_title(papers, values[j][0]), score]
        return df
    
    def build_graph_from_links(self, links_df):
        """ This just takes the dataframe from link_papers and builds a graph from it.
        """
        G = nx.Graph()
        for _, row in links_df.iterrows():
            paper1 = row["paper_1"]
            paper2 = row["paper_2"]
            weight = row["score"]
            G.add_edge(paper1, paper2, weight=weight)
        return G

    def display_graph_with_weights(self, G, title="Paper Similarity Graph"):
        """ Displays the graph using matplotlib. It uses the spring layout to position the nodes and draws the edges with
        weights. The nodes are colored light blue and the edges are colored based on their weights. The labels are also
        displayed with the weights of the edges. The title of the graph is really just preset.
        """
        plt.figure(figsize=(14, 10))

        pos = nx.spring_layout(G, seed=42, k=1.2, scale=3.0)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)

        edge_weights = [G[u][v]['weight'] for u, v in G.edges]
        nx.draw_networkx_edges(
            G, pos, width=[3 * w for w in edge_weights], 
            edge_color=edge_weights, edge_cmap=plt.cm.Blues
        )

        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
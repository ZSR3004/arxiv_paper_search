import numpy as np
import pandas as pd
import json
import networkx as nx
import matplotlib.pyplot as plt
from pydantic import BaseModel
import time

class WordClass(BaseModel):
    word: str
    score: float

class Embed():
    def __init__(self, client, similarity_threshold=0.75):
        self.client = client
        self.similarity_threshold = similarity_threshold

    def _get_keywords(self, summary):
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
        response = self.client.models.embed_content(
            model="gemini-embedding-exp-03-07",
            contents=word
        )
        return response.model_dump()["embeddings"][0]['values']

    def _cosine_similarity(self, vec1, vec2):
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def _combine_keywords(self, words):
        embeddings = {}
        for word in words:
            embeddings[word] = self._embed_word(word)
            time.sleep(1)  # Delay API calls to avoid rate limits

        keys = list(embeddings.keys())
        merged = {}
        to_remove = set()

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
        self._combine_keywords(words)
        return words

    def extract_keywords(self, df):
        word_map = {}
        for i in range(len(df)):
            index = df.index[i]
            summary = df.loc[index, "summary"]
            try:
                keywords = self._get_keywords(summary)
            except Exception as e:
                # print(f"Failed to get keywords for row {index}: {e}")
                continue

            for word in keywords:
                key = word["word"]
                score = word["score"]
                if key in word_map:
                    word_map[key].append((index, score))
                else:
                    word_map[key] = [(index, score)]

        reduced = self._reduce_keywords(df, word_map)
        return reduced

    def _filter_keywords(self, all_keywords):
        d = {}
        for keyword, values in all_keywords.items():
            if (len(values) <= 1):
                continue
            d[keyword] = values
        return d

    def _get_score(self, paper1, paper2):
        score1 = paper1
        score2 = paper2
        avg_score = (score1 + score2) / 2
        return avg_score

    def _idx_to_title(self, df, idx):
        try:
            title = df.iloc[idx]['title']
        except KeyError:
            title = None
        return title
    
    def link_papers(self, papers, all_keywords):
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
        G = nx.Graph()
        for _, row in links_df.iterrows():
            paper1 = row["paper_1"]
            paper2 = row["paper_2"]
            weight = row["score"]
            G.add_edge(paper1, paper2, weight=weight)
        return G

    def display_graph_with_weights(self, G, title="Paper Similarity Graph"):
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
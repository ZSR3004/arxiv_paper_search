import pandas as pd
import networkz as nwz
from google import genai
from pydantic import BaseModel 

class Keyword(BaseModel):
    word: str
    score: float

class embed():
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = genai.Client(api_key= self.api_key)

    def __get_keywords(self, summary):
        prompt = f"""
        You are given a summary of a research paper. Your task it to extract keywords from it that can be used to find other, related papers. Please
        include anything directly included in the summary that may be helpful. Additionally, if you believe another field of study may have papers
        that are related (for example, a similar algorithm between biology and physics), please include those as well. You will also assign a score
        from 0 to 1 to each keyword, with 1 being the most relevant. The score should be based on how well the keyword relates to the summary.
        The keywords should be in the format of a list of dictionaries, where each dictionary has the following keys:
        word (string): score (float).

        Here is the summary text:
        {summary}
        """

        response = self.client.models.generate_content(
        model = 'gemini-2.0-flash',
        contents = prompt,
        config = {
            'response_mime_type': 'application/json',
            'response_schema': dict,
            })
        return response.model_dump()
       
    def extract_keyword(self, df):
        d = {}
        for i in range(len(df)):
            words = self.__get_keywords(df.iloc[i]['summary'])
            for word in words:
                if word in d:
                    d[word].append((i, words[word]))
                else:
                    d[word] = [(i, words[word])]
        return d

"""
File: Query.py
Author: Ziyad Rahman
Date: 2025-22-05

Brief: This file contains the Query class which is used to query the arxiv.org API.
"""

import re
import datetime
import urllib.parse
import pandas as pd
import feedparser as fp

class Query():
    """ This class is just used to query from arxiv.org. It's main function is to turn the data into a 
    pandas dataframe. For the rest of the program to work with.
    """
    def __init__(self, keyword, max_query = 20):
        self.keyword = keyword
        self.max_query= max_query
        self.encoded_keyword = urllib.parse.quote(keyword)
        self.query = "http://export.arxiv.org/api/query?search_query=all:" + self.encoded_keyword + "&start=0&max_results=" + str(self.max_query)
        self.time = datetime.datetime.now()
        self.raw_data = fp.parse(self.query) 
        self.df = self.parse_entry()

    def __get_authors(self, authors):
        """ This function is used to get the authors from the raw data. It returns a list of authors. 
        """
        t = []
        for author in authors:
            t.append(author['name'])
        return t
    
    def __get_date(self, date):
        """ This function is used to get the date from the raw data. It returns the year. It's not
        used in the final produce (unless the user wants to use it). This was mostly used for debugging
        purposes.
        """
        regexp = r"(\d{4}).*"
        match = re.search(regexp, date)
        if match:
            return match.group(1)
        else:
            return None
    
    def parse_entry(self):
        """ This function is used to parse the raw data into a pandas dataframe.
        """
        data = self.raw_data['entries']
        df = pd.DataFrame(columns = ['title', 'summary', 'published', 'authors', 'link'])
        for entry in data:
            title = entry['title']
            summary = entry['summary']
            published = self.__get_date(entry['published'])
            authors = self.__get_authors(entry['authors'])
            link = entry['link']
            df.loc[len(df)] = [title, summary, published, authors, link]
        return df
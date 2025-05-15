import feedparser as fp
import pandas as pd
import re
import datetime

class ArxivQuery():
    def __init__(self, keyword, max_query = 20):
        self.keyword = keyword
        self.max_query= max_query
        self.query = "http://export.arxiv.org/api/query?search_query=all:" + self.keyword + "&start=0&max_results=" + str(self.max_query)
        self.time = datetime.datetime.now()
        self.raw_data = fp.parse(self.query) 
        self.df = self.parse_entry()

    def __get_authors(self, authors):
        t = []
        for author in authors:
            t.append(author['name'])
        return t
    
    def __get_date(self, date):
        regexp = r"(\d{4}).*"
        match = re.search(regexp, date)
        if match:
            return match.group(1)
        else:
            return None
    
    def parse_entry(self):
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
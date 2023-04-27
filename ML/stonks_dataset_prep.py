# https://python-tradingview-ta.readthedocs.io/en/latest/overview.html
# https://github.com/StreamAlpha/tvdatafeed

from tradingview_ta import TA_Handler, Interval, Exchange
from tvDatafeed import TvDatafeed
from tvDatafeed import Interval as tvDFI
import tv_datafeed_utils as tvd_utils

import stonks_strategy as strategy

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as plt_markers

import datetime
import pytz
from tzlocal import get_localzone



# For dataset collection
# URL libs
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import requests
from bs4 import BeautifulSoup # For extracting contents of html pages
from datetime import datetime
from readability import Document

from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait 
from selenium.webdriver.common.by import By
import selenium.webdriver.support.expected_conditions as EC

import pickle
import os.path


current_timezone = get_localzone()
nasdaq_timezone = pytz.timezone("America/New_York")

# TODO:
#   Pad TV_bars before reshaping to day format so that previous days have correct ranges
#
#   Log into trading view properly using google account so that we can access
#   very short time intervals   



class StockData:
    def __init__(
            self,
            ticker,
            exchange='NASDAQ',
            interval=tvDFI.in_15_minute,
            num_days=1) -> None:
        self.ticker = ticker
        self.exchange = exchange
        self.interval = interval
        self.num_days = num_days


        # Meta data
        self.metadata_path = f'./Dataset/{ticker}-meta'
        if os.path.exists(self.metadata_path):
            # Try load from pickle
            meta_file = open(self.metadata_path, 'rb')
            self.metadata = pickle.load(meta_file)
        else:
            self.metadata = MetaData()
            # Save using pickle
            meta_file = open(self.metadata_path, 'wb')
            pickle.dump(self.metadata, meta_file)




        # Fetch price data
        self.price_dataset_path = f'./Dataset/{ticker}-price-data-{interval}'
        if os.path.exists(self.price_dataset_path):
            # Try load from pickle
            price_dataset_file = open(self.price_dataset_path, 'rb')
            self.bars = pickle.load(price_dataset_file)
        else:
            # Establish trading view connection
            self.tv_dataset = TVConnection(ticker=ticker, exchange=exchange, interval=interval)
            
            # Fetch technical data (Stock price timeseries data)
            self.bars = self.tv_dataset.get_dataset(num_days=num_days, show_plot=False)

            # Fetch stock analysis
            # self.analysis = self.tv_dataset.get_analysis()

            # save using pickle
            price_dataset_file = open(self.price_dataset_path, 'wb')
            pickle.dump(self.bars, price_dataset_file)





        # Fetch FinViz sentiment data (Timeseries sentiment data from FinViz articles)
        self.article_collector_path = f'./Dataset/{ticker}-article-collector'
        if os.path.exists(self.article_collector_path):
            # Try load from pickle
            article_collector_file = open(self.article_collector_path, 'rb')
            self.article_collector = pickle.load(article_collector_file)
        else:
            # Fetch articles using webscraper
            self.article_collector = ArticleCollector(ticker, exchange)
            self.article_collector.fetch_FinViz_articles()
            # save using pickle
            article_collector_file = open(self.article_collector_path, 'wb')
            pickle.dump(self.article_collector, article_collector_file)



        # Fetch Motley sentiment data (Motley Fool articles)
        self.motley_article_collector_path = f'./Dataset/{ticker}-motley-article-collector'
        if os.path.exists(self.motley_article_collector_path):
            # Try load from pickle
            motley_article_collector_file = open(self.motley_article_collector_path, 'rb')
            self.motley_article_collector = pickle.load(motley_article_collector_file)
        else:
            # Fetch articles using webscraper
            self.motley_article_collector = ArticleCollector(ticker, exchange)
            self.motley_article_collector.fetch_Motley_articles()
            # save using pickle
            motley_article_collector_file = open(self.motley_article_collector_path, 'wb')
            pickle.dump(self.motley_article_collector, motley_article_collector_file)




class MetaData:
    def __init__(self) -> None:
        self.dataset_capture_date = datetime.now()
        self.dataset_capture_date_str = datetime.now().strftime('%m-%d %H:%M:%S')




# 6 hours 30 minutes in regular trading hours (390 minutes)
# Without login: 1 minute interval will return at most 15 days of data (5824 bars)

# Trading View Connection class
class TVConnection:
    # tv_username = "*******@gmail.com"
    # tv_password = ""
    # tv_df = TvDatafeed(tv_username, tv_password)


    def __init__(
            self,
            ticker,
            exchange='NASDAQ',
            interval=tvDFI.in_15_minute,
            ):

        self.ticker = ticker
        self.exchange = exchange
        self.interval = interval

        self.tv_df = TvDatafeed()

        self.handler = TA_Handler(
            symbol=self.ticker,
            screener="america",
            exchange=self.exchange,
            interval=self.interval
        )
    

    def get_dataset(
            self,
            num_days=1,
            show_plot=False):

        bars_per_day = int(math.ceil(tvd_utils.bars_per_day(self.interval.value)))
        bars = num_days * bars_per_day

        if bars <= 10:
            print('Invalid call to \'get_dataset\': too few bars')
            return

        # returns:
        #   ['NASDAQ:NVDA' 153.15 153.3 153.04 153.04 6003.0]
        bars = self.tv_df.get_hist(
            symbol=self.ticker,
            exchange=self.exchange,
            interval=self.interval,
            n_bars=bars)
        
        if type(bars) == type(None):
            print("Network error when retreiving bar data")
            return

        # Column 4 = current price
        times = np.array([current_timezone.localize(bar).timestamp() for bar in bars.index])
        bars = bars.to_numpy()[:, 4].view(dtype=object)
        bars = np.dstack((times, bars))[0]

        bars = bars.reshape((num_days, -1, 2))
        bars = np.lib.pad(bars, ((0, 0), (0, 0), (0, 1)), constant_values=(0)) # add one column to mark where to buy & sell

        for day in range(bars.shape[0]):
            bars[day] = strategy.strategy(bars[day])

            if show_plot:
                x_axis_display = np.array([
                    datetime.datetime.fromtimestamp(timestamp, tz=nasdaq_timezone).strftime('%m-%d %H:%M:%S') for timestamp in bars[day][:, 0]
                ])
                bars_plt = bars[day].copy()
                bars_plt[:, 0] = x_axis_display
                plt.plot(bars_plt[:, 0], bars_plt[:, 1])

                markers_up = np.array([bar for bar in bars_plt if bar[2] == 1])
                if len(markers_up) > 0:
                    plt.plot(markers_up[:, 0], markers_up[:, 1], marker=6, ls='none')

                markers_down = np.array([bar for bar in bars_plt if bar[2] == -1])
                if len(markers_down) > 0:
                    plt.plot(markers_down[:, 0], markers_down[:, 1], marker=7, ls='none')

                plt.title(f'{self.exchange}:{self.ticker}  -  Interval:{self.interval.value}')
                plt.show()

        print(f'Fetched TV bars:\n\t{self.interval}\n\t{bars_per_day} bars per day with {num_days} days')

        return bars

    def get_analysis(self):
        return self.handler.get_analysis().summary






FINVIZ_DATE_TYPE = 'FINVIZ_DATE'
MOTLEY_DATE_TYPE = 'MOTLEY_DATE'

class Sentiment_Article: # Eg. FinViz or Motley article
    def __init__(self, url, date_string, date_type) -> None:
        self.url = url
        self.title = ""
        self.body = ""
        
        if date_type == FINVIZ_DATE_TYPE:
            try:
                self.date = datetime_from_str(date_string)
                self.has_date = True
            except:
                self.date = time_from_str(date_string)
                self.has_date = False
        else:
            self.date = date_from_str_simple(date_string)
            self.has_date = True

    def set_date(self, new_date): # Set the date while keeping time the same
        self.date = self.date.replace(year=new_date.year, month=new_date.month, day=new_date.day)
        self.has_date = True
    
    def is_loaded(self):
        return self.title != "" and self.body != ""

    def __repr__(self):
        return f'{self.date}\t{self.title}\n\t{self.url}\n'





class ArticleCollector:
    def __init__(self, ticker, exchange) -> None:
        self.SCRAPEOPS_API_KEY = '58118b46-64d0-4212-bd38-5b00feeba57e' # Brandon's API key
        self.USE_SCAPEOPS = False # Please don't set to true before setting up your own api key

        self.ticker = ticker
        self.exchange = exchange
        self.articles = [] # array of Sentiment_Article



    def fetch_FinViz_articles(self):
        print(f'FETCHING FINVIZ ARTICLES FOR TICKER: {self.ticker}')
        self.articles = self.fetch_related_links_FinViz(ticker=self.ticker)

        for article in self.articles:
        # for article in self.articles[:10]:
            print(f'Fetching article for {self.ticker}: {article.url}')
            self.read_yahoo_finance_article(article)
            print(f'\t\tFinViz article fetched:\t {repr(article)}')
        
        print(f'FINISHED FETCHING FINVIZ ARTICLES FOR TICKER: {self.ticker}')
    
    def fetch_Motley_articles(self):
        print(f'FETCHING MOTLEY ARTICLES FOR TICKER: {self.ticker}')
        self.articles = self.fetch_related_links_Motley(ticker=self.ticker, exchange=self.exchange)

        for article in self.articles:
        # for article in self.articles[:10]:
            print(f'Fetching article for {self.ticker}: {article.url}')
            self.read_article(article)
            print(f'\t\tMotley article fetched:\t {repr(article)}')
        
        print(f'FINISHED FETCHING MOTLEY ARTICLES FOR TICKER: {self.ticker}')



    def fetch_html_source(self, url):
        # User agent generator - https://www.useragentstring.com/pages/useragentstring.php
        request = Request(url = url, headers={'User-Agent': 'Opera/9.80 (X11; Linux i686; Ubuntu/14.10) Presto/2.12.388 Version/12.16.2'})
        html = ""

        try: # Try simple request
            html = str(urlopen(request).read())

        except HTTPError as e:
            if e.code == 403:
                # Connection refused by host - Try proxy using scrape ops
                if self.USE_SCAPEOPS:
                    reponse = requests.get(
                        url='https://proxy.scrapeops.io/v1/', 
                        params={'api_key': self.SCRAPEOPS_API_KEY, 'url': url}
                    )
                    html = reponse.content

                else:
                    print(f'Failed to fetch html source (Scrape Ops disabled)')
        
        return html

    def fetch_related_links_FinViz(self, ticker):
        articles = []
        url = f"https://finviz.com/quote.ashx?t={ticker}&p=d"
        html = self.fetch_html_source(url)

        soup = BeautifulSoup(html, 'html.parser')

        l = soup.findAll("div", {"class":"news-link-container"})

        for div in l:
            date_string = div.parent.parent.findChildren('td', recursive=False)[0].text
            for i in div.findAll("a"):
                articles.append(Sentiment_Article(url=i['href'], date_string=date_string, date_type=FINVIZ_DATE_TYPE))

        for i in range(1, len(articles), 1):
            if articles[i].has_date == False:
                articles[i].set_date(articles[i-1].date)

        return articles

    def fetch_related_links_Motley(self, ticker, exchange):
        articles = []

        # Create webdriver object
        driver = webdriver.Chrome()
        driver.get(f"https://www.fool.com/quote/{exchange.lower()}/{ticker.lower()}/")

        load_more_clicks_count = 500 # Change this to load more articles
        for i in range(load_more_clicks_count):
            load_more_button = driver.find_element(By.XPATH, '//*[@id="quote-news-analysis"]/div[3]/div[1]/button')
            # Scroll to load more button
            driver.execute_script("arguments[0].scrollIntoView();", load_more_button)
            # Wait until button is visible
            WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="quote-news-analysis"]/div[3]/div[1]/button')))
            # Wait until button is clickable, then click
            driver.execute_script(
                "arguments[0].click();",
                WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="quote-news-analysis"]/div[3]/div[1]/button')))
            )
            print(f'Loading Motley Fool articles: ({i}/{load_more_clicks_count})')
            # Wait until loading indicator is hidden
            WebDriverWait(driver, 20).until(EC.invisibility_of_element_located((By.XPATH, '//*[@id="quote-news-analysis"]/div[3]/div[1]/button/svg')))
            # Sometimes the WebDriverWait events still exit early so we wait a little bit extra (This isn't ideal solution)
            driver.implicitly_wait(1)

        html = driver.page_source
        driver.close()

        soup = BeautifulSoup(html, 'html.parser')
        l = soup.findAll("div", {"id":"news-analysis-container"})

        links = []
        for div in l:
            for i in div.findAll("a"):
                href = i['href']
                if href.startswith('http'): # Motley fool related article links don't start with HTTPs, although some links like ads/sign up links exist in this div
                    continue
                if href.startswith('/terms'): # Discard unrelated articles - Eg https://fool.com/terms/d/digital wallet
                    continue
                # We are only interested in links containing the date prefix
                # Relevant links should start with investing/{date}/
                if href.startswith('/investing/20'):
                    links.append(f'https://fool.com{href}')

        links_count_with_duplicates = len(links)
        links = list(set(links)) # Convert to set and back to list to remove duplicate links
        if links_count_with_duplicates - len(links) > 0:
            print(f'WARNING (Motley links):\t {links_count_with_duplicates - len(links)} duplicate links found and removed')

        for link in links:
            link_split = link.split('/')
            date_string = f'{link_split[6]}/{link_split[5]}/{link_split[4]}' # DD/MM/YYYY   Eg. 21/04/2023
            articles.append(Sentiment_Article(url=link, date_string=date_string, date_type=MOTLEY_DATE_TYPE))
        
        return articles





    def print_links(links, limit=-1):
        print(f'Total links fetched: {len(links)}\n')
        if limit == -1:
            for link in links:
                print(link)
        else:
            for i in range(limit):
                print(links[i])

    

    def read_article(self, article):
        html = self.fetch_html_source(article.url)
        doc = Document(html)

        try: # Dereferencing doc might fail if document is empty for some reason
            article.title = doc.title()
            article.body = doc.summary(html_partial=True)
        except:
            pass

    def read_yahoo_finance_article(self, article):
        html = self.fetch_html_source(article.url)
        soup = BeautifulSoup(html, 'html.parser')

        l = soup.findAll("div", {"class":"caas-readmore caas-readmore-collapse"})
        
        external_link = ""
        for div in l:
            for i in div.findAll("a"):
                external_link = i['href']
            external_link = external_link.replace('www.', '')
        
        if external_link != "": # Yahoo article had 'Read more' button
            html = self.fetch_html_source(external_link)

        doc = Document(html)

        try: # Dereferencing doc might fail if document is empty for some reason
            article.title = doc.title()
            article.body = doc.summary(html_partial=True)
        except:
            pass



def datetime_from_str(date):
    return datetime.strptime(date, '%b-%d-%y %I:%M%p')
def time_from_str(date):
    return datetime.strptime(date, '%I:%M%p')

def date_from_str_simple(date):
    return datetime.strptime(date, '%d/%m/%Y')  # Eg. 21/04/2023




if __name__=="__main__":
    nvda_data = StockData(ticker='NVDA', interval=tvDFI.in_daily, num_days=2600)
    # amd_data = StockData(ticker='AMD', interval=tvDFI.in_1_hour, num_days=365)
    # qcom_data = StockData(ticker='QCOM', interval=tvDFI.in_1_hour, num_days=12)
    # txn_data = StockData(ticker='TXN', interval=tvDFI.in_1_hour, num_days=12)
    # avgo_data = StockData(ticker='AVGO', interval=tvDFI.in_1_hour, num_days=12)
    # tsm_data = StockData(ticker='TSM', exchange='NYSE', interval=tvDFI.in_1_hour, num_days=12)
    # motley_article_collector_file = open('NVDA-motley-article-collector.pkl', 'wb')

    # motley_article_collector = ArticleCollector(ticker='NVDA', exchange='nasdaq')
    # motley_article_collector.fetch_Motley_articles()
    # pickle.dump(motley_article_collector, motley_article_collector_file)

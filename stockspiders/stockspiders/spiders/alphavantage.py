import scrapy
import os

class AlphaVantage(scrapy.Spider):
    name = "alphavantage"

    def start_requests(self):
        ticker = getattr(self, 'ticker', None)
        if (ticker is None):
            raise ValueError('Please provide a ticker symbol!')

        apikey = os.getenv('alphavantage_apikey')
        print(apikey)
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol={0}&apikey={1}'.format(ticker, apikey)

        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        ticker = getattr(self, 'ticker', None)
        with open(ticker + '.txt', 'wb') as f:
            f.write(response.body)
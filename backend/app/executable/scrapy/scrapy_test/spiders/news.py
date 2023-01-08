from urllib.parse import urljoin

import scrapy
from scrapy_test.items import NewsItem


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["news.yahoo.co.jp"]
    start_urls = ["https://news.yahoo.co.jp/"]

    def parse(self, response):

        itm = NewsItem()
        itm["url"] = response.url
        itm["html"] = response.body.decode(response.encoding)

        yield itm

        for ank in response.css("a::attr(href)"):
            url: str = ank.get()
            if "/articles/" not in url:
                continue

            if url[0] == "/":
                url = urljoin(response.url, url)
            yield scrapy.Request(url, callback=self.parse)

        return

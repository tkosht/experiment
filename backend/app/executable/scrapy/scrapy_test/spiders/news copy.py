from urllib.parse import urljoin

import scrapy
from scrapy_test.items import NewsItem

# from bs4 import BeautifulSoup
#
# def _filter_text(response) -> str:
#     soup: BeautifulSoup = BeautifulSoup(response.body, "lxml")
#     for tg in ["script", "noscript", "meta"]:
#         try:
#             soup.find(tg).replace_with(" ")
#         except Exception:
#             # NOTE: Not Found `tg` tag
#             pass
#     return soup.get_text()


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["news.yahoo.co.jp"]
    start_urls = ["https://news.yahoo.co.jp/"]

    def parse(self, response):
        # html_text = "".join(response.selector.xpath("//body//text()").extract()).strip()
        # soup = BeautifulSoup(html_text, "lxml")
        # text = _filter_text(response)

        itm = NewsItem()
        itm["url"] = response.url
        itm["html"] = response.body
        # itm["text"] = text
        yield itm

        # urls = [a.get() for a in response.css("main").css("a::attr(href)") if a.get()[:len("https://")] == "https://"]
        # for a in response.css("a::attr(href)"):
        #     url: str = a.get()
        #     if "/articles/" not in url:
        #         continue

        #     if url[0] == "/":
        #         url = urljoin(response.url, url)
        #     yield scrapy.Request(url, callback=self.parse)

        return

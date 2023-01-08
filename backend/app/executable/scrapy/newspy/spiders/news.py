import logging
from urllib.parse import urljoin

import scrapy
from newspy.items import NewsItem

g_logger = logging.getLogger(__name__)


class NewsSpider(scrapy.Spider):
    name = "news"
    allowed_domains = ["news.yahoo.co.jp"]
    start_urls = ["https://news.yahoo.co.jp/"]

    def parse(self, response):

        item = NewsItem()
        item["url"] = response.url
        item["html"] = response.body.decode(response.encoding)

        yield item

        for ank in response.css("a::attr(href)"):
            url: str = ank.get().strip()
            if not url:
                continue

            if url[0] in ["/"]:
                url = urljoin(response.url, url)

            scheme = "http"
            if url[: len(scheme)] != scheme:
                g_logger.warning(f"Found non {scheme} scheme: skipped [{url=}]")
                continue

            yield scrapy.Request(url, callback=self.parse)

        return

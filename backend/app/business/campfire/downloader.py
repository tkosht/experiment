import datetime
import re
import time
from inspect import signature

import typer
from omegaconf import DictConfig
from retry import retry
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.ui import WebDriverWait

from app.business.campfire.graphdb import GraphDb
from app.business.campfire.project import ProjectDetails, ProjectRecord


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def pickup_numbers(text: str) -> int:
    return int(re.sub(r"\D", "", text))


class CampfireFetcher(object):
    """
    A class for automating web scraping using Selenium.
    """

    def __init__(self) -> None:
        """
        Initializes the AutomateSelenium class.
        """
        self.driver = self.create_driver()

    def create_driver(self) -> webdriver.Chrome:
        """
        Creates a Selenium WebDriver instance with the specified options.
        Returns:
            driver (webdriver.Chrome): The created WebDriver instance.
        """
        options = Options()
        options.binary_location = "/opt/chrome-linux64/chrome"
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--enable-chrome-browser-cloud-management")
        options.add_argument("--enable-javascript")
        service = Service("/usr/local/bin/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        return driver

    @retry(TimeoutException, tries=3, delay=2)
    def _fetch(
        self, url: str, elm_path: str, wait_path: str = None, max_tries: int = 3
    ) -> list[WebElement]:
        """
        Fetches elements from a web page using the specified URL and element paths.
        Args:
            url (str): The URL of the web page to fetch elements from.
            elm_path (str): The XPath of the elements to fetch.
            wait_path (str, optional): The XPath of the element to wait for before fetching. If specified, the method will wait for the element to be present before fetching. Defaults to None.
            max_tries (int, optional): The maximum number of tries to fetch the elements. Defaults to 3.
        Returns:
            elms (list): The fetched elements.
        Raises:
            TimeoutException: If the element to wait for is not found within the specified time.
        """
        max_wait: int = 13  # secs

        self.driver.get(url)

        if wait_path is not None:
            WebDriverWait(self.driver, max_wait).until(
                expected_conditions.presence_of_element_located((By.XPATH, wait_path))
            )

        # TODO: 以下の処理を、scroll が終わるまで繰り返す
        # scroll の高さ高さを取得する
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        # 画面の最下部までスクロールする
        scroll_size = 200
        for idx in range(0, last_height, scroll_size):
            # lazy loading で追加される要素を取得する
            self.driver.execute_script(f"window.scrollTo({idx}, {idx+scroll_size});")

            # scroll が終わるまで待つ
            time.sleep(0.25)

        elms: list[WebElement] = self.driver.find_elements(by=By.XPATH, value=elm_path)
        return elms

    def fetch_project_list_boxes(
        self, page: int = 1, sortby: str = "popular"
    ) -> list[WebElement]:
        """
        Fetches the project list boxes from Campfire website.

        Args:
            page (int): The page number to fetch (default is 1).
            sortby (str): The sorting option for the projects (default is "popular").

        Returns:
            list: A list of project boxes fetched from the page.
        """
        url = f"https://camp-fire.jp/projects?page={page}&sort={sortby}"
        xpath = '//*[@id="section_popular"]/div'
        boxes: list[WebElement] = self._fetch(
            url=url, elm_path=xpath, wait_path='//*[@id="fb-root"]'
        )

        return boxes

    def fetch_projects(self, params: DictConfig) -> list[ProjectRecord]:
        """
        Fetches projects from Campfire based on the given parameters.

        Args:
            params (DictConfig): A dictionary-like object containing the parameters for fetching projects.

        Returns:
            list[ProjectRecord]: A list of ProjectRecord objects representing the fetched project records.
        """
        projects = []
        for n_page in range(1, params.max_pages + 1):
            print(f"{now()} Fetching page {n_page} ...")
            project_boxes: list[WebElement] = self.fetch_project_list_boxes(
                page=n_page, sortby=params.sortby
            )
            project_records: list[ProjectRecord] = [
                self.fetch_project(n_page, bx) for bx in project_boxes
            ]
            print(f"{now()} {n_page=} {len(project_records)} projects fetched.")
            projects += project_records

        print(f"{now()} total is {len(projects)} projects fetched.")
        return projects

    def fetch_project(self, page: int, bx: WebElement) -> ProjectRecord:
        """
        Fetches project information from a web element.

        Args:
            bx (WebElement): The web element containing the project information.

        Returns:
            ProjectRecord: An instance of the ProjectRecord class containing the fetched project information.
        """
        # NOTE: ファンディング情報を取得する
        thumnail_elm: WebElement = bx.find_element(
            By.XPATH, './/*[@class="box-thumbnail"]'
        )
        img_url = thumnail_elm.find_element(By.XPATH, ".//img").get_attribute("src")
        # detail_url like 'https://camp-fire.jp/projects/view/736020?list=projects_popular_page1'
        detail_url = thumnail_elm.find_elements(By.XPATH, ".//a")[-1].get_attribute(
            "href"
        )  # first or second element in the <a> tag list
        try:
            area: str = bx.find_element(By.XPATH, './/*[@class="area"]').text
        except NoSuchElementException:
            # NOTE: 一部のプロジェクトは、エリア情報がない場合がある
            area = None

        title: str = bx.find_element(By.XPATH, './/*[@class="box-title"]').text
        status: str = "OPEN"
        try:
            meter: str = bx.find_element(By.XPATH, './/*[@class="meter"]').text
        except NoSuchElementException:
            # NOTE: 100% になっている場合があるので、その場合は、success-summary から取得する
            meter: str = "100%"
            status: str = bx.find_element(
                By.XPATH, './/*[@class="success-summary"]'
            ).text

        category: str = bx.find_element(By.XPATH, './/*[@class="category"]').text
        owner: str = bx.find_element(By.XPATH, './/*[@class="owner"]').text
        current_funding: str = bx.find_element(By.XPATH, './/*[@class="total"]').text
        supporters: str = bx.find_element(By.XPATH, './/*[@class="rest"]').text
        remaining_days: str = bx.find_element(By.XPATH, './/*[@class="per"]').text

        # NOTE: 形式が以下のようになっているので、整形する
        # - meter:  1009% -> 1009: int
        # - current_funding:  現在\n30,270,000円 -> 30270000: int
        # - supporters:  支援者\n450人 -> 450: int
        # - remaining_days:  残り\n7日 -> 7: int
        meter = int(pickup_numbers(meter))
        current_funding = int(pickup_numbers(current_funding))
        supporters = int(pickup_numbers(supporters))

        if "終了" in remaining_days:
            remaining_days = 0
        else:
            remaining_days = int(pickup_numbers(remaining_days))

        return ProjectRecord(
            img_url,
            page,
            detail_url,
            area,
            title,
            meter,
            category,
            owner,
            current_funding,
            supporters,
            remaining_days,
            status,
        )

    def fetch_project_detail(self, detail_url: str) -> ProjectDetails:
        """
        Fetches the detailed information of a project from the given URL.

        Args:
            detail_url (str): The URL of the project detail page.

        Returns:
            ProjectRecord: An instance of the ProjectRecord class containing the fetched project information.
        """
        elms: list[WebElement] = self._fetch(
            url=detail_url,
            elm_path='//label[@class="project-name"]',
            wait_path='//*[@id="fb-root"]',
        )
        title = elms[0].text
        print(title)

        elms: list[WebElement] = self._fetch(
            url=detail_url,
            elm_path="//",
            wait_path='//*[@id="fb-root"]',
        )

    def quit_driver(self):
        """
        Quits the Selenium WebDriver instance.
        """
        self.driver.quit()
        self.driver = None


# TODO: Campfire のサイトから クラウドファンディングの情報一覧を取得する処理を関数にする
# TODO: Campfire のサイトから クラウドファンディングの情報一覧に対応する情報詳細を取得する処理を関数にする


def _main(params: DictConfig):
    # Campfire のサイトから クラウドファンディングの情報一覧を取得する
    cfr = CampfireFetcher()
    project_records: list[ProjectRecord] = cfr.fetch_projects(params)

    # GraphDb に取得した情報を保存する
    g = GraphDb()
    try:
        for pr in project_records:
            # g.add_node(
            #     label="Project",
            #     img_url=pr.img_url,
            #     page=pr.page,
            #     detail_url=pr.detail_url,
            #     area=pr.area,
            #     title=pr.title,
            #     meter=pr.meter,
            #     category=pr.category,
            #     owner=pr.owner,
            #     current_funding=pr.current_funding,
            #     supporters=pr.supporters,
            #     remaining_days=pr.remaining_days,
            #     status=pr.status,
            # )
            cfr.fetch_project_detail(pr.detail_url)
        print(f"{now()} {len(project_records)} projects saved to the graph database.")
    finally:
        g.close()
        cfr.quit_driver()


def config():
    cfg = DictConfig(dict(is_experiment=True, do_share=False))
    return cfg


def main(
    max_pages: int = 1,
    sortby: str = "popular",
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    typer.run(main)

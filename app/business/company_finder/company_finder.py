import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import requests
import whois
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from rich import print as rprint
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class CompanyAttributes:
    """企業の属性情報を格納するクラス"""

    industry: Optional[str] = None
    location: Optional[str] = None
    established_year: Optional[int] = None


@dataclass
class CompanyWebsite:
    """企業のWebサイト情報を格納するクラス"""

    url: str
    confidence_score: float
    ssl_verified: bool
    whois_matched: bool
    meta_matched: bool
    last_checked: datetime


class CompanyWebsiteFinder:
    """企業のWebサイトを特定するクラス"""

    def __init__(self, api_key: str, cse_id: str):
        """
        Args:
            api_key (str): Google Custom Search API Key
            cse_id (str): Google Custom Search Engine ID
        """
        self.api_key = api_key
        self.cse_id = cse_id
        self.service = build("customsearch", "v1", developerKey=api_key)
        self.cache = {}  # Simple in-memory cache

    def find_company_website(
        self, company_name: str, attributes: Optional[CompanyAttributes] = None
    ) -> Optional[CompanyWebsite]:
        """企業の公式Webサイトを特定します。

        Args:
            company_name (str): 企業名
            attributes (Optional[CompanyAttributes]): 企業の属性情報

        Returns:
            Optional[CompanyWebsite]: 企業のWebサイト情報。見つからない場合はNone。
        """
        # キャッシュをチェック
        cache_key = f"{company_name}_{attributes.industry if attributes else ''}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            # Google Custom Search APIで検索
            search_results = self._search_company(company_name, attributes)
            if not search_results:
                return None

            # 検索結果をフィルタリングして評価
            candidates = self._evaluate_candidates(search_results, company_name)
            if not candidates:
                return None

            # 最も確度の高い候補を選択
            best_candidate = max(candidates, key=lambda x: x[1])
            url, confidence_score = best_candidate

            # 真正性の確認
            ssl_verified = self._verify_ssl(url)
            whois_matched = self._verify_whois(url, company_name)
            meta_matched = self._verify_meta(url, company_name)

            # 結果を作成
            result = CompanyWebsite(
                url=url,
                confidence_score=confidence_score,
                ssl_verified=ssl_verified,
                whois_matched=whois_matched,
                meta_matched=meta_matched,
                last_checked=datetime.now(),
            )

            # キャッシュに保存
            self.cache[cache_key] = result
            return result

        except Exception as e:
            rprint(f"[bold red]Error finding company website: {str(e)}[/bold red]")
            return None

    def _search_company(
        self, company_name: str, attributes: Optional[CompanyAttributes]
    ) -> List[str]:
        """Google Custom Search APIを使用して企業を検索します。"""
        query = company_name
        if attributes:
            if attributes.industry:
                query += f" {attributes.industry}"
            if attributes.location:
                query += f" {attributes.location}"

        try:
            result = self.service.cse().list(q=query, cx=self.cse_id, num=10).execute()
            return [item["link"] for item in result.get("items", [])]
        except Exception as e:
            rprint(f"[bold red]Search API error: {str(e)}[/bold red]")
            return []

    def _evaluate_candidates(
        self, urls: List[str], company_name: str
    ) -> List[Tuple[str, float]]:
        """URLの候補を評価し、スコアを付けます。"""
        scored_candidates = []
        company_name_normalized = company_name.lower().replace(" ", "")

        for url in urls:
            score = 0.0
            domain = self._extract_domain(url)

            # ドメインに企業名が含まれている
            if company_name_normalized in domain.lower().replace(".", ""):
                score += 0.4

            # 法人ドメインの評価
            if domain.endswith(".co.jp"):
                score += 0.3
            elif domain.endswith(".or.jp"):
                score += 0.2
            elif domain.endswith(".com"):
                score += 0.1

            # URLパターンの評価
            if re.search(r"/(company|corporate|about)/?", url):
                score += 0.2

            if score > 0:
                scored_candidates.append((url, score))

        return scored_candidates

    def _verify_ssl(self, url: str) -> bool:
        """SSLの有効性を確認します。"""
        try:
            requests.get(url, verify=True)
            return True
        except requests.RequestException:
            return False

    def _verify_whois(self, url: str, company_name: str) -> bool:
        """Whois情報と企業名を照合します。"""
        try:
            domain = self._extract_domain(url)
            w = whois.whois(domain)
            if not w.domain_name:
                return False

            registrant = str(w.registrant_name).lower() if w.registrant_name else ""
            return company_name.lower() in registrant
        except (whois.parser.PywhoisError, Exception):
            return False

    def _verify_meta(self, url: str, company_name: str) -> bool:
        """メタデータと企業名を照合します。"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")

            # titleタグをチェック
            if soup.title and company_name.lower() in soup.title.string.lower():
                return True

            # meta descriptionをチェック
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if (
                meta_desc
                and company_name.lower() in meta_desc.get("content", "").lower()
            ):
                return True

            return False
        except requests.RequestException:
            return False

    @staticmethod
    def _extract_domain(url: str) -> str:
        """URLからドメイン名を抽出します。"""
        return re.sub(r"^https?://", "", url).split("/")[0]


def main():
    """メイン関数"""
    # 環境変数から認証情報を取得
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")

    if not api_key or not cse_id:
        rprint(
            "[bold red]Error: GOOGLE_API_KEY and GOOGLE_CSE_ID "
            "environment variables are required.[/bold red]"
        )
        return

    finder = CompanyWebsiteFinder(api_key, cse_id)

    # テスト用の企業情報
    company_name = "株式会社サイバーエージェント"
    attributes = CompanyAttributes(
        industry="IT・インターネット",
        location="東京都渋谷区",
    )

    # 企��のWebサイトを検索
    result = finder.find_company_website(company_name, attributes)

    if result:
        # 結果を表示
        table = Table(title="Company Website Search Result")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("URL", result.url)
        table.add_row("Confidence Score", f"{result.confidence_score:.2f}")
        table.add_row("SSL Verified", "✓" if result.ssl_verified else "✗")
        table.add_row("WHOIS Matched", "✓" if result.whois_matched else "✗")
        table.add_row("Meta Matched", "✓" if result.meta_matched else "✗")
        table.add_row("Last Checked", result.last_checked.strftime("%Y-%m-%d %H:%M:%S"))

        console.print(table)
    else:
        rprint("[bold red]No company website found.[/bold red]")


if __name__ == "__main__":
    main()

import requests
from bs4 import BeautifulSoup, NavigableString, Tag


def html_to_markdown(element):
    markdown = ""
    if isinstance(element, NavigableString):
        return element.strip()
    elif isinstance(element, Tag):
        if element.name == "h1":
            return "# " + "".join([html_to_markdown(e) for e in element.contents]) + "\n\n"
        elif element.name == "h2":
            return "## " + "".join([html_to_markdown(e) for e in element.contents]) + "\n\n"
        elif element.name == "h3":
            return "### " + "".join([html_to_markdown(e) for e in element.contents]) + "\n\n"
        elif element.name == "h4":
            return "#### " + "".join([html_to_markdown(e) for e in element.contents]) + "\n\n"
        elif element.name == "p":
            return "".join([html_to_markdown(e) for e in element.contents]) + "\n\n"
        elif element.name == "ul":
            return "".join(["* " + html_to_markdown(li) + "\n" for li in element.find_all("li")]) + "\n"
        elif element.name == "ol":
            return (
                "".join([f"{i+1}. " + html_to_markdown(li) + "\n" for i, li in enumerate(element.find_all("li"))])
                + "\n"
            )
        elif element.name == "strong" or element.name == "b":
            return "**" + "".join([html_to_markdown(e) for e in element.contents]) + "**"
        elif element.name == "em" or element.name == "i":
            return "*" + "".join([html_to_markdown(e) for e in element.contents]) + "*"
        elif element.name == "a":
            href = element.get("href", "")
            text = "".join([html_to_markdown(e) for e in element.contents])
            return f"[{text}]({href})"
        else:
            return "".join([html_to_markdown(e) for e in element.contents])
    else:
        return ""


def to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return "".join([html_to_markdown(e) for e in soup.contents])


if __name__ == "__main__":
    url = "https://www.google.com/"
    response = requests.get(url)
    html = response.text
    markdown = to_markdown(html)
    print(markdown)

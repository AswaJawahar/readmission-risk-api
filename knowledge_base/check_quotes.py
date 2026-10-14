"""Checks that every quote in sources.json appears word for word on its source page.

Needs internet access, so it is run by hand (not part of pytest):
    python knowledge_base/check_quotes.py

Reviewers can use it before replacing verified_by = "pending" with their name;
it confirms the quote is real, but the summary still needs a human read.
"""
import html
import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

SOURCES_PATH = Path(__file__).resolve().parent / "sources.json"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
SKIP_TAGS = {"script", "style", "noscript", "svg"}
# Only block elements separate words; inline tags (links, emphasis) must not add spaces
BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "br",
              "section", "article", "table", "ul", "ol", "dd", "dt", "blockquote", "figcaption"}


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.skip_depth = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        elif tag in BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def normalize(text):
    return re.sub(r"\s+", " ", html.unescape(text).replace("\xa0", " ")).strip()


def page_text(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8", errors="replace")
    parser = TextExtractor()
    parser.feed(raw)
    return normalize("".join(parser.parts))


def main():
    sources = json.loads(SOURCES_PATH.read_text(encoding="utf-8"))["sources"]
    missing = 0
    for source in sources:
        text = page_text(source["retrieved_from"])
        for excerpt in source["excerpts"]:
            found = normalize(excerpt["quote"]) in text
            missing += not found
            print(f"{'ok     ' if found else 'MISSING'} {excerpt['excerpt_id']}")
    print(f"\n{missing} quote(s) not found" if missing else "\nAll quotes found on their source pages")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())

import requests
from googlesearch import search
from bs4 import BeautifulSoup as soup
import time
from SPARQLWrapper import SPARQLWrapper, JSON
import wikipedia
from urllib.parse import unquote
from typing import List, Optional, Dict, Union
import random
import sys

def get_results(data):
    qid_ids = ""
    item_out = {}
    item_out_list = {}
    for counter, qid_id_i in enumerate(data):
        if counter == 0:
            qid_ids += f"wd:{qid_id_i}"
        else:
            qid_ids += f" wd:{qid_id_i}"
    user_agent = "WDQS-example Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    endpoint_url = "https://query.wikidata.org/sparql"
    languages = ["de", "en"]
    for language in languages:
        query = f"""
        SELECT ?item ?itemLabel ?itemDescription ?altLabel ?labelLang
    WHERE
    {{
      VALUES ?item {{{qid_ids}}} .
      
      # Get the label, but only for English and German
      ?item rdfs:label ?itemLabel .
      FILTER(LANG(?itemLabel) IN ("{language}"))
      BIND(LANG(?itemLabel) AS ?labelLang)
    
      # Get the description, only for English and German
      OPTIONAL {{ ?item schema:description ?itemDescription . FILTER(LANG(?itemDescription) IN ("{language}")) }}
    
      # Get the aliases (also known as), only for English and German
      OPTIONAL {{ ?item skos:altLabel ?altLabel . FILTER(LANG(?altLabel) IN ("{language}")) }}
    }}
    """
        sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        result = sparql.query().convert()
        for item_i in result["results"]["bindings"]:
            qid = item_i["item"]["value"].split("/")[-1]
            if qid not in item_out:
                item_out[qid] = {}
            if language not in item_out[qid]:
                item_out[qid][language] = {
                    "url": item_i["item"]["value"]
                }
            label = item_i["itemLabel"]["value"]
            if "altLabel" in item_i:
                alt_label = item_i["altLabel"]["value"]
            else:
                alt_label = None
            if "itemDescription" in item_i:
                description = item_i["itemDescription"]["value"]
            else:
                description = None
            if "Label" not in item_out[qid][language]:
                item_out[qid][language]["Label"] = set()
            item_out[qid][language]["Label"].add(label)
            if "Description" not in item_out[qid][language]:
                item_out[qid][language]["Description"] = set()
            if description is not None:
                item_out[qid][language]["Description"].add(description)
            if "AltLabel" not in item_out[qid][language]:
                item_out[qid][language]["AltLabel"] = set()
            if alt_label is not None:
                item_out[qid][language]["AltLabel"].add(alt_label)
    for qid in item_out:
        item_out_list[qid] = {}
        for language in item_out[qid]:
            item_out_list[qid][language] = {}
            for val_i in ["Label", "Description", "AltLabel"]:
                item_out_list[qid][language][val_i] = list(item_out[qid][language][val_i])
            item_out_list[qid][language]["url"] = item_out[qid][language]["url"]
    return item_out_list

def search_wikidata(query, langauge):
    headers = {'Accept': 'application/json'}
    response = requests.get(f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={query}&language={langauge}&format=json", headers=headers)
    data = response.json()
    out_i = []
    if len(data["search"]) > 0:
        out_i.append(data["search"][0]["id"])
    return out_i

def wikipedia_search(keywords: List[str], results=3):
    output = []
    for keyword in keywords:
        try:
            page = wikipedia.search(keyword, results=results)
            output.append(page)
        except Exception as e:
            print(e)
    return output

def wikipedia_text_extract(title: str) -> Dict[str, str]:
    """
    This function extracts the text from a wikipedia article.
    :param title: Title of the wikipedia article
    :return: Text of the wikipedia article
    """
    output = {}
    counter = 1
    try:

        # wikipedia.set_lang("de")
        # page = wikipedia.search(pageid=5541153)
        page = wikipedia.page(title, auto_suggest=False)
        output["search"] = title
        output["title"] = page.title
        output["url"] = page.url
        output["summary"] = page.summary
        output["content"] = page.content
        content_split = output["content"].split("\n")
        chapter_start = "Einleitung"
        chapter_map = {}
        chapter_middle = ""
        chapters_out = {
            "Einleitung": {
                "text": "",
                "subchapter": {}
            }
        }
        chapter_map["Einleitung"] = counter
        complete_text = ""
        for content_i in content_split:
            if content_i == "":
                continue
            if content_i.startswith("==") and not content_i.startswith("==="):
                chapter_start = content_i.replace("=", "")
                chapter_middle = ""
                continue
            if content_i.startswith("==="):
                chapter_middle = content_i.replace("=", "")
                counter += 1
                chapter_map[chapter_middle] = counter
                continue
            if chapter_start not in chapters_out:
                chapters_out[chapter_start] = {
                    "text": "",
                    "subchapter": {}
                }
                counter += 1
                chapter_map[chapter_start] = counter
                # continue
            if chapter_middle != "":
                if chapter_middle not in chapters_out[chapter_start]["subchapter"]:
                    chapters_out[chapter_start]["subchapter"][chapter_middle] = ""
                chapters_out[chapter_start]["subchapter"][chapter_middle] += f"{content_i}\n"
                complete_text += f"{content_i}\n"
            else:
                chapters_out[chapter_start]["text"] += f"{content_i}\n"
                complete_text += f"{content_i}\n"
        output["chapters"] = chapters_out
        search_text = ""
        if "#" in title:
            subtitles = title.split("#")
            sub_start = subtitles[1]
            sub_end = subtitles[-1]
            if sub_end == sub_start:
                sub_end = ""
                if sub_start in chapters_out:
                    search_text = chapters_out[sub_start]["text"]
            else:
                if sub_start in chapters_out and sub_end in chapters_out[sub_start]["subchapter"]:
                    if sub_end in chapters_out[sub_start]["subchapter"]:
                        search_text = chapters_out[sub_start]["subchapter"][sub_end]
        else:
            search_text = complete_text
        output["search_text"] = search_text
        output["map"] = chapter_map
        return output
    except Exception as e:
        print(e)
    return output

def search_google(query, lang):
    output_i = search(f"{query}", num_results=10, lang=lang, advanced=True, sleep_interval=5, timeout=5, safe="active", unique=True)
    safe_query = {}
    for output in output_i:
        try:
            url = output.url
            if url == "":
                page = "Empty"
                title = "Empty"
                text = "Empty"
                url = "Empty"
                page_soup = "Empty"
            else:
                page = requests.get(url)
                page_soup = soup(page.content, 'html.parser')
                text = page_soup.get_text("\n", strip=True).strip()
                title = output.title
            safe_query[url] = {
                "html": str(page_soup),
                "text": text,
                "url": url,
                "title": title
            }
            time.sleep(8)
        except Exception as e:
            try:
                page = requests.get(url)
                page_soup = soup(page.content, 'html.parser', from_encoding="iso-8859-1")
                text = page_soup.get_text("\n", strip=True).strip()
                title = output.title
                safe_query[url] = {
                    "html": str(page_soup),
                    "text": text,
                    "url": url,
                    "title": title
                }
            except Exception as e:
                print(f"Error: {e}")
                safe_query[url] = {
                    "html": "Error",
                    "text": "Error",
                    "url": url,
                    "title": "Error"
                }
    return safe_query

def google_search_words(keyword: str, pre_search: str, lang: str, results=2) -> Dict[str, List[str]]:
    """
    This function searches for the keywords in the search string and returns the keywords found.
    :param keywords: List of keywords to search for
    :param pre_search: Search string
    :return: List of keywords found in the search string
    """
    found_keywords = {}
    output_search = search(f"{pre_search} {keyword}",  num_results=results, lang=lang, advanced=True, sleep_interval=5, timeout=5, safe="active", unique=True)
    for j_utf in output_search:
        try:
            j = unquote(j_utf.url)
            title = j.split("/")[-1]
            title = title.replace("_", " ")
            main_title = title.split("#")[0]
            found_keywords[main_title] = j.split("#")[0]
            wait_time = random.Random().randint(5, 10)
            time.sleep(wait_time)
        except Exception as e:
            pass
    wait_time = random.Random().randint(8, 10)
    time.sleep(wait_time)
    return found_keywords
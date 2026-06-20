from langchain.tools import DuckDuckGoSearchRun


search_tool = DuckDuckGoSearchRun()


def web_search(query):
    return search_tool.run(query)
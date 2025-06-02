import os
from typing import List, Dict, Any, Optional
from langchain_community.utilities import GoogleSearchAPIWrapper
# from langchain_core.documents import Document # Not strictly needed if returning dicts

class LangchainGoogleSearchRetriever:
    """
    A retriever class using LangChain's GoogleSearchAPIWrapper.
    Requires GOOGLE_API_KEY and GOOGLE_CSE_ID environment variables
    to be set if not provided directly during instantiation.
    """
    def __init__(self, k: int = 5, api_key: Optional[str] = None, cse_id: Optional[str] = None):
        """
        Initializes the LangchainGoogleSearchRetriever.

        Args:
            k (int, optional): Default number of search results to return. Defaults to 5.
            api_key (Optional[str], optional): Google API Key. If None, reads from
                                               GOOGLE_API_KEY environment variable.
            cse_id (Optional[str], optional): Google Custom Search Engine ID. If None, reads from
                                              GOOGLE_CSE_ID environment variable.

        Raises:
            ValueError: If API key or CSE ID is not provided and not found in environment variables.
        """
        self.k = k
        self.resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.resolved_cse_id = cse_id or os.getenv("GOOGLE_CSE_ID")

        if not self.resolved_api_key:
            raise ValueError("Google API Key (GOOGLE_API_KEY) not provided or found in environment variables.")
        if not self.resolved_cse_id:
            raise ValueError("Google CSE ID (GOOGLE_CSE_ID) not provided or found in environment variables.")

        self.search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=self.resolved_api_key,
            google_cse_id=self.resolved_cse_id
        )

    def search(self, query: str, k: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Performs a Google search for the given query using the underlying
        LangChain GoogleSearchAPIWrapper.

        Args:
            query (str): The search query.
            k (Optional[int]): The number of results to return.
                               If None, uses the default `k` set during initialization.

        Returns:
            List[Dict[str, str]]: A list of search results, where each result is a
                                 dictionary containing "title" and "text" (snippet).
                                 Returns an empty list if no results are found or
                                 if the input `raw_results` is None or empty.
        """
        num_to_fetch = k if k is not None else self.k

        # The .results() method of GoogleSearchAPIWrapper returns a list of dictionaries.
        # Each dictionary contains 'title', 'link', and 'snippet'.
        raw_results = self.search_wrapper.results(query=query, num_results=num_to_fetch)

        processed_results: List[Dict[str, str]] = []
        if raw_results: # raw_results is a list of dicts
            for item in raw_results:
                # Ensure item is a dictionary before calling .get, though .results() should guarantee this.
                if isinstance(item, dict):
                    processed_results.append({
                        "title": item.get("title", ""),
                        "text": item.get("snippet", "")
                        # "link": item.get("link", "") # Link can be added if needed later
                    })
                else:
                    # Log or handle unexpected item type if necessary
                    print(f"Warning: Unexpected item type in search results: {type(item)}")
        return processed_results

# Example Usage (can be removed or kept for testing)
if __name__ == '__main__':
    # To run this example, ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your environment.
    # Example:
    # export GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
    # export GOOGLE_CSE_ID="YOUR_GOOGLE_CSE_ID"

    print("Attempting to initialize LangchainGoogleSearchRetriever...")
    try:
        # Test with k=2 default results
        retriever = LangchainGoogleSearchRetriever(k=2)
        print("Retriever initialized.")

        search_query = "What is LangChain?"
        print(f"\nSearching for: '{search_query}' (expecting up to 2 results by default k)")
        results = retriever.search(search_query)
        if results:
            for i, res in enumerate(results):
                print(f"Result {i+1}:")
                print(f"  Title: {res['title']}")
                print(f"  Text: {res['text'][:100]}...") # Print first 100 chars of text
        else:
            print("No results found or an error occurred.")

        print(f"\nSearching again for: '{search_query}' but asking for k=3 results")
        results_k3 = retriever.search(search_query, k=3)
        if results_k3:
            for i, res in enumerate(results_k3):
                print(f"Result {i+1}:")
                print(f"  Title: {res['title']}")
                print(f"  Text: {res['text'][:100]}...")
            print(f"Number of results received: {len(results_k3)}")
        else:
            print("No results found or an error occurred for k=3 search.")

    except ValueError as ve:
        print(f"ValueError during initialization or search: {ve}")
        print("Please ensure GOOGLE_API_KEY and GOOGLE_CSE_ID are set in your environment.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

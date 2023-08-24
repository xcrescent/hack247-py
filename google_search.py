from googleapiclient.discovery import build

# Set your API key
api_key = "YOUR_API_KEY"

# Create a custom search service
service = build("customsearch", "v1", developerKey=api_key)

# Set the search query and parameters
query = "your search query"
num_results = 10  # Number of results to retrieve

# Perform the search
result = service.cse().list(q=query, cx="YOUR_CX_ID", num=num_results).execute()

# Extract and print URLs from search results
for item in result.get("items", []):
    print("URL:", item.get("link"))

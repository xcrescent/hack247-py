import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


def scrape_articles(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    article_containers = soup.find_all("div", class_="article-container")
    articles = []

    for container in article_containers:
        title_element = container.find("h2", class_="article-title")
        if title_element:
            title = title_element.get_text()
            link = url + title_element.find("a")["href"]
            articles.append({"title": title, "link": link})

    return articles


def scrape_data(url):
    # Send an HTTP GET request to the URL
    response = requests.get(url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        # Extracting different types of data

        # Extracting Text Data
        text_data = soup.get_text()
        print("Text Data:", text_data)  # Print first 500 characters

        # Extracting Image URLs
        image_urls = [img["src"] for img in soup.find_all("img")]
        print("Image URLs:", image_urls[:5])  # Print first 5 image URLs

        # Extracting Links
        links = [link["href"] for link in soup.find_all("a")]
        print("Links:", links[:10])  # Print first 10 links

        # Extracting Table Data
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                row_data = [cell.get_text() for cell in cells]
                print("Table Row:", row_data)
            # Extract video links
        video_links = []
        for video_tag in soup.find_all("video"):
            video_src = video_tag.get("src")
            if video_src:
                video_links.append(video_src)

        # Extract form information
        form_info = []
        for form_tag in soup.find_all("form"):
            form_name = form_tag.get("name")
            form_action = form_tag.get("action")
            form_method = form_tag.get("method")
            form_info.append({"name": form_name, "action": form_action, "method": form_method})

        # Print the extracted data
        print("Video Links:", video_links)
        print("Form Information:", form_info)

        # ... Add more code to extract other types of data (videos, forms, etc.) ...

        # Note: Be mindful of website structure and data availability

        # Note: Always respect the website's terms of use and applicable laws

    else:
        print("Failed to fetch data. Status code:", response.status_code)


# Set the maximum number of pages to crawl
max_pages = 10
index_ = 0
# Initialize a list to store visited URLs
visited_urls = []


def crawl(url, depth=index_):
    if depth >= max_pages:
        print("Maximum depth reached.")
        print("Visited URLs:", visited_urls)
        return

    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")

        # Process the page here
        # print("Crawling:", url)

        # Add the URL to the list of visited URLs
        if url not in visited_urls:
            visited_urls.append(url)

        # Find links on the page and follow them
        for link in soup.find_all("a"):
            if depth >= max_pages:
                break
            next_url = link.get("href")
            if next_url not in visited_urls:
                print("Unique URL:", next_url)
                next_url = urljoin(url, next_url)
                depth += 1
                print("Next URL:", next_url)
                crawl(next_url)


def main():
    base_url = "https://www.eliteacademy.co.in"
    page = 1
    all_articles = []

    while True:
        page_url = f"{base_url}/page/{page}"
        articles = scrape_articles(page_url)

        if not articles:
            data = scrape_data(base_url)
            print("No more articles found.")
            print(data)
            break

        all_articles.extend(articles)
        page += 1

    for article in all_articles:
        print("Title:", article["title"])
        print("Link:", article["link"])
        print("-" * 40)


if __name__ == "__main__":
    main()
    # Start crawling from the specified URL
    crawl("https://www.eliteacademy.co.in")

import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re


# =========================================================
# 1. BOOKS TO SCRAPE
# =========================================================

books = []

for page in range(1, 6):
    url = f"https://books.toscrape.com/catalogue/page-{page}.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    for book in soup.select("article.product_pod"):

        title = book.h3.a["title"]

        # Extract only the number from the price
        price_text = book.select_one(".price_color").get_text(strip=True)
        price = float(re.search(r"\d+\.\d+", price_text).group())

        availability = book.select_one(".availability").get_text(strip=True)

        rating_name = book.select_one(".star-rating")["class"][1]

        rating_values = {
            "One": 1,
            "Two": 2,
            "Three": 3,
            "Four": 4,
            "Five": 5
        }

        rating = rating_values[rating_name]

        books.append([title, price, availability, rating])


books_df = pd.DataFrame(
    books,
    columns=["Title", "Price", "Availability", "Rating"]
)

print("\n===== BOOKS =====")
print(books_df.head())

print("\n5 Least Expensive Books:")
print(books_df.nsmallest(5, "Price"))

print("\n5 Most Expensive Books:")
print(books_df.nlargest(5, "Price"))

print("\nAverage Price:")
print(books_df["Price"].mean())

books_df.to_csv("books_dataset.csv", index=False)

plt.hist(books_df["Price"])
plt.title("Distribution of Book Prices")
plt.xlabel("Price")
plt.ylabel("Number of Books")
plt.show()


# =========================================================
# 2. QUOTES TO SCRAPE
# =========================================================

quotes = []

for page in range(1, 11):

    url = f"https://quotes.toscrape.com/page/{page}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    for quote in soup.select(".quote"):

        text = quote.select_one(".text").get_text(strip=True)
        author = quote.select_one(".author").get_text(strip=True)

        tags = []
        for tag in quote.select(".tag"):
            tags.append(tag.get_text(strip=True))

        quotes.append([text, author, ", ".join(tags)])


quotes_df = pd.DataFrame(
    quotes,
    columns=["Quote", "Author", "Tags"]
)

print("\n===== QUOTES =====")
print(quotes_df.head())

print("\nNumber of Quotes per Author:")
print(quotes_df["Author"].value_counts())

all_tags = []

for tags in quotes_df["Tags"]:
    all_tags.extend(tags.split(", "))

print("\nTop 5 Tags:")
print(pd.Series(all_tags).value_counts().head(5))

longest_quote = quotes_df.loc[
    quotes_df["Quote"].str.len().idxmax()
]

print("\nLongest Quote:")
print(longest_quote["Quote"])

quotes_df.to_csv("quotes_dataset.csv", index=False)

author_counts = quotes_df["Author"].value_counts()

author_counts.plot(kind="bar")

plt.title("Number of Quotes per Author")
plt.xlabel("Author")
plt.ylabel("Number of Quotes")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# =========================================================
# 3. FRANKFURTER EXCHANGE RATE API
# =========================================================

url = "https://api.frankfurter.app/latest"

response = requests.get(url)
data = response.json()

rates = data["rates"]

exchange_df = pd.DataFrame(
    list(rates.items()),
    columns=["Currency", "Rate"]
)

print("\n===== EXCHANGE RATES =====")
print(exchange_df)

print("\n5 Highest Exchange Rates:")
print(exchange_df.nlargest(5, "Rate"))

print("\nUSD, EUR, INR, GBP and JPY:")
print(
    exchange_df[
        exchange_df["Currency"].isin(
            ["USD", "EUR", "INR", "GBP", "JPY"]
        )
    ]
)

print("\nStrongest Currency:")
print(exchange_df.loc[exchange_df["Rate"].idxmax()])

print("\nWeakest Currency:")
print(exchange_df.loc[exchange_df["Rate"].idxmin()])

exchange_df.to_csv("exchange_rates.csv", index=False)

exchange_df.plot(
    x="Currency",
    y="Rate",
    kind="bar",
    figsize=(12, 5)
)

plt.title("Exchange Rates")
plt.xlabel("Currency")
plt.ylabel("Rate")
plt.tight_layout()
plt.show()


# =========================================================
# 4. OPEN LIBRARY API
# =========================================================

url = "https://openlibrary.org/search.json?q=python"

response = requests.get(url)
data = response.json()

books = []

for book in data["docs"][:100]:

    title = book.get("title", "")

    authors = book.get("author_name", [])
    author = authors[0] if authors else ""

    year = book.get("first_publish_year", "")

    publishers = book.get("publisher", [])
    publisher = publishers[0] if publishers else ""

    isbn_list = book.get("isbn", [])
    isbn = isbn_list[0] if isbn_list else ""

    books.append([
        title,
        author,
        year,
        publisher,
        isbn
    ])


openlibrary_df = pd.DataFrame(
    books,
    columns=[
        "Title",
        "Author",
        "Publication_Year",
        "Publisher",
        "ISBN"
    ]
)

print("\n===== OPEN LIBRARY =====")
print(openlibrary_df.head())

# Convert year to numeric
openlibrary_df["Publication_Year"] = pd.to_numeric(
    openlibrary_df["Publication_Year"],
    errors="coerce"
)

print("\nBooks Published After 2020:")
print(
    openlibrary_df[
        openlibrary_df["Publication_Year"] > 2020
    ]
)

print("\nNumber of Books per Author:")
print(openlibrary_df["Author"].value_counts())

openlibrary_df.to_csv(
    "openlibrary_books.csv",
    index=False
)

print("\n10 Most Recently Published Books:")
print(
    openlibrary_df.sort_values(
        "Publication_Year",
        ascending=False
    ).head(10)
)

print("\nPublication Year Statistics:")
print(
    openlibrary_df["Publication_Year"].describe()
)

print("\n===== EXPERIMENT 5 COMPLETED =====")
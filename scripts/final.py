from github import GithubCrawler
from domain.documents import UserDocument

# Create user
user = UserDocument(id="ronelsolomon", full_name="Ronel Solomon")
user.save()

# Crawl your entire GitHub profile
crawler = GithubCrawler()
crawler.extract(link="https://github.com/ronelsolomon/", user=user)
from urllib.parse import urlparse
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from loguru import logger
from domain.documents import ArticleDocument
from application.crawlers.base import BaseCrawler


class CustomArticleCrawler(BaseCrawler):
    model = ArticleDocument
    
    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")
            return

        logger.info(f"Starting scrapping article: {link}")
        
        try:
            loader = AsyncHtmlLoader([link])
            docs = loader.load()
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)
            doc_transformed = docs_transformed[0]

            content = {
                "Title": doc_transformed.metadata.get("title"),
                "Subtitle": doc_transformed.metadata.get("description"),
                "Content": doc_transformed.page_content,
                "language": doc_transformed.metadata.get("language"),
            }

            user = kwargs["user"]
            platform = urlparse(link).netloc

            instance = self.model(
                content=content,
                link=link,
                platform=platform,
                author_id=user.id,
                author_full_name=user.full_name,
            )
            instance.save()
            logger.info(f"Finished scrapping custom article: {link}")
            
        except Exception as e:
            logger.error(f"Error scraping article {link}: {e}")
            raise
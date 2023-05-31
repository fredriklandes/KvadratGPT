from typing import Callable

import requests
from bs4 import BeautifulSoup

from ConsultantProfile import ConsultantProfile


class WebScraper:
    def __init__(self, base_url='www.kvadrat.se'):
        self.visited_urls = set()
        self.base_url = base_url
        self.paragraphs = []
        self.consultant_profiles = []

    def scrape_consultant_profile_page(self, url):
        # create new consultant profile
        consultant_profile = ConsultantProfile()
        print(f"Scraping {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # find consultant name
        consultant_name = soup.find('h1', class_='consultant-name')
        consultant_profile.name = consultant_name.text.strip().replace('\r', '')
        # find consultant title
        consultant_title = soup.find('p', class_='consultant-title')
        consultant_profile.title = consultant_title.text.strip().replace('\r', '')

        consultant_preamble = soup.find('p', class_='consultant-preamble')
        consultant_profile.preamble = consultant_preamble.text.strip().replace('\r', '')

        consultant_article = soup.find('article', class_='consultant-article')
        consultant_profile.article = consultant_article.text.strip().replace('\r', '')

        article_containers = soup.find_all('section', class_='consultant-articleContainer')
        sections = self.get_sections(article_containers)

        if 'cv' in sections:
            self.scrape_list(sections['cv'], self.scrape_cv_item, consultant_profile)
        if 'kompetens' in sections:
            self.scrape_list(sections['kompetens'], self.scrape_competence_item, consultant_profile)
        if 'utbildningar' in sections:
            self.scrape_list(sections['utbildningar'], self.scrape_education_item, consultant_profile)
        if 'anställningar' in sections:
            self.scrape_list(sections['anställningar'], self.scrape_employment_item, consultant_profile)
        self.consultant_profiles.append(consultant_profile)

    def get_sections(self, article_containers):
        dictionary = {}
        index = 0
        for element in article_containers[1:]:
            text = element.find_all('h2')[0].text.lower()
            if 'cv' in text:
                dictionary['cv'] = element
            if 'kompetensområden' in text:
                dictionary['kompetens'] = element
            if 'anställningar' in text:
                dictionary['anställningar'] = element
            if 'utbildningar' in text:
                dictionary['utbildningar'] = element
            index += 1
        return dictionary

    def scrape_education_item(self, item, consultant_profile: ConsultantProfile):
        # education = Education()
        description = item.find('h3').text.lower().strip().replace('\r', '').replace('\n', '')
        year = item.find('p').text.lower().strip().replace('\r', '').replace('\n', '')
        consultant_profile.education_list.append(f"{description} {year}")
        # education_desc = item.find('h3').text
        # print(education_desc)
        # education_year = item.find('p').text
        # print(education_year)

    def scrape_employment_item(self, item, consultant_profile: ConsultantProfile):
        # employment = Employment()
        description = item.find('h3').text.lower().strip().replace('\r', '').replace('\n', '')
        year = item.find('p').text.lower().strip().replace('\r', '').replace('\n', '')
        consultant_profile.employment_list.append(f"{description} {year}")
        # employment_desc = item.find('h3').text
        # print(employment_desc)
        # employment_year =item.find('p').text
        # print(employment_year)

    def scrape_competence_item(self, item, consultant_profile: ConsultantProfile):
        consultant_cv_content = item.select_one('[class="consultant-cvContent"]').text.strip().replace('\r', '').replace('\n', '')
        # competence = Competence()
        description = consultant_cv_content.lower().strip().replace('\r', '').replace('\n', '')
        consultant_profile.competence_list.append(description)
        # print(consultant_cv_content)

    def scrape_cv_item(self, item, consultant_profile: ConsultantProfile):
        consultant_cv_content = item.select_one('[class="consultant-cvContent"]')
        cv_description = ''
        if consultant_cv_content is not None:
            cv_description = consultant_cv_content.text.lower().strip().replace('\r', '').replace('\n', '')
        consultant_cv_position = item.select_one('[class="consultant-cvPosition"]').text.lower().strip().replace('\r', '').replace('\n', '')
        consultant_cv_header = item.select_one('[class="consultant-cvPostHeader"]').text.lower().strip().replace('\r', '').replace('\n', '')
        # cv = CV()
        description = cv_description
        position = consultant_cv_position
        header = consultant_cv_header
        consultant_profile.cv_list.append(f"{header} {position} {description}")

    def scrape_list(self, section, scrape_item: Callable, consultant_profile: ConsultantProfile):
        header = section.select('h2')
        if section is not None:
            list = section.select('ul li')
            for consultant_item in list:
                scrape_item(consultant_item, consultant_profile)

    # Web scrape pages consisting of consultant profiles and return a list of all the profiles
    def scrape_consultant_profile_pages(self, url):
        print(f"Scraping {url}")
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        links = soup.find_all('a', itemprop='url')

        for link in links:
            next_url = link.get('href')
            if next_url and next_url.startswith('http') and next_url:
                print(f"Found absolute link: {next_url}")
                self.scrape_consultant_profile_page(next_url)
            else:
                tmp = f'https://{self.base_url}/{next_url}'
                print(f"Found relative link: {tmp}")
                self.scrape_consultant_profile_page(tmp)

    def scrape_website(self, url):
        print(f"Scraping {url}")
        if url in self.visited_urls:
            return

        self.visited_urls.add(url)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Collect the text from the current web page
        self.collect_text(soup)

        # Find all links on the current web page
        links = soup.find_all('a')
        for link in links:
            next_url = link.get('href')
            if next_url and next_url.startswith('http') and next_url:
                print(f"Found absolute link: {next_url}")
                self.scrape_website(next_url)
            else:
                tmp = f'https://{self.base_url}/{next_url}'
                print(f"Found relative link: {tmp}")
                self.scrape_website(tmp)

    def collect_text(self, soup):
        # Implement your logic here for collecting text from the web page
        # For example, you can find specific elements using BeautifulSoup
        # and extract the text from those elements.

        # Example: Extract all paragraph texts
        paragraphs = soup.find_all('p')
        for paragraph in paragraphs:
            self.paragraphs.append(paragraph.text)
            print(paragraph.text)

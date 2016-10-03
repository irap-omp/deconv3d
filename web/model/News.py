from os import listdir
from os.path import isfile, join, dirname, basename, abspath
import vendor.yaml as yaml


class News:
    """
    One news. Built from YAML by NewsCollection.
    """
    def __init__(self, slug, title='', corpus=''):
        self.slug = slug
        self.title = title
        self.corpus = corpus
        self.date = self.extract_date(slug)

    def extract_date(self, slug):
        import re
        import datetime
        m = re.search(r'^\d\d\d\d-\d\d-\d\d', slug)
        return datetime.datetime.strptime(m.group(0), "%Y-%m-%d").date()


class NewsCollection:
    """
    Read news from YAML files, using the [PyYAML](http://pyyaml.org) module.
    Lazy loads the `.yml` files and memoizes their content.

    directory: string
        The directory from where to load the YAML config files.
        If you provide a relative path (no leading `/`),
        it will be relative to this __file__'s directory.
    """

    def __init__(self, directory='news'):
        self.directory = directory
        self.news_files = [f for f in listdir(directory) if isfile(join(directory, f))]
        self.news_files.sort()
        self.news_files.reverse()  # decreasing date
        self._cache = {}

    def find(self, start=0, count=10):
        """
        Returns an array of News.
        """
        found = []
        length = len(self.news_files)
        if start < 0:
            start = 0
        count = min(count, length - start)
        i = 0
        while i < count:
            found.append(self.get(self.news_files[start+i]))
            i += 1
        return found

    def get(self, filename):
        """
        The standard accessor to news.

        filename: string
            The filename to load relative to the directory provided on instantiation.

        Returns the News object stored in the YAML.
        """
        if filename not in self._cache:
            filepath = join(dirname(__file__), self.directory, filename)
            with open(filepath) as f:
                data = yaml.load(f)
                slug = basename(filepath)
                title = data['title'] if 'title' in data else ''
                corpus = data['corpus'] if 'corpus' in data else ''
                self._cache[filename] = News(slug, title=title, corpus=corpus)
        return self._cache[filename]
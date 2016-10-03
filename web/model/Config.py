import os
import vendor.yaml as yaml


class Config:
    """
    Read configuration from YAML files, using the [PyYAML](http://pyyaml.org) module.
    Lazy loads the `.yml` files and memoizes their content.

    With example `config/database.yml` as :

        default:
            username: 'TheDoctor'
            password: 'Who?'

    Use like this :

        config = Config(directory='config')
        username = config['database']['default']['username']


    directory: string
        The directory from where to load the YAML config files.
        If you provide a relative path (no leading `/`),
        it will be relative to this file's directory.
    """

    def __init__(self, directory='config'):
        self.directory = directory
        self._cache = {}

    def __getitem__(self, item):
        return self.get(item)

    def get(self, filename):
        """
        The standard accessor to config files.

            config.get('filename')
            # is equivalent to
            config['filename']

        filename: string
            The filename to load, without the `.yml` extension,
            relative to the directory provided on instantiation.

        Returns the object (usually a dict) stored in the YAML.
        """
        if filename not in self._cache:
            filepath = os.path.join(os.path.dirname(__file__), self.directory, filename+'.yml')
            with open(filepath) as f:
                self._cache[filename] = yaml.load(f)
        return self._cache[filename]

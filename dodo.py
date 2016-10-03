# This file is the tasks repository for `doit`.
# See http://pydoit.org/tasks.html#intro

# Simply run `doit` from the projects directory to build the website.

import sys
import os
from os import listdir
from os.path import isfile, join, basename, abspath
import tarfile
import shutil

from flask_frozen import Freezer

from lib import __version__


sys.path.insert(1, 'web')

from web.run import app, downloads_path


# TASKS #######################################################################

DOIT_CONFIG = {'default_tasks': ['website']}


def task_doc():
    """
    Generate the documentation webpages using sphinx.
    """
    return {
        'actions': [build_doc],
        'verbosity': 2
    }


def task_tarball():
    """
    Generate a tarball of module, ready for download.
    """
    return {
        'actions': [build_tarball],
        'verbosity': 2
    }


def task_website():
    """
    Re-build the whole website. (documentation pages included)
    """
    return {
        'actions': [build_doc, build_website],
        'verbosity': 2
    }


def task_publish():
    """
    Upload the website to the remote webserver.
    """
    return {
        'actions': [publish],
        'verbosity': 2
    }


def task_all():
    """
    Do all the tasks.
    """
    return {
        'actions': [build_doc, build_tarball, build_website, publish],
        'verbosity': 2
    }


def task_test():
    """
    Run the test suite.
    """
    return {
        'actions': ['nosetests'],
        'verbosity': 2
    }


# HELPERS #####################################################################

def filter_tarball(tarinfo):
    """
    A filter for the files we do not want in the tarball.
    Return None to exclude, or return the `tarinfo` if okay.
    """
    name = tarinfo.name
    # Remove the module directory name
    i = name.find('/')
    if i > 0:
        name = name[i+1:]
    # Exclude all these
    if name.endswith(".pyc") or \
            name.startswith('web') or \
            name.startswith('data/debug') or \
            name.startswith('build') or \
            name.startswith('.idea') or \
            name.startswith('.git'):
        return None
    return tarinfo


def make_tarball(output_filename, sources_dirnames):
    """
    Helper to make `output_filename` a tarball (.tar.gz) file from the files
    list `sources_dirnames`.
    """
    print("Generating tarball %s..." % output_filename)
    with tarfile.open(output_filename, "w:gz") as tar:
        for source_dirname in sources_dirnames:
            tar.add(source_dirname,
                    arcname=basename(source_dirname),
                    filter=filter_tarball)


# URL GENERATORS ##############################################################

# Freezer needs help to generate the downloads filenames
def downloads_filename():
    downloads_files = [f for f in listdir(downloads_path)
                       if isfile(join(downloads_path, f))]
    for tarball in downloads_files:
        yield {'filename': tarball}


# ACTIONS #####################################################################

def build_doc():
    """
    Generate static webpages from the sphinx documentation.
    """
    print("Generating sphinx documentation webpages...")
    os.chdir('doc')
    os.system('make html_silently')
    os.chdir('..')


def build_website():
    """
    Generate static webpages from the flask app.
    """
    print("Generating static webpages...")
    freezer = Freezer(app)
    freezer.register_generator(downloads_filename)
    freezer.freeze()

    # Copy sphinx documentation static pages into website's doc/
    source = abspath("doc/build/html")
    target = abspath("web/build/doc")
    print("Copy documentation from %s to %s..." % (source, target))
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)


def build_tarball():
    source = abspath(".")
    target = abspath("build/deconv3d_"+__version__+".tar.gz")
    make_tarball(target, [source])


def publish():
    print("Upload files to webserver...")
    cmd = "rsync -r --delete --protocol=29 " \
          "web/build/ deconv3d@deconv3d.irap.omp.eu:/home/deconv3d/www"
    os.system(cmd)

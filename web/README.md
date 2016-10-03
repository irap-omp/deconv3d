
The source files for [deconv3d's presentation website](http://deconv3d.irap.omp.eu).

The `build` directory holds the static (frozen) files, ready for upload.


# PRE-REQUISITES

You'll need :

- [flask](http://flask.pocoo.org/)
- [frozen-flask](https://pythonhosted.org/Frozen-Flask/)

Installation of each package individually via `pip install <package>` should work,
but you can also simply run :

    $ pip install -r requirements.txt


# OVERVIEW

This website is made using flask. Here are the points of interest :

- `build/` : the output directory of the `freeze` generator.
             These are the files to upload to the webserver.
             Do NOT edit manually.
- `config/` : holds the contents of the website, in the form of YAML files.
              These are the files you'll edit, most of the time.
- `model/` : miscellaneous models. The Model part of our MVC.
             Pretty much empty right now, except for the simple `Config` YAML reader wrapper, and `News`.
- `news/` : one YAML file per news story.
- `page/` : one MD file per page static enough to be written in markdown.
- `static/` : static assets to be copied as-is into `build/`. Javascript, CSS, images, etc.
- `vendor/` : embedded third-party python libraries. (that are too painful to install via pip)
- `view/` : jinja2 templates. The View part of our MVC. They all extend the `layout.html`.
- `run.py` : where Flask is configured and routes are defined.
             This is the Controller part of our MVC.
             Run this to launch a development webserver on http://localhost:5000


# HOW TO EDIT

Content that may change is into YAML files in the `config/` folder.
Links, page metas, and such informations are also in the YAML files.

You may also want to change the HTML (in the `view/` folder).

You can also add news in the `news` folder.
The YAML filename MUST start with a valid %Y-%m-%d date.

Edit at will.
Then, publish. (see How to publish section)


# HOW TO DEVELOP

Run:

    $ cd web
    $ python run.py

Then, hack some code.
Open http://localhost:5000 to see your changes in real-time.

This is not a mandatory step, just a convenience webserver for a fast development cycle.


# HOW TO PUBLISH

You need to re-generate the static files.

From the project's root directory, simply run:

    $ doit website

or, if you want to re-generate the distributed tarball too:

    $ doit tarball website

Then, upload the contents of `build`.

To that effect, you may use the `doit publish` task, that launches a `rsync`:

    $ doit publish

You'll of course need valid credentials for the server.

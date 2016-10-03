#!/bin/sh

# /!\ Launch this from web/
# It will re-generate the website in the `web/build/` folder
# and then rsync it with the webserver

# This script is deprecated, use `doit publish` instead.
# It does have useful SSH tunnel configuration so we're not deleting it yet.

## RSYNC WITH SERVER
echo "6. Upload files to webserver..."
# The --protocol=29 is to make up for the server's old rsync version

## INSIDE OF IRAP'S LAN
rsync -r --delete --protocol=29 build/ deconv3d@deconv3d.irap.omp.eu:/home/deconv3d/www

## OUTSIDE OF IRAP'S LAN
# Establish a tunnel (in another terminal)
# ssh -L localhost:22222:deconv3d.irap.omp.eu:22 agoutenoir@gw.irap.omp.eu
# And then synchronize
# rsync -rv -e 'ssh -p 22222' --progress --delete --protocol=29 build/ deconv3d@localhost:/home/deconv3d/www
# rsync -rv -e 'ssh -p 22222' --progress --delete --protocol=29 --exclude build/download/ build/ deconv3d@localhost:/home/deconv3d/www


echo "ALL DONE!"

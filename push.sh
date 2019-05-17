#!/bin/sh

git add .
git commit -m "$(curl https://whatthecommit.com/index.txt)"
git push
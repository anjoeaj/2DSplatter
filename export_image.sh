# sudo apt-get install imagemagick

montage -geometry +0+0 -tile 5x5 $(ls -v *.jpg) optim_grid.jpg

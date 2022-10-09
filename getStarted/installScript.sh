## installation of escript (ubuntu 20.04)
sudo apt install python3-dev python3-numpy python3-pyproj python3-gdal \
 python3-sympy python3-matplotlib python3-scipy \
 libnetcdf-cxx-legacy-dev libnetcdf-c++4-dev libnetcdf-dev \
 libboost-random-dev libboost-python-dev libboost-iostreams-dev\
 scons lsb-release libsuitesparse-dev

# Caution: escript doesn't support sympy 1.2 and higher
# please use:
pip3 install sympy==1.1

cd ~/escript
cd ~/escript/src
scons -j12 options_file=scons/templates/focal_options.py boost_libs='boost_python38'

# if you are going to install the escript in 22nd version of ubuntu, please check and using the jelly options file
scons -j1 options_file=scons/templates/focal_options.py boost_libs='boost_python310'
 
 ## installation of Yade (ubuntu 20.04)
sudo add-apt-repository ppa:yade-users/external
sudo apt-get update
git clone https://gitlab.com/guanshaoheng/trunk.git
 sudo apt install cmake git freeglut3-dev libloki-dev libboost-all-dev fakeroot \
dpkg-dev build-essential g++ python3-dev python3-ipython python3-matplotlib \
libsqlite3-dev python3-numpy python3-tk gnuplot libgts-dev python3-pygraphviz \
 libeigen3-dev python3-xlib python3-pyqt5 pyqt5-dev-tools python3-mpi4py \
python3-pyqt5.qtwebkit gtk2-engines-pixbuf python3-pyqt5.qtsvg libqglviewer-dev-qt5 \
python3-pil libjs-jquery python3-sphinx python3-git libxmu-dev libxi-dev libcgal-dev \
help2man libbz2-dev zlib1g-dev libopenblas-dev libsuitesparse-dev \
libmetis-dev python3-bibtexparser python3-future coinor-clp coinor-libclp-dev \
python3-mpmath libmpfr-dev libmpfrc++-dev libvtk6-dev

# caution: if install these on 22.04
git clone https://gitlab.com/yade-dev/trunk.git
sudo apt install libvtk7-dev

cd ~
mkdir yade
cd ~/yade
mkdir build install
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ../trunk
make -j 8
make install


# environments varibles
## python
alias python='/usr/bin/python3'

## yade
alias yade='/home/shguan/yade/install/bin/yade-2022-05-16.git-92a03fb'
export PYTHONPATH=$PYTHONPATH:/home/shguan/fem-ml-dem:/home/shguan/fem-ml-dem/FEMxDEM:/home/shguan/fem-ml-dem/FEMxML
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shguan/yade/install/lib/x86_64-linux-gnu/yade-2022-05-16.git-92a03fb

## esys-escript
export PYTHONPATH=$PYTHONPATH:/home/shguan/esys-escript
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/shguan/esys-escript/lib





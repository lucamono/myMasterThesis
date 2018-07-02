# FlexSight Core #


## Building ##

```
#!bash

cd flexsight_core
mkdir externals
cd externals
git clone https://alberto_pretto@bitbucket.org/alberto_pretto/cv_ext.git
git clone https://alberto_pretto@bitbucket.org/alberto_pretto/d2co.git
cd d2co
git checkout -b flexsight origin/flexsight
# Download here (i.e., in externals) the file spinnaker_sdk.tar.gz from https://drive.google.com/file/d/0B6Nvp-r2hOVvRHRxZkdLaXVVV2s
tar xzvf spinnaker_sdk.tar.gz
# Follow the guidelines in spinnaker_sdk/TAN2012007-Using-Linux-USB3.pdf
cd ..
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=<Release or Debug> -DCMAKE_INSTALL_PREFIX=<Path to Eigen includes, e.g. /opt/eigen/include/eigen3>
make

```

## Contact information ##

Alberto Pretto [pretto@dis.uniroma1.it](mailto:pretto@dis.uniroma1.it)
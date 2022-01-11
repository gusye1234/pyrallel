UNAME_S:=$(shell uname -s)
# echo ${UNAME_S}

ifeq ($(UNAME_S),Darwin)
    CC=g++-10
    OS_flag=-undefined dynamic_lookup
else
    CC=g++
    OS_flag=
endif
OMP_TARGET=operator_omp`python3-config --extension-suffix`



install: build
# 	@echo "Rebuild"
	python setup.py install

build: ./apply/src/omp.cpp
	python setup.py clean --all
	python setup.py build_ext --inplace

	

clean:
	python setup.py clean --all
	-rm apply/*.so

.PHONY: build install clean
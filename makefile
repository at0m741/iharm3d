# Problem to compile
PROB = torus

# Top directory of HDF5, or blank if using h5pcc
HDF5_DIR = /usr/local/opt/hdf5
# Top directory of MPI, or blank if using mpicc
MPI_DIR = /usr/local/opt/open-mpi
# Top directory of GSL, or blank if installed to system
GSL_DIR = /usr/local/opt/gsl
# System /lib equivalent (can be /usr/lib, /lib64, /usr/lib64)
SYSTEM_LIBDIR = /lib64

# Try pointing this to h5pcc on your machine, before hunting down libraries
CC=/usr/local/opt/llvm/bin/clang
# Example CFLAGS for going fast with Clang
CFLAGS = -std=gnu99 -O3 -march=native -mtune=native -flto -fopenmp -mavx2 -funroll-loops -g
MATH_LIB = -lm

# Name of the executable
EXE = harm

# Override these defaults if we know the machine we're working with
# Once you know what compiles, add it as a machine def here
MAKEFILE_PATH := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
HOST := $(shell hostname)
ifneq (,$(findstring stampede2,$(HOST)))
	-include $(MAKEFILE_PATH)/machines/stampede2.make
endif
ifneq (,$(findstring theta,$(HOST)))
        -include $(MAKEFILE_PATH)/machines/theta.make
endif
ifneq (,$(findstring frontera,$(HOST)))
        -include $(MAKEFILE_PATH)/machines/frontera.make
endif
ifneq (,$(findstring beginsbh,begins$(HOST)))
        -include $(MAKEFILE_PATH)/machines/bh.make
endif
-include $(MAKEFILE_PATH)/machines/$(HOST).make

# Everything below this should be static

## VERSION PRESERVATION ##
GIT_VERSION := $(shell cd $(MAKEFILE_PATH); git describe --dirty --always --tags)

## LINKING PARAMETERS ##

LINK = $(CC)
LDFLAGS = $(CFLAGS) -L/usr/local/opt/llvm/lib -fopenmp

HDF5_LIB = -lhdf5_hl -lhdf5
MPI_LIB = -lmpi
GSL_LIB = -lgsl -lgslcblas

## LOGIC FOR PATHS ##

CORE_DIR := $(MAKEFILE_PATH)/core/
PROB_DIR := $(MAKEFILE_PATH)/prob/$(PROB)/
VPATH = $(CORE_DIR):$(PROB_DIR)

#ARC_DIR := $(MAKEFILE_PATH)/prob/$(PROB)/build_archive/
# TODO this is I think gmake-specific
ARC_DIR := $(CURDIR)/build_archive/

SRC := $(wildcard $(CORE_DIR)/*.c) $(wildcard $(PROB_DIR)/*.c)
HEAD := $(wildcard $(CORE_DIR)/*.h) $(wildcard $(PROB_DIR)/*.h)

HEAD_ARC := $(addprefix $(ARC_DIR)/, $(notdir $(HEAD)))
OBJ := $(addprefix $(ARC_DIR)/, $(notdir $(SRC:%.c=%.o)))

INC = -I$(ARC_DIR) -I/usr/local/opt/llvm/include -I$(HDF5_DIR)/include -I$(MPI_DIR)/include -I$(GSL_DIR)/include
LIBDIR =
LIB = $(MATH_LIB) $(GSL_LIB) $(HDF5_LIB) $(MPI_LIB)

# Add HDF and MPI directories only if compiler doesn't
ifneq ($(strip $(HDF5_DIR)),)
	INC += -I$(HDF5_DIR)/include/
	LIBDIR += -L$(HDF5_DIR)/lib/
	LIB += $(HDF5_LIB)
endif
ifneq ($(strip $(MPI_DIR)),)
	INC += -I$(MPI_DIR)/include/
	LIBDIR += -L$(MPI_DIR)/lib/
	LIB += $(MPI_LIB)
endif
ifneq ($(strip $(GSL_DIR)),)
	INC += -I$(GSL_DIR)/include/
	LIBDIR += -L$(GSL_DIR)/lib/
endif
ifneq ($(strip $(SYSTEM_LIBDIR)),)
	# Prefer user libraries (above) to system
	LIBDIR += -L$(SYSTEM_LIBDIR)
endif

## TARGETS ##

.PRECIOUS: $(ARC_DIR)/$(EXE) $(ARC_DIR)/%

default: build

build: $(EXE)
	@echo -e "Completed build of prob: $(PROB)"
	@echo -e "CFLAGS: $(CFLAGS)"

debug: CFLAGS += -g -Wall -Werror
debug: CFLAGS += -DDEBUG=1
debug: build

profile: CFLAGS += -g -pg
profile: build

vtune: CFLAGS += -g -Wall -Werror
vtune: CFLAGS += -debug inline-debug-info -shared-intel
vtune: build


clean:
	@echo "Cleaning build files..."
	@rm -f $(EXE) $(OBJ)

distclean: clean
	@echo "Cleaning config files..."
	@rm -rf build_archive

archive-invalidation: distclean

$(EXE): $(ARC_DIR)/$(EXE)
	@cp $(ARC_DIR)/$(EXE) .

$(ARC_DIR)/$(EXE): $(OBJ)
	@echo -e "\tLinking $(EXE)"
	@$(LINK) $(LDFLAGS) $(OBJ) $(LIBDIR) $(LIB) -o $(ARC_DIR)/$(EXE)
	@rm $(OBJ) # This ensures full recompile

$(ARC_DIR)/%.o: $(ARC_DIR)/%.c $(HEAD_ARC)
	@echo -e "\tCompiling $(notdir $<)"
	@$(CC) $(CFLAGS) $(INC) -DGIT_VERSION=$(GIT_VERSION) -c $< -o $@

$(ARC_DIR)/%: % | $(ARC_DIR)
	@cp $< $(ARC_DIR)

$(ARC_DIR):
	@mkdir $(ARC_DIR)

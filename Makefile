# --------------------------------------------------------------------------
# VERBOSE Compile Information 
ifdef VERBOSE
        VERBOSE = true
        VERBOSE_ECHO = @ echo
        VERBOSE_SHOW =
        QUIET_ECHO = @ echo > /dev/null
else
        VERBOSE = false
        VERBOSE_ECHO = @ echo > /dev/null
        VERBOSE_SHOW = @
        QUIET_ECHO = @ echo
endif

# uncomment this to compile ISPC
COMPILE_CUDA := 1
ifeq ($(COMPILE_CUDA),1)

CUDALDFLAGS:=-L/usr/local/depot/cuda-8.0/lib64/ -lcudart
CUDALIBS := GL glut cudart
CUDALDLIBS  := $(addprefix -l, $(CUDALIBS))

NVCC=nvcc
NVCCFLAGS= -std=c++11 -O3 -m64 --gpu-architecture compute_35 -Iinclude/ -DCOMPILE_CUDA

else
CUDALDFLAGS:=
CUDALIBS       :=
CUDALDLIBS  :=

endif

# uncomment this to compile ISPC
# COMPILE_ISPC := 1
ifeq ($(COMPILE_ISPC),1)
ISPC=ispc
#ISPCFLAGS=-O2 --target=sse4-i32x8 --arch=x86-64
ISPCFLAGS=-O2 --target=avx2-i32x8 --arch=x86-64
COMMONDIR=common

TASKSYS_CXX=$(COMMONDIR)/tasksys.cpp
TASKSYS_LIB=-lpthread
TASKSYS_OBJ=$(addprefix $(BUILD)/obj/, $(subst $(COMMONDIR)/,, $(TASKSYS_CXX:.cpp=.o)))
else
TASKSYS_CXX:=
TASKSYS_LIB:=-lpthread
TASKSYS_OBJ:=
endif

# --------------------------------------------------------------------------
# BUILD directory
ifndef BUILD
    ifdef DEBUG
        BUILD := build-debug
    else
        BUILD := build
    endif
endif

# --------------------------------------------------------------------------
# Setup TinyML flags 

TINYML_CFLAGS :=

# --------------------------------------------------------------------------
# These CFLAGS assume a GNU compiler.  For other compilers, write a script
# which converts these arguments into their equivalent for that particular
# compiler.

ifndef CFLAGS
    ifdef DEBUG
        CFLAGS := -g
    else
        CFLAGS := -O3
    endif
endif

CFLAGS += -std=c++11 -Wall  -Wshadow -Wextra -Iinclude -I$(BUILD)/obj

LDFLAGS := 
MYFLAGS :=
LIBRARY :=

ifeq ($(COMPILE_ISPC),1)
MYFLAGS := $(MYFLAGS) -DCOMPILE_ISPC 
LIBRARY := $(LIBRARY) $(TASKSYS_LIB)
endif

ifeq ($(COMPILE_CUDA),1)
MYFLAGS := $(MYFLAGS) -DCOMPILE_CUDA
endif
# LIBRARY = -ls3 -lcurl -lxml2

# --------------------------------------------------------------------------
# Default targets are everything

.PHONY: all
all: tinyml

$(BUILD)/obj/%.o: src/%.cpp
	$(QUIET_ECHO) $@: Compiling object
	@ mkdir -p $(dir $(BUILD)/dep/$<)
	@ g++ $(CFLAGS) $(MYFLAGS) -M -MG -MQ $@ -DCOMPILINGDEPENDENCIES \
        -o $(BUILD)/dep/$(<:%.cpp=%.d) -c $<
	@ mkdir -p $(dir $@)
	$(VERBOSE_SHOW) g++ $(CFLAGS) $(MYFLAGS) -o $@ -c $<

# --------------------------------------------------------------------------
# CUDA Rules
ifeq ($(COMPILE_CUDA),1)

$(BUILD)/obj/%.o: src/%.cu
	$(QUIET_ECHO) $@: Compiling object
	@ mkdir -p $(dir $@)
	$(VERBOSE_SHOW)	$(NVCC) $< $(NVCCFLAGS) -c -o $@

CUDA_SRC = $(wildcard src/*/*.cu src/*.cu)
CUDA_OBJS = $(patsubst src/%.cu, $(BUILD)/obj/%.o, $(CUDA_SRC))
else
CUDA_OBJS :=
endif

# --------------------------------------------------------------------------
# ISPC Rules
ifeq ($(COMPILE_ISPC),1)
$(BUILD)/obj/operations/matrixOp.o: src/operations/matrixOp.cpp $(BUILD)/obj/operations/matrixOpISPC.h
	$(QUIET_ECHO) $@: Compiling object
	@ mkdir -p $(dir $(BUILD)/dep/$<)
	@ g++ $(CFLAGS) $(MYFLAGS) -M -MG -MQ $@ -DCOMPILINGDEPENDENCIES \
	      -o $(BUILD)/dep/$(<:%.cpp=%.d) -c $<
	@ mkdir -p $(dir $@)
	$(VERBOSE_SHOW) g++ $(CFLAGS) $(MYFLAGS) -o $@ -c $<

$(BUILD)/obj/%.o: $(COMMONDIR)/%.cpp
	$(QUIET_ECHO) $@: Compiling object
	@ mkdir -p $(dir $(BUILD)/dep/$<)
	@ g++ $(CFLAGS) $(MYFLAGS) -M -MG -MQ $@ -DCOMPILINGDEPENDENCIES \
        -o $(BUILD)/dep/$(<:%.cpp=%.d) -c $<
	@ mkdir -p $(dir $@)
	$(VERBOSE_SHOW) g++ $(CFLAGS) $(MYFLAGS) -o $@ -c $<

$(BUILD)/obj/sys.o: src/operations/matrixOp.cpp $(BUILD)/obj/operations/matrixOpISPC.h
	$(QUIET_ECHO) $@: Compiling object
	@ mkdir -p $(dir $(BUILD)/dep/$<)
	@ g++ $(CFLAGS) $(MYFLAGS) -M -MG -MQ $@ -DCOMPILINGDEPENDENCIES \
	      -o $(BUILD)/dep/$(<:%.cpp=%.d) -c $<
	@ mkdir -p $(dir $@)
	$(VERBOSE_SHOW) g++ $(CFLAGS) $(MYFLAGS) -o $@ -c $<

$(BUILD)/obj/operations/%ISPC.h $(BUILD)/obj/operations/%ISPC.o: src/operations/%.ispc
	@mkdir -p $(BUILD)/obj/operations
	echo "command is" $(ISPC)
	$(ISPC) $(ISPCFLAGS) $< -o $(BUILD)/obj/operations/$*ISPC.o -h $(BUILD)/obj/operations/$*ISPC.h

$operations/matrixOpISPC.h :$(BUILD)/obj/operations/%ISPC.h

ISPC_OBJS := $(BUILD)/obj/operations/matrixOpISPC.o
else # COMIPLE_ISPC
ISPC_OBJS :=
endif


# --------------------------------------------------------------------------
# TinyML targets

.PHONY: tinyml
tinyml : $(BUILD)/bin/tinyml

SRC = $(wildcard src/*/*.cpp src/*.cpp)
TINYML_OBJS = $(patsubst src/%.cpp, $(BUILD)/obj/%.o, $(SRC))

$(BUILD)/bin/tinyml: $(TINYML_OBJS) $(ISPC_OBJS) $(TASKSYS_OBJ) $(CUDA_OBJS)
	$(QUIET_ECHO) $@: Building executable
	@ mkdir -p $(dir $@)
	g++ -o $@ $^ $(LDFLAGS) $(CUDALDFLAGS) $(CUDALDLIBS) $(LIBRARY) 

# --------------------------------------------------------------------------
# Clean target

.PHONY: clean
clean:
	$(QUIET_ECHO) $(BUILD): Cleaning
	$(VERBOSE_SHOW) rm -rf $(BUILD)

# --------------------------------------------------------------------------
# Tests

run_mlp: tinyml
	./run.sh

-include build/dep/src/*.d
-include build/dep/src/*/*.d

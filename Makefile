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

CFLAGS += -std=c++11 -Wall  -Wshadow -Wextra -Iinclude

LDFLAGS := 
MYFLAGS :=
#	MYFLAGS = -D USE_LRU_ALGO -D NOT_DO_COMPRESS
LIBRARY :=
# LIBRARY = -ls3 -lcurl -lxml2

# --------------------------------------------------------------------------
# Default targets are everything

.PHONY: all
all: tinyml

# --------------------------------------------------------------------------
# Compile target patterns

$(BUILD)/obj/%.o: src/%.cpp
	$(QUIET_ECHO) $@: Compiling object
	@ mkdir -p $(dir $(BUILD)/dep/$<)
	@ g++ $(MYFLAGS) $(CFLAGS) -M -MG -MQ $@ -DCOMPILINGDEPENDENCIES \
        -o $(BUILD)/dep/$(<:%.c=%.d) -c $<
	@ mkdir -p $(dir $@)
	$(VERBOSE_SHOW) g++ $(MYFLAGS) $(CFLAGS) -o $@ -c $<

# --------------------------------------------------------------------------
# TinyML targets

.PHONY: tinyml
tinyml : $(BUILD)/bin/tinyml

SRC = $(wildcard src/*/*.cpp src/*.cpp)
TINYML_OBJS = $(patsubst src/%.cpp, $(BUILD)/obj/%.o, $(SRC))

$(BUILD)/bin/tinyml: $(TINYML_OBJS)
	$(QUIET_ECHO) $@: Building executable
	@ mkdir -p $(dir $@)
	$(VERBOSE_SHOW) g++ -o $@ $^ $(LDFLAGS) $(LIBRARY) 

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
